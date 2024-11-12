import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from metrics.myAUC import AUCMeter
from utils.utils import AverageMeter
from metrics.accuracy import accuracy
from metrics.confu_metrics import confusion_metrics
import sklearn.metrics
import numpy as np


def validate(val_loader, model, criterion_cls, criterion_gcn, args):
    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_gcn = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    eval_auc = AUCMeter()
    end = time.time()

    eval_auc_init = AUCMeter()

    target_old = np.zeros((len(val_loader.dataset), 1))
    pred_old = np.zeros((len(val_loader.dataset), 2))
    pred_init_old = np.zeros((len(val_loader.dataset), 2))
    batch_begin = 0

    y_pred_all = torch.tensor([], dtype=torch.int64).cuda()
    y_true_all = torch.tensor([], dtype=torch.int64).cuda()
    # with torch.no_grad():
    for i, (input, target, A_1, gcn_target, m_id, h_id)  in enumerate(val_loader):

        target = target.cuda(non_blocking=True)
        target_var = target.cuda(non_blocking=True)
        input_var = input.cuda(non_blocking=True)
        gcn_target_var = gcn_target.cuda(non_blocking=True)
        m_id = m_id.cuda(non_blocking=True)
        h_id = h_id.cuda(non_blocking=True)
        A_1 = A_1.cuda(non_blocking=True).float()

        output = model(input_var, m_id, h_id, A_1, gcn_target_var, criterion_gcn, 'val')

        loss_cls = criterion_cls(output[-1], target_var)
        loss_gcn = criterion_gcn(output[-3], gcn_target_var)


        acc1, acc5 = accuracy(output[-1], target_var, topk=(1, 1))
        # AUC
        needata = output[-1]
        _, predi = needata.topk(1, 1, True, True)
        predi = predi.view(len(predi))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses_gcn.update(loss_gcn.item(), input.size(0))
        eval_auc.update(predi, target_var)
        top1.update(acc1[0], input[0].size(0))
        top5.update(acc5[0], input[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # calculate confusion matrix
        _, predicted_labels = torch.max(output[-1], dim=1)
        y_pred_all = torch.cat((y_pred_all, predicted_labels), dim=0)
        y_true_all = torch.cat((y_true_all, target), dim=0)

        target_old[batch_begin:batch_begin + input_var.size(0)] = target_var.unsqueeze(1).detach().cpu().numpy()
        pred_old[batch_begin:batch_begin + input_var.size(0)] = output[-1].detach().cpu().numpy()
        pred_init_old[batch_begin:batch_begin + input_var.size(0)] = output[-2].detach().cpu().numpy()
        batch_begin += input_var.size(0)
        
        # AUC_init
        needata_init = output[-2]
        _, predi_init = needata_init.topk(1, 1, True, True)
        predi_init = predi_init.view(len(predi))
        eval_auc_init.update(predi_init, target_var)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                    # 'AUC {AUC}\t'
                    # 'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'
                    .format(
                batch_begin, len(val_loader.dataset), batch_time=batch_time, loss_cls=losses_cls))


    # print(' * Acc@1 {top1.avg:.3f} AUC {AUC}'
    #       .format(top1=top1, AUC=eval_auc.get_auc()))
    
    # sensi, speci = confusion_metrics(y_pred_all, y_true_all)

    AUC_old = sklearn.metrics.roc_auc_score(target_old, pred_old[:, 1])
    AUC_init_old = sklearn.metrics.roc_auc_score(target_old, pred_init_old[:, 1])
    acc_old = sklearn.metrics.accuracy_score(target_old, np.argmax(pred_old, axis=1))
    cm_minus = sklearn.metrics.confusion_matrix(target_old, np.argmax(pred_old, axis=1))
    specificity_old = cm_minus[0, 0] / (cm_minus[0, 0] + cm_minus[0, 1])
    sensitivity_old = cm_minus[1, 1] / (cm_minus[1, 0] + cm_minus[1, 1])
    # print("CM in sklearn: AUC acc sensi speci", AUC_old, acc_old, sensitivity_old, specificity_old)

    # return top1.avg, eval_auc.get_auc(), sensi, speci, eval_auc_init.get_auc()
    return acc_old, AUC_old, sensitivity_old, specificity_old, AUC_init_old