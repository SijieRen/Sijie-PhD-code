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



def train_baseline(train_loader, model, criterion_cls, criterion_gcn, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    # eval_auc_gcn = AUCMeter()
    eval_auc = AUCMeter()
    end = time.time()

    target_old = np.zeros((len(train_loader.dataset), 1))
    pred_old = np.zeros((len(train_loader.dataset), 2))
    batch_begin = 0

    y_pred_all = torch.tensor([], dtype=torch.int64).cuda()
    y_true_all = torch.tensor([], dtype=torch.int64).cuda()
    # all_predicted_labels = torch.tensor([], dtype=torch.int64)

    for i, (input, target, A_1, gcn_target, m_id, h_id) in enumerate(train_loader): #TODO modify the dataloader
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        target_var = target.cuda(non_blocking=True)
        input_var = input.cuda(non_blocking=True)
        m_id = m_id.cuda(non_blocking=True)
        h_id = h_id.cuda(non_blocking=True)
        A_1 = A_1.cuda(non_blocking=True).float()


        gcn_target = gcn_target.cuda(non_blocking=True)
        # gcn_target_var = gcn_target.cuda(non_blocking=True)

        # output = model(input_var, torch.cat((A_1, gcn_target), dim=1))
        output = model(input_var, A_1)
        # print(target_var.size())
        loss_cls = criterion_cls(output, target_var)


        loss = loss_cls# + loss_A2
        acc1, acc5 = accuracy(output, target_var, topk=(1, 1))
        # AUC
        needata = output
        _, predi = needata.topk(1, 1, True, True)
        predi = predi.view(len(predi))

        losses.update(loss.item(), input.size(0))

        eval_auc.update(predi, target_var)
        top1.update(acc1[0], input[0].size(0))
        top5.update(acc5[0], input[0].size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate confusion matrix
        _, predicted_labels = torch.max(output, dim=1)
        y_pred_all = torch.cat((y_pred_all, predicted_labels), dim=0)
        y_true_all = torch.cat((y_true_all, target), dim=0)

        target_old[batch_begin:batch_begin + input_var.size(0)] = target_var.unsqueeze(1).detach().cpu().numpy()
        pred_old[batch_begin:batch_begin + input_var.size(0)] = output.detach().cpu().numpy()
        # pred_init_old[batch_begin:batch_begin + input_var.size(0)] = output[-2].detach().cpu().numpy()
        batch_begin += input_var.size(0)



        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'AUC {AUC}\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, AUC=eval_auc.get_auc(), top1=top1))
    # print(y_pred_all)
    # sensi, speci = confusion_metrics(y_pred_all, y_true_all)

    AUC_old = sklearn.metrics.roc_auc_score(target_old, pred_old[:, 1])
    # AUC_init_old = sklearn.metrics.roc_auc_score(target_old, pred_init_old[:, 1])
    acc_old = sklearn.metrics.accuracy_score(target_old, np.argmax(pred_old, axis=1))
    cm_minus = sklearn.metrics.confusion_matrix(target_old, np.argmax(pred_old, axis=1))
    specificity_old = cm_minus[0, 0] / (cm_minus[0, 0] + cm_minus[0, 1])
    sensitivity_old = cm_minus[1, 1] / (cm_minus[1, 0] + cm_minus[1, 1])
    # print("CM in sklearn: AUC acc sensi speci", AUC_old, acc_old, sensitivity_old, specificity_old)

    # return top1.avg, eval_auc.get_auc(), sensi, speci, eval_auc_init.get_auc()
    return acc_old, AUC_old, sensitivity_old, specificity_old