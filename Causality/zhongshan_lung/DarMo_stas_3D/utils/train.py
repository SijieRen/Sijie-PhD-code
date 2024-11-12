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
import numpy as np
import sklearn.metrics




def train(train_loader, model, criterion_cls, criterion_gcn, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    losses_recon = AverageMeter()
    losses_kl = AverageMeter()
    losses_cls = AverageMeter()
    losses_gcn = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_rex_cls = 0
    loss_rex_gcn = 0
    # switch to train mode
    model.train()
    eval_auc_gcn = AUCMeter()
    eval_auc = AUCMeter()
    end = time.time()

    target_old = np.zeros((len(train_loader.dataset), 1))
    pred_old = np.zeros((len(train_loader.dataset), 2))
    batch_begin = 0
    
    y_pred_all = torch.tensor([], dtype=torch.int64).cuda()
    y_true_all = torch.tensor([], dtype=torch.int64).cuda()

    for i, (input, target, A_1, gcn_target, m_id, h_id) in enumerate(train_loader): #TODO modify the dataloader
        # measure data loading time

        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        target_var = target.cuda(non_blocking=True)
        input_var = input.cuda(non_blocking=True)
        m_id = m_id.cuda(non_blocking=True)
        h_id = h_id.cuda(non_blocking=True)
        A_1 = A_1.cuda(non_blocking=True).float()

        gcn_target_var = gcn_target.cuda(non_blocking=True)
        # print("gcn_target_var", gcn_target_var.size())


        m = 5
        h = 2
        for machine in range(m):
            if input_var[m_id==machine,:,:,:].size(0) > 1:

                output = model(input_var[m_id==machine,:,:,:], machine, 0, A_1[m_id==machine,:], gcn_target_var, criterion_gcn,'train')
                # print(target_var.size())
                loss_cls = criterion_cls(output[-1], target_var[m_id==machine])
                loss_gcn = criterion_gcn(output[-2], gcn_target_var[m_id==machine,:])
                loss_vae_dict = model.loss_function(*output, M_N=int(args.batch_size) / len(train_loader.dataset.samples))
                # loss_A2 = model.loss_A2(output[-3], A2) # add a2 optimization loss

                loss = loss_vae_dict['loss'] + int(args.para_cls) * loss_cls + int(args.para_gcn) *loss_gcn# + loss_A2
                acc1, acc5 = accuracy(output[-1], target_var[m_id==machine], topk=(1, 1))
                # AUC
                needata = output[-1]
                _, predi = needata.topk(1, 1, True, True)
                predi = predi.view(len(predi))

                losses.update(loss.item(), input[m_id==machine,:,:,:].size(0))
                losses_recon.update(loss_vae_dict['Reconstruction_Loss'].item(), input.size(0))
                losses_kl.update(loss_vae_dict['KLD'].item(), input.size(0))
                losses_cls.update(loss_cls.item(), input[m_id==machine,:,:,:].size(0))
                losses_gcn.update(loss_gcn.item(), input[m_id==machine,:,:,:].size(0))
                eval_auc.update(predi, target_var[m_id==machine])
                top1.update(acc1[0], input[m_id==machine,:,:,:][0].size(0))
                top5.update(acc5[0], input[m_id==machine,:,:,:][0].size(0))


                if type(loss_rex_cls) == int:
                    loss_rex_cls = loss_cls.detach()
                    loss_rex_gcn = loss_gcn.detach()
                    loss_cls_gcn = 0
                else:
                    cat_rex_cls = torch.cat((loss_rex_cls.unsqueeze(0).unsqueeze(1), loss_cls.detach().unsqueeze(0).unsqueeze(1)),1)
                    loss1 = torch.var(cat_rex_cls)
                    loss_rex_cls = torch.mean(cat_rex_cls)

                    cat_rex_gcn = torch.cat((loss_rex_gcn.unsqueeze(0).unsqueeze(1), loss_gcn.detach().unsqueeze(0).unsqueeze(1)), 1)
                    loss2 = torch.var(cat_rex_gcn)
                    loss_rex_gcn = torch.mean(cat_rex_gcn)
                    loss_cls_gcn = loss1 + loss2

                loss = loss + loss_cls_gcn

                # # calculate confusion matrix
                # _, predicted_labels = torch.max(output[-1], dim=1)
                # y_pred_all = torch.cat((y_pred_all, predicted_labels), dim=0)
                # y_true_all = torch.cat((y_true_all, target_var[m_id==machine]), dim=0)

                target_old[batch_begin:batch_begin + input_var[m_id==machine,:,:,:].size(0)] = target_var[m_id==machine].unsqueeze(1).detach().cpu().numpy()
                pred_old[batch_begin:batch_begin + input_var[m_id==machine,:,:,:].size(0)] = output[-1].detach().cpu().numpy()
                batch_begin += input_var[m_id==machine,:,:,:].size(0)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                  'Loss_gcn {loss_gcn.val:.4f} ({loss_gcn.avg:.4f})\t'
                  'Loss_recon {loss_recon.val:.4f} ({loss_recon.avg:.4f})\t'
                  'Loss_kl {loss_kl.val:.4f} ({loss_kl.avg:.4f})\t'
                #   'AUC {AUC}\t'
                #   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(
                epoch, batch_begin, len(train_loader.dataset), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_cls=losses_cls, loss_gcn=losses_gcn, loss_recon=losses_recon,
                loss_kl=losses_kl))
            
    # sensi, speci = confusion_metrics(y_pred_all, y_true_all)
    

    AUC_old = sklearn.metrics.roc_auc_score(target_old, pred_old[:, 1])
    acc_old = sklearn.metrics.accuracy_score(target_old, np.argmax(pred_old, axis=1))
    cm_minus = sklearn.metrics.confusion_matrix(target_old, np.argmax(pred_old, axis=1))
    specificity_old = cm_minus[0, 0] / (cm_minus[0, 0] + cm_minus[0, 1])
    sensitivity_old = cm_minus[1, 1] / (cm_minus[1, 0] + cm_minus[1, 1])

    # print("CM in Darmo: AUC acc sensi speci", eval_auc.get_auc(), top1.avg,  sensi, speci, )
    # print("CM in sklearn: AUC acc sensi speci", AUC_old, acc_old, sensitivity_old, specificity_old)

    # return top1.avg, eval_auc.get_auc(), sensi, speci
    return acc_old, AUC_old, sensitivity_old, specificity_old