from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models import *
from utils import *
from utils import init_metric, get_opts, print_model_parm_nums
import time
import pickle
import copy
import datetime
import xlrd
from utils_dataloder import DataLoaderX, ppa_dataloader_order2, get_all_dataloader_order2
from utils_save_results_baseline import save_results_baseline_order2


def train_baseline(args,
                   model_1,
                   model_2_generate,
                   train_loader,
                   optimizer_M_1,
                   optimizer_M_2_generate,
                   epoch):
    model_1.train()
    model_2_generate.train()

    train_loss_M2_generate = AverageMeter()

    image_sequence = 0
    correct_generate = 0
    for batch_idx, (data_1, data_2, data_3, ppa_t1, ppa_t2, target_ppa, feature_1, feature_2, feature_3,
                    grad_1, grad_2, grad_3) in enumerate(train_loader):
        data_1, data_2, data_3, ppa_t1, ppa_t2, target_ppa, feature_1, feature_2, feature_3, \
            grad_1, grad_2, grad_3 = \
            data_1.cuda(), data_2.cuda(), data_3.cuda(), ppa_t1.cuda(), ppa_t2.cuda(), target_ppa.cuda(), feature_1.cuda(), \
            feature_2.cuda(), feature_3.cuda(), grad_1.cuda(), grad_2.cuda(), grad_3.cuda()

        loss_generate_sum = torch.FloatTensor([0.0]).cuda()
        image_sequence += data_1.size(0)
        for p in model_1.parameters():
            p.requires_grad = True
        for p in model_2_generate.parameters():
            p.requires_grad = True

        # featuremap_1 = model_1(data_1)
        # featuremap_2 = model_1(data_2)
        if args.model == 'MM_F':
            output_generate_1 = model_2_generate(featuremap_1, torch.cat([feature_1,
                                                                          (grad_3 - grad_1).float().view(data_2.size(0), -1)], 1))
            output_generate_2 = model_2_generate(featuremap_2, torch.cat([feature_2,
                                                                          (grad_3 - grad_2).float().view(data_2.size(0), -1)], 1))
        else:
            output_generate_1 = model_1(data_1)
            output_generate_2 = model_1(data_1)

        if args.pred_future:
            loss_M_generate = F.cross_entropy(
                (output_generate_1 + output_generate_2)/2, target_ppa)

        else:
            sample_num = 0
            loss_M_generate = torch.FloatTensor([0.0]).cuda()

            for ii in range(data_1.size(0)):
                if target_ppa[ii].detach().cpu().numpy() == 0:
                    sample_num += 2
                    loss_M_generate = torch.add(
                        loss_M_generate,
                        F.cross_entropy(output_generate_1[ii].unsqueeze(
                            0), torch.LongTensor([0]).cuda())
                        + F.cross_entropy(output_generate_2[ii].unsqueeze(0), torch.LongTensor([0]).cuda()))
                elif int(grad_1[ii].detach().cpu().numpy()) == 1:
                    sample_num += 1
                    loss_M_generate = torch.add(
                        loss_M_generate,
                        F.cross_entropy(output_generate_1[ii].unsqueeze(0),
                                        torch.LongTensor([0]).cuda()))  # torch.LongTensor([0]).cuda()))
                elif int(grad_1[ii].detach().cpu().numpy()) == 6:
                    sample_num += 1
                    loss_M_generate = torch.add(
                        loss_M_generate,
                        F.cross_entropy(output_generate_1[ii].unsqueeze(0), target_ppa[ii].unsqueeze(0)))
                else:
                    pass

                if target_ppa[ii].detach().cpu().numpy() == 1 and int(grad_2[ii].detach().cpu().numpy()) == 6:
                    sample_num += 1
                    loss_M_generate = torch.add(
                        loss_M_generate,
                        F.cross_entropy(output_generate_2[ii].unsqueeze(0), target_ppa[ii].unsqueeze(0)))

            if sample_num == 0:
                continue
            loss_M_generate = loss_M_generate / sample_num

        res = F.softmax(output_generate_1, dim=1)[
            :, 1] - F.softmax(output_generate_2, dim=1)[:, 1] + args.delta1
        res[res < 0] = 0
        reg_loss = torch.mean(res)
        loss_generate_sum += loss_M_generate + args.alpha * reg_loss

        target_ppa = target_ppa.reshape(-1, 1)
        pred_generate_1 = output_generate_1.argmax(dim=1, keepdim=True)
        correct_generate += pred_generate_1.eq(
            target_ppa.view_as(pred_generate_1)).sum().item()
        if not loss_generate_sum == 0:
            optimizer_M_1.zero_grad()
            # optimizer_M_2_generate.zero_grad()
            loss_generate_sum.backward(retain_graph=False)
            optimizer_M_1.step()
            # optimizer_M_2_generate.step()

            train_loss_M2_generate.update(
                loss_generate_sum.item(), data_1.size(0))  # add loss_MG
            args.logger.info('Model Train Epoch: {} [{}/{} ({:.0f}%)] Loss_M1: {:.6f}'.format(
                epoch, batch_idx * len(data_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss_M2_generate.avg))
    args.logger.info('In epoch {}, acc is : {}'.format(
        epoch, correct_generate / len(train_loader.dataset)))

    loss = {
        'loss_M_generate': train_loss_M2_generate.avg,
    }
    return loss


def evaluate_baseline(args,
                      model_1,
                      model_2_generate,
                      test_loader,
                      epoch):
    model_1.eval()
    model_2_generate.eval()

    pred_result_generate = np.zeros((len(test_loader.dataset), 2))
    correct_generate = 0
    ppa_1 = np.zeros((len(test_loader.dataset),))
    ppa_2 = np.zeros((len(test_loader.dataset),))
    target = np.zeros((len(test_loader.dataset),))
    pred_label_generate = np.zeros((len(test_loader.dataset), 1))
    name = []
    test_loss_generate = 0
    with torch.no_grad():
        batch_begin = 0
        for batch_idx, (data_1, data_2, data_3, ppa_t1, ppa_t2, target_ppa, feature_1, feature_2, feature_3,
                        grad_1, grad_2, grad_3) in enumerate(test_loader):
            data_1, data_2, data_3, ppa_t1, ppa_t2, target_ppa, feature_1, feature_2, feature_3, \
                grad_1, grad_2, grad_3 = \
                data_1.cuda(), data_2.cuda(), data_3.cuda(), ppa_t1.cuda(), ppa_t2.cuda(), target_ppa.cuda(), feature_1.cuda(), \
                feature_2.cuda(), feature_3.cuda(), grad_1.cuda(), grad_2.cuda(), grad_3.cuda()

            # featuremap_1 = model_1(data_1)
            # featuremap_2 = model_1(data_2)
            if args.model == 'MM_F':
                output_generate_1 = model_2_generate(featuremap_1, torch.cat([feature_1,
                                                                              (grad_3 - grad_1).float().view(
                                                                                  data_2.size(0), -1)], 1))
                output_generate_1 = output_generate_1 / 2 + model_2_generate(featuremap_2, torch.cat([feature_2,
                                                                                                      (grad_3 - grad_2).float().view(
                                                                                                          data_2.size(0), -1)], 1)) / 2
            else:  # GFNET
                output_generate_1 = model_1(data_1) / 2 + model_1(data_2) / 2

            output_generate_1 = F.softmax(output_generate_1, dim=1)
            pred_generate = output_generate_1.argmax(dim=1, keepdim=True)
            # #增加ppa_t1 的判断
            # # rulebase
            # pred_generate = pred_generate.detach().cpu()
            # for i in range(len(pred_generate.size(0))):
            #     if ppa_t1[i].detach().detach().cpu().numpy()[0] == 1 or ppa_t2[i].detach().detach().cpu().numpy()[0] == 1:
            #         pred_generate[i] = 1
            # pred_generate = pred_generate.cuda()
            # #  rulebase
            correct_generate += pred_generate.eq(
                target_ppa.view_as(pred_generate)).sum().item()
            test_loss_generate += F.cross_entropy(
                output_generate_1, target_ppa, reduction='sum').item()
            pred_result_generate[batch_begin:batch_begin +
                                 data_1.size(0), :] = output_generate_1.cpu().numpy()
            pred_label_generate[batch_begin:batch_begin +
                                data_1.size(0)] = pred_generate.cpu().numpy()

            target[batch_begin:batch_begin +
                   data_1.size(0)] = target_ppa.cpu().numpy()
            ppa_1[batch_begin:batch_begin +
                  data_1.size(0)] = ppa_t1.cpu().numpy()
            ppa_2[batch_begin:batch_begin +
                  data_1.size(0)] = ppa_t2.cpu().numpy()

            for i in range(data_1.size(0)):
                name.append(
                    test_loader.dataset.image_path_all_1[batch_begin + i])

            batch_begin = batch_begin + data_1.size(0)

    test_loss_generate /= len(test_loader.dataset)

    AUC_generate = sklearn.metrics.roc_auc_score(target,
                                                 pred_result_generate[:, 1])  # pred - output, labelall-target
    acc_generate = sklearn.metrics.accuracy_score(
        target, np.argmax(pred_result_generate, axis=1))
    cm_generate = sklearn.metrics.confusion_matrix(
        target, np.argmax(pred_result_generate, axis=1))
    sensitivity_generate = cm_generate[0, 0] / \
        (cm_generate[0, 0] + cm_generate[0, 1])
    specificity_generate = cm_generate[1, 1] / \
        (cm_generate[1, 0] + cm_generate[1, 1])

    args.logger.info(
        'In epoch {} for generate, AUC is {}, acc is {}. loss is {}'.format(epoch, AUC_generate, acc_generate,
                                                                            test_loss_generate))

    results = {
        'AUC_average_all': AUC_generate,
        'acc_average_all': acc_generate,
        'sensitivity': sensitivity_generate,
        'specificity': specificity_generate,
        'pred_result': pred_result_generate,
        'pred_label': pred_label_generate,
        'loss': test_loss_generate,
        'target': target,
        'image_path': name,

    }
    return results


def main():
    from GFnet_3d_3 import GFNet
    # Training settings
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader_list, test_loader_list = get_all_dataloader_order2(
        args)

    model_1 = GFNet(img_size=(256, 256),
                    patch_size=(16, 16),
                    embed_dim=768,
                    num_classes=2,
                    in_channels=3,
                    drop_rate=0.5,
                    depth=8,
                    mlp_ratio=12.,
                    representation_size=None,
                    uniform_drop=False,
                    drop_path_rate=0.6,
                    norm_layer=False,
                    dropcls=0.25).cuda()
    # if args.model == 'MM_F':
    #     model_2_generate = RN18_last_attr_e(num_classes=args.class_num,
    #                                       feature_dim=len(args.feature_list)+1).cuda()
    # else:
    #     model_2_generate = RN18_last_e(num_classes=args.class_num).cuda()

    print('-' * 20)
    print('-' * 20)
    print('print the number of model param :')
    print_model_parm_nums(model_1, logger=args.logger)
    # print_model_parm_nums(model_2_generate, logger=args.logger)
    print('-' * 20)
    print('-' * 20)

    optimizer_M_1 = optim.SGD([{'params': model_1.parameters(), 'lr': args.lr,
                                'weight_decay': args.wd, 'momentum': args.momentum}])
    optimizer_M_2_generate = optim.SGD(
        [{'params': model_1.parameters(), 'lr': args.lr2,
          'weight_decay': args.wd2, 'momentum': args.momentum}])

    full_results = {}
    args = init_metric(args)
    try:
        for epoch in range(1, args.epochs + 1):
            args.logger.info('Ours BS-order2')
            start_time = time.time()
            train_results = train_baseline(args, model_1, model_1, train_loader,
                                           optimizer_M_1, optimizer_M_2_generate, epoch)
            test_results_list = []
            val_results_list = []
            for ss in range(len(val_loader_list)):
                test_results_list.append(
                    evaluate_baseline(args, model_1, model_1, test_loader_list[ss], epoch))
                val_results_list.append(
                    evaluate_baseline(args, model_1, model_1, val_loader_list[ss], epoch))

            adjust_learning_rate(optimizer_M_1, epoch, args)
            adjust_learning_rate(optimizer_M_2_generate, epoch, args)

            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results_baseline_order2(
                args,
                model_1,
                model_1,
                train_results,
                val_results_list,
                test_results_list,
                full_results,
                optimizer_M_1,
                optimizer_M_2_generate,
                epoch)
    finally:
        args.logger.info('save_results_path: %s' % args.save_dir)
        args.logger.info('Ours BS-order2')
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)


if __name__ == '__main__':
    main()
