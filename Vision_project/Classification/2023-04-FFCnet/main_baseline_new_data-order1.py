from __future__ import print_function
import argparse
from typing_extensions import runtime_checkable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models import *
from utils import *
import time
import pickle
import copy
import datetime
import xlrd
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
from utils_dataloder import DataLoaderX, ppa_dataloader_order1, get_all_dataloader_order1
from utils_save_results_baseline import save_results_baseline_order1
from utils import init_metric, get_opts, print_model_parm_nums
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import BowelDataset
from model import resnet18
from torchvision import transforms
import time
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import random

from model import resnet18


def train_baseline(args,
                   model,
                   train_loader,
                   optimizer,
                   epoch):
    model.train()
    # model_2_generate.train()

    train_loss_M2_generate = AverageMeter()

    image_sequence = 0
    correct_generate = 0
    for batch_idx, (data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(train_loader):
        data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
            data_1.cuda(), data_2.cuda(), ppa_t1.cuda(), target_ppa.cuda(), feature_1.cuda(), \
            feature_2.cuda(), grad_1.cuda(), grad_2.cuda()

        loss_generate_sum = torch.FloatTensor([0.0]).cuda()
        image_sequence += data_1.size(0)
        for p in model.parameters():
            p.requires_grad = True
        # for p in model_2_generate.parameters():
        #     p.requires_grad = True

        # featuremap_1 = model_1(data_1)
        # featuremap_2 = model_1(data_2)
        # if args.model == 'MM_F':
        #     output_generate_1 = model_2_generate(featuremap_1, torch.cat(
        #         [feature_1,grad_1.float().view(data_2.size(0), -1)], 1))
        #     output_generate_2 = model_2_generate(featuremap_2, torch.cat(
        #         [feature_2, grad_2.float().view(data_2.size(0), -1)], 1))
        # else:
        #     output_generate_1 = model_2_generate(featuremap_1)
        #     output_generate_2 = model_2_generate(featuremap_2)

        output_generate_1 = model(data_1)
        output_generate_2 = model(data_2)

        if args.pred_future:  # BS
            # print(output_generate_1.size())
            # print(target_ppa.size())
            loss_resnet = F.cross_entropy(output_generate_1, target_ppa)

        else:  # train Extractor 和botong确认
            loss_resnet = (F.cross_entropy(output_generate_1, ppa_t1) +
                           F.cross_entropy(output_generate_2, target_ppa))/2
            # sample_num = 0
            # loss_resnet = torch.FloatTensor([0.0]).cuda()

            # for ii in range(data_1.size(0)):
            #     if target_ppa[ii].detach().cpu().numpy() == 0: # 针对之前没有中间数据，但ppa6=0可以保证训练数据
            #         sample_num += 2
            #         loss_resnet = torch.add(
            #             loss_resnet,
            #             F.cross_entropy(output_generate_1[ii].unsqueeze(0), torch.LongTensor([0]).cuda())
            #             + F.cross_entropy(output_generate_2[ii].unsqueeze(0), torch.LongTensor([0]).cuda()))
            #     elif int(grad_1[ii].detach().cpu().numpy()) == 1: # 如果不能把保证当前数据，只用来训练GAN
            #         sample_num += 1
            #         loss_resnet = torch.add(
            #             loss_resnet,
            #             F.cross_entropy(output_generate_1[ii].unsqueeze(0),
            #                             torch.LongTensor([0]).cuda()))  # torch.LongTensor([0]).cuda()))
            #     elif int(grad_1[ii].detach().cpu().numpy()) == 6: # cvpr已经不再使用这个分支了，通过数据集修改去除了这个分支的可能
            #         sample_num += 1
            #         loss_resnet = torch.add(
            #             loss_resnet,
            #             F.cross_entropy(output_generate_1[ii].unsqueeze(0), target_ppa[ii].unsqueeze(0)))
            #     else:
            #         pass

            #     if target_ppa[ii].detach().cpu().numpy() == 1 and int(grad_2[ii].detach().cpu().numpy()) == 6:
            #         sample_num += 1
            #         loss_resnet = torch.add(
            #             loss_resnet,
            #             F.cross_entropy(output_generate_2[ii].unsqueeze(0), target_ppa[ii].unsqueeze(0)))

            # if sample_num == 0:
            #     continue
            # loss_resnet = loss_resnet / sample_num

        res = F.softmax(output_generate_1, dim=1)[
            :, 1] - F.softmax(output_generate_2, dim=1)[:, 1] + args.delta1
        res[res < 0] = 0
        reg_loss = torch.mean(res)
        loss_generate_sum += loss_resnet + args.alpha * reg_loss

        target_ppa = target_ppa.reshape(-1, 1)
        pred_generate_1 = output_generate_1.argmax(dim=1, keepdim=True)
        correct_generate += pred_generate_1.eq(
            target_ppa.view_as(pred_generate_1)).sum().item()
        if not loss_generate_sum == 0:
            optimizer.zero_grad()
            # optimizer_M_2_generate.zero_grad()
            loss_generate_sum.backward(retain_graph=False)
            optimizer.step()
            # optimizer_M_2_generate.step()

            train_loss_M2_generate.update(
                loss_generate_sum.item(), data_1.size(0))  # add loss_MG
            args.logger.info('Model Train Epoch: {} [{}/{} ({:.0f}%)] Loss_M1: {:.6f}'.format(
                epoch, batch_idx * len(data_1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss_M2_generate.avg))
    args.logger.info('In epoch {}, acc is : {}'.format(
        epoch, correct_generate / len(train_loader.dataset)))

    loss = {
        'loss_resnet': train_loss_M2_generate.avg,
    }
    return loss


def evaluate_baseline(args,
                      model,
                      test_loader,
                      epoch):
    model.eval()
    # model_2_generate.eval()

    pred_result_generate = np.zeros((len(test_loader.dataset), 2))
    correct_generate = 0
    ppa_1 = np.zeros((len(test_loader.dataset),))
    target = np.zeros((len(test_loader.dataset),))
    pred_label_generate = np.zeros((len(test_loader.dataset), 1))
    name = []
    test_loss_generate = 0
    with torch.no_grad():
        batch_begin = 0
        for batch_idx, (data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(test_loader):
            data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
                data_1.cuda(), data_2.cuda(), ppa_t1.cuda(),  target_ppa.cuda(), feature_1.cuda(), \
                feature_2.cuda(), grad_1.cuda(), grad_2.cuda()

            # featuremap_1 = model_1(data_1)
            # if args.model == 'MM_F':
            #     output_generate_1 = model_2_generate(featuremap_1, torch.cat(
            #         [feature_1, grad_1.float().view(data_2.size(0), -1)], 1))
            # else:
            #     output_generate_1 = model_2_generate(featuremap_1)
            # sijie modify
            output_generate_1 = model(data_1)

            # SoftMax()=tensor([[0.3,0.7],[0.4,0.6],......])
            output_generate_1 = F.softmax(output_generate_1, dim=1)
            pred_generate = output_generate_1.argmax(dim=1, keepdim=True)
            # #增加ppa_t1 的判断
            # # rulebase
            # pred_generate = pred_generate.detach().cpu()
            # for i in range(len(pred_generate.size(0))):
            #     if ppa_t1[i].detach().detach().cpu().numpy()[0] == 1:
            #         pred_generate[i] = 1
            # pred_generate = pred_generate.cuda()
            # #  rulebase
            correct_generate += pred_generate.eq(
                target_ppa.view_as(pred_generate)).sum().item()
            test_loss_generate += F.cross_entropy(
                output_generate_1, target_ppa, reduction='sum').item()

            # print(output_generate_1.size())
            pred_result_generate[batch_begin:batch_begin +
                                 data_1.size(0), :] = output_generate_1.cpu().numpy()
            # print(pred_result_generate.shape)
            pred_label_generate[batch_begin:batch_begin +
                                data_1.size(0)] = pred_generate.cpu().numpy()

            ppa_1[batch_begin:batch_begin +
                  data_1.size(0)] = ppa_t1.cpu().numpy()
            target[batch_begin:batch_begin +
                   data_1.size(0)] = target_ppa.cpu().numpy()

            for i in range(data_1.size(0)):
                name.append(
                    test_loader.dataset.image_path_all_1[batch_begin + i])

            batch_begin = batch_begin + data_1.size(0)

    test_loss_generate /= len(test_loader.dataset)

    AUC_generate = sklearn.metrics.roc_auc_score(target,
                                                 pred_result_generate[:, 1])  # pred -> Softmax(model output), pred_label -> prediction
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
    # from GFnet_3d_3 import GFNet

    # Training settings
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader_list, test_loader_list = get_all_dataloader_order1(
        args)

    # model_1 = RN18_front().cuda()
    # if args.model == 'MM_F':
    #     model_2_generate = RN18_last_attr_e(num_classes=args.class_num,
    #                                       feature_dim=len(args.feature_list) + 1, dropout=args.dropout, dp=args.dp).cuda()
    # else:
    #     model_2_generate = RN18_last_e(num_classes = args.class_num, dropout=args.dropout, dp=args.dp).cuda()

    GFNet = resnet18(num_classes=args.class_num, inputchannel=3).cuda()

    print('-' * 20)
    print('-' * 20)
    print('print the number of model param :')
    print_model_parm_nums(GFNet, logger=args.logger)
    #print_model_parm_nums(model_2_generate, logger=args.logger)
    print('-' * 20)
    print('-' * 20)

    optimizer = optim.SGD([{'params': GFNet.parameters(), 'lr': args.lr,
                            'weight_decay': args.wd, 'momentum': args.momentum}])
    optimizer_M_2_generate = optim.SGD(
        [{'params': GFNet.parameters(), 'lr': args.lr2,
          'weight_decay': args.wd2, 'momentum': args.momentum}])

    full_results = {}
    args = init_metric(args)
    try:
        for epoch in range(1, args.epochs + 1):
            args.logger.info('Ours BS-order1')
            start_time = time.time()
            train_results = train_baseline(
                args, GFNet, train_loader, optimizer, epoch)
            test_results_list = []
            val_results_list = []
            for ss in range(len(val_loader_list)):
                test_results_list.append(
                    evaluate_baseline(args, GFNet, test_loader_list[ss], epoch))
                val_results_list.append(
                    evaluate_baseline(args, GFNet, val_loader_list[ss], epoch))

            adjust_learning_rate(optimizer, epoch, args)
            #adjust_learning_rate(optimizer_M_2_generate, epoch, args)

            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            if args.save_checkpoint:
                save_results_baseline_order1(
                    args,
                    GFNet,
                    GFNet,
                    train_results,
                    val_results_list,
                    test_results_list,
                    full_results,
                    optimizer,
                    optimizer_M_2_generate,
                    epoch)
    finally:
        args.logger.info('save_results_path: %s' % args.save_dir)
        args.logger.info('Ours BS-order1')
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)


if __name__ == '__main__':
    main()
