from __future__ import print_function
import argparse
from typing_extensions import runtime_checkable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models import *
from models import RN18_last_e_extractor
from utils import *
import time
import pickle
import copy
import datetime
import xlrd
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
from utils_dataloder import DataLoaderX, ppa_dataloader_order1, get_all_dataloader_order1_Extractor
from utils_save_results_baseline import save_results_baseline_order1
from utils import init_metric, get_opts, print_model_parm_nums


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
    train_loss_curr = AverageMeter()
    train_loss_futu = AverageMeter()
    
    image_sequence = 0
    correct_generate = 0
    correct_curr = 0
    correct_future = 0
    for batch_idx, (data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(train_loader):
        data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
            data_1.cuda(), data_2.cuda(), ppa_t1.cuda(), target_ppa.cuda(), feature_1.cuda(), \
            feature_2.cuda(), grad_1.cuda(), grad_2.cuda()
        
        loss_generate_sum = torch.FloatTensor([0.0]).cuda()
        image_sequence += data_1.size(0)
        for p in model_1.parameters():
            p.requires_grad = True
        for p in model_2_generate.parameters():
            p.requires_grad = True
        
        featuremap_1 = model_1(data_1)
        featuremap_2 = model_1(data_2)
        sample_num = 0
        loss_curr = torch.FloatTensor([0.0]).cuda()
        loss_future = torch.FloatTensor([0.0]).cuda()# 保证Extractor可以提取
        # for idx in range(data_1.size(0)):
        #         all_generated_feature = G_net_list[0](feature1=featuremap_1[idx].unsqueeze(0),
        #                                               feature2=featuremap_2[idx].unsqueeze(0),
        #                                               feature3=featuremap_3[idx].unsqueeze(0),
        #                                               z=z[idx].unsqueeze(0),
        #                                               att_1=feature_1[idx].unsqueeze(0),
        #                                               att_2=feature_2[idx].unsqueeze(0),
        #                                               att_3=feature_3[idx].unsqueeze(0),
        #                                               grad_1=grad_1[idx].detach().cpu().numpy() - 1,
        #                                               grad_2=grad_2[idx].detach().cpu().numpy() - 1,
        #                                               grad_3=grad_3[idx].detach().cpu().numpy() - 1,
        #                                               grad_4=grad_4[idx].detach().cpu().numpy() - 1,
        #                                               index=idx,
        #                                               return_all=1)
        #         if idx == 0:
        #             generate_feature_2_by_1 = all_generated_feature[0]
        #             generate_feature_3_by_12 = all_generated_feature[1]
        #             generate_feature_4_by_123 = all_generated_feature[2]
        #         else:
        #             generate_feature_2_by_1 = torch.cat([generate_feature_2_by_1,
        #                                                  all_generated_feature[0]], dim=0)
        #             generate_feature_3_by_12 = torch.cat([generate_feature_3_by_12,
        #                                                   all_generated_feature[1]], dim=0)
        #             generate_feature_4_by_123 = torch.cat([generate_feature_4_by_123,
        #                                                    all_generated_feature[2]], dim=0)
        output_curr2 = model_2_generate(featuremap_2, deltaT = None)
        for batch in range(data_1.size(0)):
            output_curr1_all , output_future1_all = model_2_generate(featuremap_1[batch].unsqueeze(0), 
                                                int(grad_2[batch].detach().cpu().numpy()) - int(grad_1[batch].detach().cpu().numpy()))
            if batch == 0:
                output_curr1 = output_curr1_all
                output_future1 = output_future1_all
            else:
                output_curr1 = torch.cat([output_curr1, output_curr1_all], dim = 0)
                output_future1 = torch.cat([output_future1, output_future1_all], dim = 0)
            sample_num += 1
        # args.logger.info(output_curr1.size())
        # args.logger.info(output_curr1)
        # args.logger.info(ppa_t1.size())
        # args.logger.info(ppa_t1)
        # args.logger.info(output_curr2.size())
        # args.logger.info(output_curr2)

        loss_curr = torch.add(
                        loss_curr,
                        F.cross_entropy(output_curr1, ppa_t1)
                        )
        loss_curr = torch.add(
                        loss_curr,
                        F.cross_entropy(output_curr2, target_ppa)
                        )
        loss_future = torch.add(
                        loss_future,
                        F.cross_entropy(output_future1, target_ppa)
                        )
        
        if sample_num == 0:
            continue
        loss_curr = loss_curr / 2
        loss_future = loss_future
        
  
        
        res = F.softmax(output_curr1, dim=1)[:, 1] - F.softmax(output_curr2, dim=1)[:, 1] + args.delta1# 可以加入,尝试让Extractor多学一个前后相对的数据信息
        res[res < 0] = 0
        reg_loss = torch.mean(res)
        loss_generate_sum += loss_curr + loss_future + args.alpha * reg_loss
        
        target_ppa = target_ppa.reshape(-1, 1)
        pred_curr_1 = output_curr1.argmax(dim=1, keepdim=True)
        pred_curr_2 = output_curr2.argmax(dim=1, keepdim=True)
        pred_future_1 = output_future1.argmax(dim=1, keepdim=True)

        correct_curr += pred_curr_1.eq(ppa_t1.view_as(pred_curr_1)).sum().item()
        correct_curr += pred_curr_2.eq(target_ppa.view_as(pred_curr_2)).sum().item()
        correct_future += pred_future_1.eq(target_ppa.view_as(pred_future_1)).sum().item()

        if not loss_generate_sum == 0:
            optimizer_M_1.zero_grad()
            optimizer_M_2_generate.zero_grad()
            loss_generate_sum.backward(retain_graph=False)
            optimizer_M_1.step()
            optimizer_M_2_generate.step()
            
            train_loss_M2_generate.update(loss_generate_sum.item(), data_1.size(0))  
            train_loss_curr.update(loss_curr.item(), data_1.size(0))
            train_loss_futu.update(loss_future.item(), data_1.size(0))
            args.logger.info('Model Train Epoch: {} [{}/{} ({:.0f}%)] Loss_sum: {:.6f}'.format(
                                                                                                epoch, 
                                                                                                batch_idx * len(data_1),
                                                                                                len(train_loader.dataset),
                                                                                                100. * batch_idx / len(train_loader), 
                                                                                                train_loss_M2_generate.avg))
            args.logger.info('Loss_curr: {:.6f}, loss_future: {:.6f}'.format(train_loss_curr.avg,
                                                                            train_loss_futu.avg))
    args.logger.info('In epoch {}, curr acc is : {}'.format(epoch, correct_curr / (2 * len(train_loader.dataset))))
    args.logger.info('In epoch {}, futu acc is : {}'.format(epoch, correct_future / len(train_loader.dataset)))

    
    loss = {
        'loss_resnet': train_loss_M2_generate.avg,
    }
    return loss


def evaluate_baseline(args,
                      model_1,
                      model_2_generate,
                      test_loader,
                      epoch):
    model_1.eval()
    model_2_generate.eval()
    
    
    correct_generate = 0
    ppa_1 = np.zeros((len(test_loader.dataset),))
    target = np.zeros((len(test_loader.dataset),))
    pred_label_generate = np.zeros((len(test_loader.dataset), 1))
    pred_result_generate = np.zeros((len(test_loader.dataset), 2))
    pred_label_curr = np.zeros((len(test_loader.dataset), 1))
    pred_result_curr = np.zeros((len(test_loader.dataset), 2))
    pred_label_future = np.zeros((len(test_loader.dataset), 1))
    pred_result_future = np.zeros((len(test_loader.dataset), 2))
    name = []
    test_loss_generate = 0
    test_loss_curr = 0
    test_loss_futu = 0
    with torch.no_grad():
        batch_begin = 0
        for batch_idx, (data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(test_loader):
            data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
                data_1.cuda(), data_2.cuda(), ppa_t1.cuda(),  target_ppa.cuda(), feature_1.cuda(), \
                feature_2.cuda(), grad_1.cuda(), grad_2.cuda()
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            output_curr2 = model_2_generate(featuremap_2, deltaT = None)
            # for batch in range(data_1.size(0)):
            #     output_curr1_all , output_future1_all = model_2_generate(featuremap_1[batch].unsqueeze(0), 
            #                                         int(grad_2[batch].detach().cpu().numpy()) - int(grad_1[batch].detach().cpu().numpy()))
            #     if batch == 0:
            #         output_curr1 = output_curr1_all
            #         output_future1 = output_future1_all
            #     else:
            #         output_curr1 = torch.cat([output_curr1, output_curr1_all], dim = 0)
            #         output_future1 = torch.cat([output_future1, output_future1_all], dim = 0)
            #     sample_num += 1
            for batch in range(data_1.size(0)):
                output_curr1_all , output_future1_all = model_2_generate(featuremap_1[batch].unsqueeze(0), 
                                                    int(grad_2[batch].detach().cpu().numpy()) - int(grad_1[batch].detach().cpu().numpy()))
                if batch == 0:
                    output_curr = output_curr1_all
                    output_future = output_future1_all
                else:
                    output_curr = torch.cat([output_curr, output_curr1_all], dim = 0)
                    output_future = torch.cat([output_future, output_future1_all], dim = 0)
            # output_curr, output_future = model_2_generate(feature_1)


            output_generate_1 = (output_curr + output_future + output_curr2) / 3
            test_loss_generate += F.cross_entropy(output_generate_1, target_ppa, reduction='sum').item()
            test_loss_curr += F.cross_entropy(output_curr, ppa_t1, reduction='sum').item()
            test_loss_futu += F.cross_entropy(output_future, target_ppa, reduction='sum').item()
            
            # 计算完loss后，过Softmax转化为预测概率
            output_curr2 = F.softmax(output_curr2, dim=1)
            output_curr = F.softmax(output_curr, dim=1)
            output_future = F.softmax(output_future, dim=1)
            # output_generate_1 = F.softmax(output_generate_1, dim=1)# SoftMax()=tensor([[0.3,0.7],[0.4,0.6],......]) 
            pred_curr = output_curr.argmax(dim=1, keepdim=True)
            pred_future = output_future.argmax(dim=1, keepdim=True)
            pred_curr2 = output_curr2.argmax(dim=1, keepdim=True)

            pred_result_generate[batch_begin:batch_begin + data_1.size(0), :] = output_generate_1.cpu().numpy()
            pred_label_generate[batch_begin:batch_begin + data_1.size(0)] = pred_curr2.cpu().numpy()# duiying curr2
            
            pred_result_curr[batch_begin:batch_begin + data_1.size(0), :] = output_curr.cpu().numpy()
            pred_label_curr[batch_begin:batch_begin + data_1.size(0), :] = pred_curr.cpu().numpy()
            pred_result_future[batch_begin:batch_begin + data_1.size(0), :] = output_future.cpu().numpy()
            pred_label_future[batch_begin:batch_begin + data_1.size(0), :] = pred_future.cpu().numpy()

            ppa_1[batch_begin:batch_begin + data_1.size(0)] = ppa_t1.cpu().numpy()
            target[batch_begin:batch_begin + data_1.size(0)] = target_ppa.cpu().numpy()
            
            for i in range(data_1.size(0)):
                name.append(test_loader.dataset.image_path_all_1[batch_begin + i])
            
            batch_begin = batch_begin + data_1.size(0)
    
    test_loss_generate /= len(test_loader.dataset)
    test_loss_curr /= len(test_loader.dataset)
    test_loss_futu /= len(test_loader.dataset)
    
    AUC_curr = sklearn.metrics.roc_auc_score(ppa_1, pred_result_curr[:, 1])  # pred -> Softmax(model output), pred_label -> prediction
    acc_curr = sklearn.metrics.accuracy_score(ppa_1, np.argmax(pred_result_curr, axis=1))
    AUC_future = sklearn.metrics.roc_auc_score(target, pred_result_future[:, 1])  # pred -> Softmax(model output), pred_label -> prediction
    acc_future = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_future, axis=1))

    AUC_curr2 = sklearn.metrics.roc_auc_score(target,
                                                 pred_result_generate[:, 1])  # pred -> Softmax(model output), pred_label -> prediction
    acc_curr2 = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_generate, axis=1))
    cm_generate = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_generate, axis=1))
    sensitivity_generate = cm_generate[0, 0] / (cm_generate[0, 0] + cm_generate[0, 1])
    specificity_generate = cm_generate[1, 1] / (cm_generate[1, 0] + cm_generate[1, 1])
    
    # args.logger.info(
    #     'In epoch {} for testing, curr2 AUC is {:.4f}, acc is {:.4f}. loss is {:.6f}'.format(epoch, AUC_curr2, acc_curr2,
    #                                                                         test_loss_generate))
    # args.logger.info(
    #     'In epoch {} for testing, Curr AUC is {:.4f}, acc is {:.4f}. loss is {:.6f}'.format(epoch, AUC_curr, acc_curr,
    #                                                                         test_loss_curr))
    # args.logger.info(
    #     'In epoch {} for testing, Futu AUC is {:.4f}, acc is {:.4f}. loss is {:.6f}'.format(epoch, AUC_future, acc_future,
    #                                                                         test_loss_futu))
    
    AUC_generate = (AUC_curr2 + AUC_curr + AUC_future)/3
    acc_generate = (acc_curr2 + acc_curr + acc_future)/3
    test_loss = (test_loss_generate + test_loss_curr + test_loss_futu)/3
    args.logger.info(
        'In epoch {} for testing, Futu AUC is {:.4f}, acc is {:.4f}. loss is {:.6f}'.format(epoch, AUC_generate, acc_generate,
                                                                            test_loss))
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
    # Training settings
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader_list, test_loader_list = get_all_dataloader_order1_Extractor(args)
    
    model_1 = RN18_front().cuda()
    if args.model == 'MM_F':
        model_2_generate = RN18_last_attr_e(num_classes=args.class_num,
                                          feature_dim=len(args.feature_list) + 1, dropout=args.dropout, dp=args.dp).cuda()
    else:# Extractor
        model_2_generate = RN18_last_e_extractor(num_classes = args.class_num, dropout=args.dropout, dp=args.dp).cuda()

    print('-' * 20)
    print('-' * 20)
    print('print the number of model param :')
    print_model_parm_nums(model_1, logger=args.logger)
    print_model_parm_nums(model_2_generate, logger=args.logger)
    print('-' * 20)
    print('-' * 20)


    
    optimizer_M_1 = optim.SGD([{'params': model_1.parameters(), 'lr': args.lr2,
                                'weight_decay': args.wd2, 'momentum': args.momentum}])
    optimizer_M_2_generate = optim.SGD(
        [{'params': model_2_generate.parameters(), 'lr': args.lr2,
          'weight_decay': args.wd2, 'momentum': args.momentum}])
    
    full_results = {}
    args = init_metric(args)
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_results = train_baseline(args, model_1, model_2_generate, train_loader,
                                  optimizer_M_1, optimizer_M_2_generate, epoch)
            test_results_list = []
            val_results_list = []
            for ss in range(len(val_loader_list)):
                test_results_list.append(
                    evaluate_baseline(args, model_1, model_2_generate, test_loader_list[ss], epoch))
                val_results_list.append(
                    evaluate_baseline(args, model_1, model_2_generate, val_loader_list[ss], epoch))
            
            adjust_learning_rate(optimizer_M_1, epoch, args)
            adjust_learning_rate(optimizer_M_2_generate, epoch, args)
            
            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results_baseline_order1(
                args,
                model_1,
                model_2_generate,
                train_results,
                val_results_list,
                test_results_list,
                full_results,
                optimizer_M_1,
                optimizer_M_2_generate,
                epoch)
    finally:
        args.logger.info('save_results_path: %s' % args.save_dir)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
if __name__ == '__main__':
    main()
