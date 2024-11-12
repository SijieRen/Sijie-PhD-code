from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models_order_2_3 import *
from models_order_2_3 import Generator_LSTM_ind as Generator
from utils import *
from utils import load_pytorch_model
import time
import pickle
import copy
import datetime
import xlrd
import torch.utils.data as data
from utils_dataloder import ppa_dataloader_order2, get_all_dataloader_order2
from utils_save_results_ours import save_results_ours_order2
from utils import init_metric, get_opts, print_model_parm_nums

def train(args,
          model_1,
          model_2_list,
          G_net_list,
          D_net,
          train_loader,
          optimizer_M_2_list,
          optimizer_G_list,
          optimizer_D,
          epoch):
    model_1.eval()
    train_loss_D = AverageMeter()
    train_loss_G = AverageMeter()
    train_loss_M2_reg_mono = AverageMeter()
    train_loss_M2_cls = AverageMeter()
    train_loss_M2_gen = AverageMeter()
    train_loss_M2_cur = AverageMeter()
    train_loss_M2 = AverageMeter()
    eps = 1e-5
    
    image_sequence = 0
    for batch_idx, (data_1, data_2, data_3, ppa_t1, ppa_t2, target_ppa, feature_1, feature_2, feature_3,
                    grad_1, grad_2, grad_3) in enumerate(train_loader):
        data_1, data_2, data_3, ppa_t1, ppa_t2, target_ppa, feature_1, feature_2, feature_3, \
        grad_1, grad_2, grad_3 = \
            data_1.cuda(), data_2.cuda(), data_3.cuda(), ppa_t1.cuda(), ppa_t2.cuda(), target_ppa.cuda(), feature_1.cuda(), \
            feature_2.cuda(), feature_3.cuda(), grad_1.cuda(), grad_2.cuda(), grad_3.cuda()
        model_1.zero_grad()
        D_net.zero_grad()
        for model_2 in model_2_list:
            model_2.zero_grad()
        for G_net in G_net_list:
            G_net.zero_grad()
        is_train_D = 1
        if batch_idx % 5 < args.D_epoch:
            is_train_D = 1
            image_sequence += data_1.size(0)
            for p in model_1.parameters():
                p.requires_grad = False
            for model_2 in model_2_list:
                for p in model_2.parameters():
                    p.requires_grad = True
            for G_net in G_net_list:
                for p in G_net.parameters():
                    p.requires_grad = False
            for p in D_net.parameters():
                p.requires_grad = True

            loss_cls = torch.FloatTensor([0.0]).cuda()
            loss_gen = torch.FloatTensor([0.0]).cuda()
            reg_mono = torch.FloatTensor([0.0]).cuda()
            loss_cur = torch.FloatTensor([0.0]).cuda()
            loss_cls_count = 0
            reg_mono_count = 0
            loss_gen_count = 0
            loss_cur_count = 0
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            featuremap_3 = model_1(data_3)
            target_ppa = target_ppa.reshape(-1, 1)

            z2 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            attr_1 = feature_1.view(data_1.size(0), -1, 1, 1)
            attr_2 = feature_2.view(data_2.size(0), -1, 1, 1)
            z_attr_1 = torch.cat((z2, attr_1), 1)
            z_attr_2 = torch.cat((z2, attr_2), 1)

            # botong adopt LSTM to implement the Generator
            # here compute mono loss and loss_gen
            for idx in range(data_1.size(0)):
                all_generated_feature = G_net_list[0](feature1=featuremap_1[idx].unsqueeze(0),
                                                      feature2=featuremap_2[idx].unsqueeze(0),
                                                      z1=z_attr_1[idx].unsqueeze(0),
                                                      z2=z_attr_2[idx].unsqueeze(0),
                                                      grad_1=grad_1[idx].detach().cpu().numpy() - 1,
                                                      grad_2=grad_2[idx].detach().cpu().numpy() - 1,
                                                      grad_3=grad_3[idx].detach().cpu().numpy() - 1,
                                                      return_all = 1)
                if idx == 0:
                    generate_feature_2_by_1 = all_generated_feature[0]
                    generate_feature_3_by_12 = all_generated_feature[1]
                else:
                    generate_feature_2_by_1 = torch.cat([generate_feature_2_by_1,
                                                          all_generated_feature[0]], dim=0)
                    generate_feature_3_by_12 = torch.cat([generate_feature_3_by_12,
                                                          all_generated_feature[1]], dim=0)

            real_loss_2_by_1 = D_net(featuremap_2).mean(0).view(1)
            fake_loss_2_by_1 = D_net(generate_feature_2_by_1).mean(0).view(1)
            
            real_loss_3_by_12 = D_net(featuremap_3).mean(0).view(1)
            fake_loss_3_by_12 = D_net(generate_feature_3_by_12).mean(0).view(1)
            
            
            loss_D = (real_loss_3_by_12 - fake_loss_3_by_12) / 2 + \
                     (real_loss_2_by_1 - fake_loss_2_by_1) / 2
            
            optimizer_D.zero_grad()
            for p in D_net.parameters():
                p.data.clamp_(-args.wcl, args.wcl)
            loss_D.backward(retain_graph=False)
            optimizer_D.step()
            
            # here feature_map_2 can replace with generate_feature_2
            P_pred_future = [
                model_2_list[1](featuremap_1, feature_1,
                                idx1=(grad_2 - grad_1 - 1).view(data_2.size(0), ).detach().cpu().numpy()),
                model_2_list[1](featuremap_1, feature_1,
                                idx1=(grad_3 - grad_1 - 1).view(data_2.size(0), ).detach().cpu().numpy()),
                model_2_list[1](featuremap_2, feature_2,
                                idx1=(grad_3 - grad_2 - 1).view(data_2.size(0), ).detach().cpu().numpy())
            ]
            P_pred = [torch.softmax(model_2_list[1](featuremap_1, feature_1,
                                                    idx1=(grad_2 - grad_1 - 1).view(
                                                        data_2.size(0), ).detach().cpu().numpy())
                                    + model_2_list[0](featuremap_2, torch.cat(
                [feature_2, grad_2.float().view(data_2.size(0), -1)], 1)), dim=1),
                      torch.softmax(model_2_list[2](torch.cat([generate_feature_3_by_12 - featuremap_1,
                                                               generate_feature_3_by_12 - featuremap_2,
                                                               (generate_feature_3_by_12 - featuremap_2) -
                                                               (featuremap_2 - featuremap_1)
                                                               ], dim=1),
                                                    torch.cat([attr_1.view(data_2.size(0), -1),
                                                               attr_2.view(data_2.size(0), -1),
                                                               (grad_2 - grad_1).float().view(data_2.size(0), -1),
                                                               (grad_3 - grad_1).float().view(data_2.size(0), -1)], dim=1)), dim=1),
                      ]
            P_pred_3_by_12 = P_pred[-2][:, 1] + P_pred[-2][:, 0] * (P_pred[-1][:, 1])
            P_pred_3_by_12 = torch.cat([1 - P_pred_3_by_12.unsqueeze(1), P_pred_3_by_12.unsqueeze(1)], dim=1)
            
            P_pred_gen = [
                model_2_list[0](featuremap_1, torch.cat(
                [feature_1,grad_1.float().view(data_2.size(0), -1)], 1)),
                model_2_list[0](featuremap_2, torch.cat(
                [feature_2, grad_2.float().view(data_2.size(0), -1)], 1)),
                model_2_list[0](generate_feature_3_by_12, None),
            ]
            
            for i in range(data_1.size(0)):
                if target_ppa[i].detach().cpu().numpy()[0] == 0:
                    loss_gen_count += 4
                    loss_gen = torch.add(loss_gen,
                                         F.nll_loss(torch.log(eps + P_pred[0][i].unsqueeze(0)), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_future[0][i].unsqueeze(0), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_future[1][i].unsqueeze(0), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_future[2][i].unsqueeze(0), target_ppa[i]))
                elif int(grad_3[i].detach().cpu().numpy()) == 6:
                    loss_gen_count += 2
                    loss_gen = torch.add(loss_gen,
                                         F.cross_entropy(P_pred_future[1][i].unsqueeze(0), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_future[2][i].unsqueeze(0), target_ppa[i])
                                         )
                if target_ppa[i].detach().cpu().numpy()[0] == 0: # 对应原来缺少中间数据部分
                    loss_cur_count += 3
                    loss_cur = torch.add(loss_cur,
                                         F.cross_entropy(P_pred_gen[0][i].unsqueeze(0), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_gen[1][i].unsqueeze(0), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_gen[2][i].unsqueeze(0), target_ppa[i])
                                         )
                elif int(grad_1[i].detach().cpu().numpy()) == 1:#### 补充了中间数据这个分支可以去掉
                    loss_cur_count += 1
                    loss_cur = torch.add(loss_cur,
                                         F.cross_entropy(eps + P_pred_gen[0][i].unsqueeze(0), 1 - target_ppa[i]))
                if target_ppa[i].detach().cpu().numpy()[0] == 1 and int(grad_3[i].detach().cpu().numpy()) == 6:
                    loss_cur_count += 1
                    loss_cur = torch.add(loss_cur,
                                         F.cross_entropy(P_pred_gen[2][i].unsqueeze(0), target_ppa[i])
                                         )
                if target_ppa[i].detach().cpu().numpy()[0] == 0:
                    loss_cls_count += 1
                    loss_cls = torch.add(loss_cls,
                                         F.nll_loss(torch.log(eps + P_pred_3_by_12[i].unsqueeze(0)), target_ppa[i])
                                         )
                elif int(grad_3[i].detach().cpu().numpy()) == 6:
                    loss_cls_count += 1
                    loss_cls = torch.add(loss_cls,
                                         F.nll_loss(torch.log(eps + P_pred_3_by_12[i].unsqueeze(0)), target_ppa[i])
                                         )
                reg_mono_count += 6
                reg_mono = torch.add(reg_mono,
                                     torch.max(F.softmax(P_pred_gen[0][i].unsqueeze(0), dim=1)[0,1] -
                                               F.softmax(P_pred_gen[1][i].unsqueeze(0), dim=1)[0,1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda()) + \
                                     torch.max(F.softmax(P_pred_gen[1][i].unsqueeze(0), dim=1)[0,1] -
                                               F.softmax(P_pred_gen[2][i].unsqueeze(0), dim=1)[0,1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda()) + \
                                     torch.max(F.softmax(P_pred_gen[1][i].unsqueeze(0), dim=1)[0, 1] -
                                               P_pred_3_by_12[i][1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda()) + \
                                     torch.max(F.softmax(P_pred_future[0][i].unsqueeze(0), dim=1)[0, 1] -
                                               F.softmax(P_pred_future[1][i].unsqueeze(0), dim=1)[0, 1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda()) + \
                                     torch.max(F.softmax(P_pred_future[0][i].unsqueeze(0), dim=1)[0, 1] -
                                               F.softmax(P_pred_future[2][i].unsqueeze(0), dim=1)[0, 1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda()) + \
                                     torch.max(F.softmax(P_pred_future[0][i].unsqueeze(0), dim=1)[0, 1] -
                                               P_pred_3_by_12[i][1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda())
                                     )
            if reg_mono_count != 0:
                reg_mono = reg_mono / reg_mono_count
            if loss_cls_count != 0:
                loss_cls = loss_cls / loss_cls_count
            if loss_gen_count != 0:
                loss_gen = loss_gen / loss_gen_count
            if loss_cur_count != 0:
                loss_cur = loss_cur / loss_cur_count
            
            loss = loss_cls + args.alpha * reg_mono + args.gamma * loss_gen + args.beta1 * loss_cur
            
            if loss.item() != 0:
                for optimizer_M_2 in optimizer_M_2_list:
                    optimizer_M_2.zero_grad()
                loss.backward(retain_graph=False)
                for optimizer_M_2 in optimizer_M_2_list:
                    optimizer_M_2.step()
            
            if reg_mono_count > 0:
                train_loss_M2_reg_mono.update(args.alpha * reg_mono.item(), reg_mono_count)
            if loss_gen_count > 0:
                train_loss_M2_gen.update(args.gamma * loss_gen.item(), loss_gen_count)
            if loss_cur_count > 0:
                train_loss_M2_cur.update(args.beta1 * loss_cur.item(), loss_cur_count)
            if loss_cls_count > 0:
                train_loss_M2_cls.update(loss_cls.item(), loss_cls_count)
                train_loss_M2.update(loss.item(), loss_cls_count)
            train_loss_D.update(loss_D.item(), data_1.size(0))
        
        if batch_idx % 5 >= args.D_epoch:
            is_train_D = 0
            image_sequence += data_1.size(0)
            for p in model_1.parameters():
                p.requires_grad = False
            for model_2 in model_2_list:
                for p in model_2.parameters():
                    p.requires_grad = True
            for G_net in G_net_list:
                for p in G_net.parameters():
                    p.requires_grad = True
            for p in D_net.parameters():
                p.requires_grad = False
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            featuremap_3 = model_1(data_3)
            target_ppa = target_ppa.reshape(-1, 1)

            z2 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            attr_1 = feature_1.view(data_1.size(0), -1, 1, 1)
            attr_2 = feature_2.view(data_2.size(0), -1, 1, 1)
            z_attr_1 = torch.cat((z2, attr_1), 1)
            z_attr_2 = torch.cat((z2, attr_2), 1)

            # botong adopt LSTM to implement the Generator
            # here compute mono loss and loss_gen
            for idx in range(data_1.size(0)):
                all_generated_feature = G_net_list[0](feature1=featuremap_1[idx].unsqueeze(0),
                                                      feature2=featuremap_2[idx].unsqueeze(0),
                                                      z1=z_attr_1[idx].unsqueeze(0),
                                                      z2=z_attr_2[idx].unsqueeze(0),
                                                      grad_1=grad_1[idx].detach().cpu().numpy() - 1,
                                                      grad_2=grad_2[idx].detach().cpu().numpy() - 1,
                                                      grad_3=grad_3[idx].detach().cpu().numpy() - 1,
                                                      return_all = 1)
                if idx == 0:
                    generate_feature_2_by_1 = all_generated_feature[0]
                    generate_feature_3_by_12 = all_generated_feature[1]
                else:
                    generate_feature_2_by_1 = torch.cat([generate_feature_2_by_1,
                                                          all_generated_feature[0]], dim=0)
                    generate_feature_3_by_12 = torch.cat([generate_feature_3_by_12,
                                                          all_generated_feature[1]], dim=0)
            loss_G = D_net(generate_feature_3_by_12).mean(0).view(1) / 2 + \
                     D_net(generate_feature_2_by_1).mean(0).view(1) / 2
                
            
            loss_cls = torch.FloatTensor([0.0]).cuda()
            loss_gen = torch.FloatTensor([0.0]).cuda()
            reg_mono = torch.FloatTensor([0.0]).cuda()
            loss_cur = torch.FloatTensor([0.0]).cuda()
            loss_cls_count = 0
            reg_mono_count = 0
            loss_gen_count = 0
            loss_cur_count = 0

            # here feature_map_2 can replace with generate_feature_2
            P_pred_future = [
                model_2_list[1](featuremap_1, feature_1,
                                idx1=(grad_2 - grad_1 - 1).view(data_2.size(0), ).detach().cpu().numpy()),
                model_2_list[1](featuremap_1, feature_1,
                                idx1=(grad_3 - grad_1 - 1).view(data_2.size(0), ).detach().cpu().numpy()),
                model_2_list[1](featuremap_2, feature_2,
                                idx1=(grad_3 - grad_2 - 1).view(data_2.size(0), ).detach().cpu().numpy())
            ]
            P_pred = [torch.softmax(model_2_list[1](featuremap_1, feature_1,
                                                    idx1=(grad_2 - grad_1 - 1).view(
                                                        data_2.size(0), ).detach().cpu().numpy())
                                    + model_2_list[0](featuremap_2, torch.cat(
                [feature_2, grad_2.float().view(data_2.size(0), -1)], 1)), dim=1),
                      torch.softmax(model_2_list[2](torch.cat([generate_feature_3_by_12 - featuremap_1,
                                                               generate_feature_3_by_12 - featuremap_2,
                                                               (generate_feature_3_by_12 - featuremap_2) -
                                                               (featuremap_2 - featuremap_1)
                                                               ], dim=1),
                                                    torch.cat([attr_1.view(data_2.size(0), -1),
                                                               attr_2.view(data_2.size(0), -1),
                                                               (grad_2 - grad_1).float().view(data_2.size(0), -1),
                                                               (grad_3 - grad_1).float().view(data_2.size(0), -1)],
                                                              dim=1)), dim=1),
                      ]
            P_pred_3_by_12 = P_pred[-2][:, 1] + P_pred[-2][:, 0] * (P_pred[-1][:, 1])
            P_pred_3_by_12 = torch.cat([1 - P_pred_3_by_12.unsqueeze(1), P_pred_3_by_12.unsqueeze(1)], dim=1)

            P_pred_gen = [
                model_2_list[0](featuremap_1, torch.cat(
                [feature_1,grad_1.float().view(data_2.size(0), -1)], 1)),
                model_2_list[0](featuremap_2, torch.cat(
                [feature_2, grad_2.float().view(data_2.size(0), -1)], 1)),
                model_2_list[0](generate_feature_3_by_12, None),
            ]

            for i in range(data_1.size(0)):
                if target_ppa[i].detach().cpu().numpy()[0] == 0:
                    loss_gen_count += 4
                    loss_gen = torch.add(loss_gen,
                                         F.nll_loss(torch.log(eps + P_pred[0][i].unsqueeze(0)), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_future[0][i].unsqueeze(0), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_future[1][i].unsqueeze(0), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_future[2][i].unsqueeze(0), target_ppa[i]))
                elif int(grad_3[i].detach().cpu().numpy()) == 6:
                    loss_gen_count += 2
                    loss_gen = torch.add(loss_gen,
                                         F.cross_entropy(P_pred_future[1][i].unsqueeze(0), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_future[2][i].unsqueeze(0), target_ppa[i])
                                         )
                if target_ppa[i].detach().cpu().numpy()[0] == 0:
                    loss_cur_count += 3
                    loss_cur = torch.add(loss_cur,
                                         F.cross_entropy(P_pred_gen[0][i].unsqueeze(0), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_gen[1][i].unsqueeze(0), target_ppa[i]) + \
                                         F.cross_entropy(P_pred_gen[2][i].unsqueeze(0), target_ppa[i])
                                         )
                elif int(grad_1[i].detach().cpu().numpy()) == 1:
                    loss_cur_count += 1
                    loss_cur = torch.add(loss_cur,
                                         F.cross_entropy(eps + P_pred_gen[0][i].unsqueeze(0), 1 - target_ppa[i]))
                if target_ppa[i].detach().cpu().numpy()[0] == 1 and int(grad_3[i].detach().cpu().numpy()) == 6:
                    loss_cur_count += 1
                    loss_cur = torch.add(loss_cur,
                                         F.cross_entropy(P_pred_gen[2][i].unsqueeze(0), target_ppa[i])
                                         )
                if target_ppa[i].detach().cpu().numpy()[0] == 0:
                    loss_cls_count += 1
                    loss_cls = torch.add(loss_cls,
                                         F.nll_loss(torch.log(eps + P_pred_3_by_12[i].unsqueeze(0)), target_ppa[i])
                                         )
                elif int(grad_3[i].detach().cpu().numpy()) == 6:
                    loss_cls_count += 1
                    loss_cls = torch.add(loss_cls,
                                         F.nll_loss(torch.log(eps + P_pred_3_by_12[i].unsqueeze(0)), target_ppa[i])
                                         )
                reg_mono_count += 6
                reg_mono = torch.add(reg_mono,
                                     torch.max(F.softmax(P_pred_gen[0][i].unsqueeze(0), dim=1)[0, 1] -
                                               F.softmax(P_pred_gen[1][i].unsqueeze(0), dim=1)[0, 1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda()) + \
                                     torch.max(F.softmax(P_pred_gen[1][i].unsqueeze(0), dim=1)[0, 1] -
                                               F.softmax(P_pred_gen[2][i].unsqueeze(0), dim=1)[0, 1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda()) + \
                                     torch.max(F.softmax(P_pred_gen[1][i].unsqueeze(0), dim=1)[0, 1] -
                                               P_pred_3_by_12[i][1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda()) + \
                                     torch.max(F.softmax(P_pred_future[0][i].unsqueeze(0), dim=1)[0, 1] -
                                               F.softmax(P_pred_future[1][i].unsqueeze(0), dim=1)[0, 1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda()) + \
                                     torch.max(F.softmax(P_pred_future[0][i].unsqueeze(0), dim=1)[0, 1] -
                                               F.softmax(P_pred_future[2][i].unsqueeze(0), dim=1)[0, 1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda()) + \
                                     torch.max(F.softmax(P_pred_future[0][i].unsqueeze(0), dim=1)[0, 1] -
                                               P_pred_3_by_12[i][1] + args.delta1,
                                               torch.FloatTensor([0.0]).cuda())
                                     )
            
            if reg_mono_count != 0:
                reg_mono = reg_mono / reg_mono_count
            if loss_cls_count != 0:
                loss_cls = loss_cls / loss_cls_count
            if loss_gen_count != 0:
                loss_gen = loss_gen / loss_gen_count
            if loss_cur_count != 0:
                loss_cur = loss_cur / loss_cur_count
            
            loss_temp = loss_cls + args.alpha * reg_mono + args.gamma * loss_gen + args.beta1 * loss_cur
            loss_G_mse = F.mse_loss(generate_feature_3_by_12, featuremap_3) / 2 + \
                         F.mse_loss(generate_feature_2_by_1, featuremap_2) / 2
            loss = loss_G + loss_temp + args.beta * loss_G_mse
            
            if loss.item() != 0:
                for optimizer_M_2 in optimizer_M_2_list:
                    optimizer_M_2.zero_grad()
                for optimizer_G in optimizer_G_list:
                    optimizer_G.zero_grad()
                loss.backward(retain_graph=False)
                for optimizer_M_2 in optimizer_M_2_list:
                    optimizer_M_2.step()
                for optimizer_G in optimizer_G_list:
                    optimizer_G.step()
            
            if reg_mono_count > 0:
                train_loss_M2_reg_mono.update(args.alpha * reg_mono.item(), reg_mono_count)
            if loss_gen_count > 0:
                train_loss_M2_gen.update(args.gamma * loss_gen.item(), loss_gen_count)
            if loss_cur_count > 0:
                train_loss_M2_cur.update(args.beta1 * loss_cur.item(), loss_cur_count)
            if loss_cls_count > 0:
                train_loss_M2_cls.update(loss_cls.item(), loss_cls_count)
                train_loss_M2.update(loss.item(), loss_cls_count)
            train_loss_G.update(loss_G.item(), 4 * data_1.size(0))
        
        args.logger.info('Model Train Epoch: {} [{}/{} ({:.0f}%)] is_train_D:{:d} loss_cls: {:.4f}, '
                         'reg_mono: {:.4f}, gen_cls: {:.4f}, cur_cls: {:.4f}, D:{:.4f}, G:{:.4f}, overall: {:.4f}'.format(
            epoch, batch_idx * len(data_1), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), is_train_D, train_loss_M2_cls.avg,
            train_loss_M2_reg_mono.avg, train_loss_M2_gen.avg, train_loss_M2_cur.avg, train_loss_D.avg,
            train_loss_G.avg, train_loss_M2.avg))
        
        args.logger.info('loss_D is real pred RA loss: {}'.format(train_loss_D.avg))
    
    loss = {
        'loss_M_current_reg': train_loss_M2_reg_mono.avg,
        'loss_M_minus': train_loss_M2_cls.avg,
        'loss': train_loss_M2.avg,
        'loss_D': train_loss_D.avg,
        'loss_G': train_loss_G.avg,
    }
    return loss


def evaluate(args,
             model_1,
             model_2_list,
             G_net_list,
             test_loader,
             epoch):
    model_1.eval()
    for model_2 in model_2_list:
        model_2.eval()
    for G_net in G_net_list:
        G_net.eval()
    
    pred_0 = np.zeros((len(test_loader.dataset), args.class_num))
    pred_1 = np.zeros((len(test_loader.dataset), args.class_num))
    pred_2 = np.zeros((len(test_loader.dataset), args.class_num))
    pred_final = np.zeros((len(test_loader.dataset), args.class_num))
    pred_gen_final = np.zeros((len(test_loader.dataset), args.class_num))
    pred_y3_cur = np.zeros((len(test_loader.dataset), args.class_num))
    pred_y3_future = np.zeros((len(test_loader.dataset), args.class_num))
    pred_res = np.zeros((len(test_loader.dataset), args.class_num))
    pred_final_fusion = np.zeros((len(test_loader.dataset), args.class_num))
    target = np.zeros((len(test_loader.dataset),))
    name = []
    
    with torch.no_grad():
        batch_begin = 0
        
        for batch_idx, (data_1, data_2, data_3,  ppa_t1, ppa_t2,target_ppa, feature_1, feature_2, feature_3,
                        grad_1, grad_2, grad_3) in enumerate(test_loader):
            data_1, data_2, data_3,  ppa_t1, ppa_t2,target_ppa, feature_1, feature_2, feature_3, grad_1, grad_2, grad_3 = \
                data_1.cuda(), data_2.cuda(), data_3.cuda(), ppa_t1.cuda(), ppa_t2.cuda(), target_ppa.cuda(), \
                feature_1.cuda(), feature_2.cuda(), feature_3.cuda(), \
                grad_1.cuda(), grad_2.cuda(), grad_3.cuda()
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)

            z2 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            attr_1 = feature_1.view(data_1.size(0), -1, 1, 1)
            attr_2 = feature_2.view(data_2.size(0), -1, 1, 1)
            z_attr_1 = torch.cat((z2, attr_1), 1)
            z_attr_2 = torch.cat((z2, attr_2), 1)

            # botong adopt LSTM to implement the Generator
            # here compute mono loss and loss_gen
            for idx in range(data_1.size(0)):
                all_generated_feature = G_net_list[0](feature1=featuremap_1[idx].unsqueeze(0),
                                                      feature2=featuremap_2[idx].unsqueeze(0),
                                                      z1=z_attr_1[idx].unsqueeze(0),
                                                      z2=z_attr_2[idx].unsqueeze(0),
                                                      grad_1=grad_1[idx].detach().cpu().numpy() - 1,
                                                      grad_2=grad_2[idx].detach().cpu().numpy() - 1,
                                                      grad_3=grad_3[idx].detach().cpu().numpy() - 1,
                                                      return_all = 1)
                if idx == 0:
                    generate_feature_3_by_12 = all_generated_feature[1]
                else:
                    generate_feature_3_by_12 = torch.cat([generate_feature_3_by_12,
                                                          all_generated_feature[1]], dim=0)
                    
            # here feature_map_2 can replace with generate_feature_2
            P_pred = [torch.softmax(model_2_list[1](featuremap_1, feature_1,
                                                    idx1=(grad_2 - grad_1 - 1).view(
                                                        data_2.size(0), ).detach().cpu().numpy())
                                    + model_2_list[0](featuremap_2, torch.cat(
                [feature_2, grad_2.float().view(data_2.size(0), -1)], 1)), dim=1),
                      torch.softmax(model_2_list[2](torch.cat([generate_feature_3_by_12 - featuremap_1,
                                                               generate_feature_3_by_12 - featuremap_2,
                                                               (generate_feature_3_by_12 - featuremap_2) -
                                                               (featuremap_2 - featuremap_1)
                                                               ], dim=1),
                                                    torch.cat([attr_1.view(data_2.size(0), -1),
                                                               attr_2.view(data_2.size(0), -1),
                                                               (grad_2 - grad_1).float().view(data_2.size(0), -1),
                                                               (grad_3 - grad_1).float().view(data_2.size(0), -1)],
                                                              dim=1)), dim=1),
                      ]
            P_pred_3_by_12 = P_pred[0][:, 1] + P_pred[0][:, 0] * (P_pred[1][:, 1])
            P_pred_3_by_12 = torch.cat([1 - P_pred_3_by_12.unsqueeze(1), P_pred_3_by_12.unsqueeze(1)], dim=1)
            
            P_pred_gen = torch.softmax(model_2_list[0](generate_feature_3_by_12, None), dim=1)
            P_pred_future = [
                torch.softmax(model_2_list[1](featuremap_1, attr_1.view(data_2.size(0), -1),
                                idx1=(grad_3 - grad_1 - 1).view(data_2.size(0), ).detach().cpu().numpy()), dim=1),
                torch.softmax(model_2_list[1](featuremap_2, attr_2.view(data_2.size(0), -1),
                                idx1=(grad_3 - grad_2 - 1).view(data_2.size(0), ).detach().cpu().numpy()), dim=1)
            ]
            
            pred_0[batch_begin:batch_begin + data_1.size(0), :] = P_pred_3_by_12.detach().cpu().numpy()
            pred_1[batch_begin:batch_begin + data_1.size(0), :] = P_pred[0].detach().cpu().numpy()
            pred_2[batch_begin:batch_begin + data_1.size(0), :] = P_pred[1].detach().cpu().numpy()

            pred_res[batch_begin:batch_begin + data_1.size(0), :] = \
                P_pred[1].detach().cpu().numpy()
            pred_final[batch_begin:batch_begin + data_1.size(0), :] = \
                P_pred_3_by_12.detach().cpu().numpy()
            pred_gen_final[batch_begin:batch_begin + data_1.size(0), :] = \
                P_pred_gen.detach().cpu().numpy()
            pred_y3_cur[batch_begin:batch_begin + data_1.size(0), :] = \
                P_pred_3_by_12.detach().cpu().numpy() / 2 + P_pred_gen.detach().cpu().numpy() / 2
            pred_y3_future[batch_begin:batch_begin + data_1.size(0), :] = \
                (P_pred_3_by_12 + P_pred_future[0] + P_pred_future[1]).detach().cpu().numpy() / 3
            pred_final_fusion[batch_begin:batch_begin + data_1.size(0), :] = \
                (P_pred_3_by_12 + P_pred_gen + P_pred_future[0] + P_pred_future[1]).detach().cpu().numpy() / 4
            target[batch_begin:batch_begin + data_1.size(0)] = target_ppa.detach().cpu().numpy()
            
            for i in range(data_1.size(0)):
                name.append(test_loader.dataset.image_path_all_1[batch_begin + i])
            
            batch_begin = batch_begin + data_1.size(0)
    
    AUC_final = sklearn.metrics.roc_auc_score(target, pred_final[:, 1])
    acc_final = sklearn.metrics.accuracy_score(target, np.argmax(pred_final, axis=1))
    
    AUC_gen = sklearn.metrics.roc_auc_score(target, pred_gen_final[:, 1])
    acc_gen = sklearn.metrics.accuracy_score(target, np.argmax(pred_gen_final, axis=1))

    AUC_res = sklearn.metrics.roc_auc_score(target, pred_res[:, 1])
    acc_res = sklearn.metrics.accuracy_score(target, np.argmax(pred_res, axis=1))

    AUC_y3_cur = sklearn.metrics.roc_auc_score(target, pred_y3_cur[:, 1])
    acc_y3_cur = sklearn.metrics.accuracy_score(target, np.argmax(pred_y3_cur, axis=1))

    AUC_y3_future = sklearn.metrics.roc_auc_score(target, pred_y3_future[:, 1])
    acc_y3_future = sklearn.metrics.accuracy_score(target, np.argmax(pred_y3_future, axis=1))
    
    AUC_average_all = sklearn.metrics.roc_auc_score(target, pred_final_fusion[:, 1])
    acc_average_all = sklearn.metrics.accuracy_score(target, np.argmax(pred_final_fusion, axis=1))
    
    args.logger.info('In epoch {} for Pred_final, AUC is {}, acc is {}.'.format(epoch, AUC_final, acc_final))
    args.logger.info('In epoch {} for Pred_generate, AUC is {}, acc is {}.'.format(epoch, AUC_gen, acc_gen))
    args.logger.info(
        'In epoch {} for Pred_fusion, AUC is {}, acc is {}'.format(epoch, AUC_average_all, acc_average_all))
    
    results = {
        'AUC_final': AUC_final,
        'acc_final': acc_final,
    
        'AUC_y3_cur': AUC_y3_cur,
        'acc_y3_cur': acc_y3_cur,
    
        'AUC_y3_future': AUC_y3_future,
        'acc_y3_future': acc_y3_future,
        
        'AUC_gen': AUC_gen,
        'acc_gen': acc_gen,
    
        'AUC_res': AUC_res,
        'acc_res': acc_res,
        
        'AUC_average_all': AUC_average_all,
        'acc_average_all': acc_average_all,
        
        'pred_0': pred_0,
        'pred_1': pred_1,
        'pred_2': pred_2,
        'pred_final': pred_final,
        'pred_gen_final': pred_gen_final,
        'pred_final_fusion': pred_final_fusion,
        
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
    
    train_loader, val_loader_list, test_loader_list = get_all_dataloader_order2(args)
    
    model_1 = RN18_front(final_tanh=args.final_tanh).cuda()
    model_2_list = []
    # for idx in range(args.order + 1):
    model_2_list.append(RN18_last_attr(num_classes=args.class_num,
                                          feature_dim=len(args.feature_list) + 1,
                                      dropout=args.dropout, dp=args.dp).cuda())
    model_2_list.append(RN18_last_attr_ind(num_classes=args.class_num,
                                           feature_dim = len(args.feature_list),
                                           concat_num=1,
                                           task_num=5).cuda())
    model_2_list.append(RN18_last_attr(num_classes=args.class_num,
                                       feature_dim = len(args.feature_list) * 2 + 2,
                                       concat_num=3,
                                       ).cuda())
    # old model
    # load_pytorch_model(model_1, r'../results/Ecur_200616_BS/2020-06-29-23-50-10_Mixed_0.010000_'
    #                             r'30.000000_size_256_ep_120_0_R_Maculae/best_val_auc_model_1.pt')

    load_pytorch_model(model_1,
                       r'../results/Ecur_211217_BS_Extractor/2024-06-21-09-21-49_Mixed_0.002000_50.000000_size_256_ep_80_0_R_Maculae/best_val_auc_model_1.pt')

    
    G_net_list = []
    if args.G_net_type == 'G_net':
        G_net_list.append(Generator(feature_num=len(args.feature_list),
                                    final_tanh=args.final_tanh,
                                    in_channel=128,
                                    RNN_hidden=args.RNN_hidden,
                                    dp=args.dp).cuda())
    elif args.G_net_type == 'U_net':
        G_net_list.append(UNet(n_channels=128,
                               n_classes=128,
                               bilinear=0,
                               feature_num=len(args.feature_list) + 1,
                               final_tanh=args.final_tanh).cuda())
        G_net_list.append(UNet(n_channels=256,
                               n_classes=128,
                               bilinear=0,
                               feature_num=len(args.feature_list) + 2,
                               final_tanh=args.final_tanh).cuda())
    

    D_net = Discriminator(128).cuda()

    print('-' * 20)
    print('-' * 20)
    print('print the number of model param :')
    print_model_parm_nums(model_1, logger=args.logger)
    print_model_parm_nums(model_2_list[0], logger=args.logger)
    print_model_parm_nums(model_2_list[1], logger=args.logger)
    print_model_parm_nums(model_2_list[2], logger=args.logger)
    print_model_parm_nums(G_net_list[0], logger=args.logger)
    print_model_parm_nums(D_net, logger=args.logger)
    print('-' * 20)
    print('-' * 20)
    
    optimizer_M_2_list = []
    optimizer_G_list = []
    for model_2 in model_2_list:
        optimizer_M_2_list.append(optim.SGD([{'params': model_2.parameters(), 'lr': args.lr2,
                                              'weight_decay': args.wd2, 'momentum': args.momentum}
                                             ]))
    for G_net in G_net_list:
        optimizer_G_list.append(
            optim.RMSprop([{'params': G_net.parameters(), 'lr': args.lr, 'weight_decay': args.wd}]))
    optimizer_D = optim.RMSprop(D_net.parameters(), lr=args.lr, weight_decay=args.wd)
    
    full_results = {}
    args = init_metric(args)
    
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_results = train(args,
                                  model_1,
                                  model_2_list,
                                  G_net_list,
                                  D_net,
                                  train_loader,
                                  optimizer_M_2_list,
                                  optimizer_G_list,
                                  optimizer_D,
                                  epoch)
            test_results_list = []
            val_results_list = []
            for ss in range(len(val_loader_list)):
                test_results_list.append(
                    evaluate(args,
                             model_1,
                             model_2_list,
                             G_net_list,
                             test_loader_list[ss],
                             epoch))
                val_results_list.append(
                    evaluate(args,
                             model_1,
                             model_2_list,
                             G_net_list,
                             val_loader_list[ss],
                             epoch))
            
            for idx in range(len(optimizer_M_2_list)):
                adjust_learning_rate(optimizer_M_2_list[idx], epoch, args)
            
            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results_ours_order2(args,
                         model_1,
                         model_2_list,
                         G_net_list,
                         D_net,
                         train_results,
                         val_results_list,
                         test_results_list,
                         full_results,
                         optimizer_M_2_list,
                         optimizer_G_list,
                         optimizer_D,
                         epoch)
    finally:
        args.logger.info('save_results_path: %s' % args.save_dir)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)


if __name__ == '__main__':
    main()
