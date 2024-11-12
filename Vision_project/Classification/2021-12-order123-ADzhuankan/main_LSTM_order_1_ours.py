from __future__ import print_function
import argparse
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
import torch.utils.data as data
from utils_dataloder import ppa_dataloader_order1, get_all_dataloader_order1
from utils_save_results_ours import save_results_ours_order1
from utils import init_metric, get_opts, print_model_parm_nums



def train(args,
          model_1,
          model_2_generate,
          model_2_res,
          G_net,
          D_net,
          train_loader,
          optimizer_M_2_generate,
          optimizer_M_2_res,
          optimizer_G,
          optimizer_D,
          epoch):
    model_1.eval()
    train_loss_D = AverageMeter()
    train_loss_G = AverageMeter()
    train_loss_M2_reg_mono = AverageMeter()
    train_loss_M2_res = AverageMeter()
    train_loss_M2 = AverageMeter()
    train_loss_M2_gen_cls = AverageMeter()
    eps = 1e-5
    
    image_sequence = 0
    for batch_idx, (data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(train_loader):
        data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
            data_1.cuda(), data_2.cuda(), ppa_t1.cuda(), target_ppa.cuda(), feature_1.cuda(), \
            feature_2.cuda(), grad_1.cuda(), grad_2.cuda()
        
        if batch_idx % 5 < args.D_epoch:
            image_sequence += data_1.size(0)
            for p in model_1.parameters():
                p.requires_grad = False
            for p in model_2_res.parameters():
                p.requires_grad = True
            for p in G_net.parameters():
                p.requires_grad = False
            for p in D_net.parameters():#训练四次D
                p.requires_grad = True
            for p in model_2_generate.parameters():
                p.requires_grad = True
            # for p in model_ESPCN.parameters():
            # p.requires_grad = False
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            target_ppa = target_ppa.reshape(-1, 1)

            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            attr_1 = feature_1.view(data_2.size(0), -1, 1, 1)
            z1_attr_1 = torch.cat((z1, attr_1, grad_1.float().view(data_2.size(0), -1, 1, 1)), 1)
            for idx in range(data_1.size(0)):
                all_generated_feature = G_net(feature1=featuremap_1[idx].unsqueeze(0),
                                              z1=z1_attr_1[idx].unsqueeze(0),
                                              grad_1=grad_1[idx].detach().cpu().numpy() - 1,
                                              grad_2=grad_2[idx].detach().cpu().numpy() - 1,
                                              return_all = 1)
                if idx == 0:
                    generate_feature_1 = all_generated_feature[0]
                else:
                    generate_feature_1 = torch.cat([generate_feature_1,
                                                          all_generated_feature[0]], dim=0)
            
            real_loss_1 = D_net(featuremap_2, z1).mean(0).view(1)
            fake_loss_1 = D_net(generate_feature_1, z1).mean(0).view(1)
            loss_D = (real_loss_1 - fake_loss_1)
            
            optimizer_D.zero_grad()
            for p in D_net.parameters():
                p.data.clamp_(-args.wcl, args.wcl)
            loss_D.backward(retain_graph=False)
            optimizer_D.step()
            
            loss_res = torch.FloatTensor([0.0]).cuda()
            reg_mono = torch.FloatTensor([0.0]).cuda()
            loss_gen = torch.FloatTensor([0.0]).cuda()
            gen_count = 0
            loss_res_count = 0
            reg_mono_count = 0
            # model_2_generate f_curr-> 高阶model2[0] , model_2_res f_prog-> 高阶model2[2] 
            res_feature_1 = generate_feature_1 - featuremap_1
            P_current_1_t = torch.softmax(model_2_generate(featuremap_1, torch.cat(# i对应order1下t1
                [feature_1,grad_1.float().view(data_2.size(0), -1)], 1)),
                                          dim=1)
            P_current_2_T = torch.softmax(model_2_generate(featuremap_2, torch.cat(# j 对应order1下T
                [feature_2, grad_2.float().view(data_2.size(0), -1)], 1)),
                                          dim=1)
            P_residue_1_i = torch.softmax(model_2_res(res_feature_1, feature_1.view(data_2.size(0), -1),
                                                      idx1=(grad_2 - grad_1 - 1).view(
                                                          data_2.size(0), ).detach().cpu().numpy()), dim=1)
            pred_gen_1 = model_2_generate(generate_feature_1, None)
        
            for i in range(data_1.size(0)):# batch内循环
                # if int(grad_1[i].detach().cpu().numpy()) == 6 and int(grad_2[i].detach().cpu().numpy()) == 6: # 通过数据集处理这个分支已经没有了
                #     print('it should not be appear')
                #     exit(1)
                # if int(grad_1[i].detach().cpu().numpy()) == 1 and int(grad_2[i].detach().cpu().numpy()) == 6:# 这个分支是因为当时选取的ppa_1=0
                #     loss_res_count += 1
                #     reg_mono_count += 1
                #     gen_count += 1
                #     loss_res = torch.add(loss_res,
                #                          F.nll_loss(torch.log(eps + P_residue_1_i[i].unsqueeze(0)), target_ppa[i]))# ！！！ groudtruth 为0与不确定是否为0训练模式不一致
                #     reg_mono = torch.add(reg_mono, torch.max(P_current_1_t[i][1] - P_current_2_T[i][1] + args.delta1,# 这里与modelEq.(1)不一致，是否这样的效果更好或者做过全部为Eq1的对比实验
                #                                              torch.FloatTensor([0.0]).cuda()))# 对应公式4， diff
                #     loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
                
                # elif int(grad_1[i].detach().cpu().numpy()) != 1 and int(grad_2[i].detach().cpu().numpy()) == 6:###这里还需要重新修改
                
                P_yj_1 = P_current_1_t[i][1] + P_current_1_t[i][0] * P_residue_1_i[i][1]
                loss_res = torch.add(loss_res,
                                        F.nll_loss(
                                            torch.log(eps + torch.cat([1 - P_yj_1.unsqueeze(0), P_yj_1.unsqueeze(0)],# ！！！ groudtruth 为0与不确定是否为0训练模式不一致
                                                                    dim=0).unsqueeze(0)), target_ppa[i]))
                reg_mono = torch.add(reg_mono, torch.max(P_current_1_t[i][1] - P_current_2_T[i][1] + args.delta1,
                                                            torch.FloatTensor([0.0]).cuda()))
                loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
                loss_res_count += 1
                reg_mono_count += 1
                gen_count += 1
                # else:###应该是可以去掉  cvpr2021数据没有中间label，只用来训练GAN
                #     reg_mono_count += 1
                #     gen_count += 2
                #     reg_mono = torch.add(reg_mono, torch.max(P_current_1_t[i][1] - P_current_2_T[i][1] + args.delta1,
                #                                              torch.FloatTensor([0.0]).cuda()))
                #     loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
            
            if reg_mono_count != 0:
                reg_mono = reg_mono / reg_mono_count
            if loss_res_count != 0:
                loss_res = loss_res / loss_res_count
            if gen_count != 0:
                loss_gen = loss_gen / gen_count
            
            loss = args.lambda2 * loss_res + args.alpha * reg_mono + args.gamma * loss_gen
            
            if loss.item() != 0:
                optimizer_M_2_generate.zero_grad()
                optimizer_M_2_res.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_M_2_generate.step()
                optimizer_M_2_res.step()
            
            if reg_mono_count > 0:
                train_loss_M2_reg_mono.update(args.alpha * reg_mono.item(), reg_mono_count)
            if loss_res_count > 0:
                train_loss_M2_res.update(loss_res.item(), loss_res_count)
                train_loss_M2.update(loss.item(), loss_res_count)
            if gen_count > 0:
                train_loss_M2_gen_cls.update(args.gamma * loss_gen.item(), gen_count)
            train_loss_D.update(loss_D.item(), 2 * data_1.size(0))
        
        if batch_idx % 5 >= args.D_epoch:
            image_sequence += data_1.size(0)
            for p in model_1.parameters():
                p.requires_grad = False
            for p in model_2_res.parameters():
                p.requires_grad = True
            for p in G_net.parameters():#训练一次G
                p.requires_grad = True
            for p in D_net.parameters():
                p.requires_grad = False
            for p in model_2_generate.parameters():
                p.requires_grad = True
            # for p in model_ESPCN.parameters():
            # p.requires_grad = True
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            
            target_ppa = target_ppa.reshape(-1, 1)

            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            attr_1 = feature_1.view(data_2.size(0), -1, 1, 1)
            z1_attr_1 = torch.cat((z1, attr_1, grad_1.float().view(data_2.size(0), -1, 1, 1)), 1)
            for idx in range(data_1.size(0)):
                all_generated_feature = G_net(feature1=featuremap_1[idx].unsqueeze(0),
                                              z1=z1_attr_1[idx].unsqueeze(0),
                                              grad_1=grad_1[idx].detach().cpu().numpy() - 1,
                                              grad_2=grad_2[idx].detach().cpu().numpy() - 1,
                                              return_all=1)
                if idx == 0:
                    generate_feature_1 = all_generated_feature[0]
                else:
                    generate_feature_1 = torch.cat([generate_feature_1,
                                                    all_generated_feature[0]], dim=0)
            # generate_feature_2 = model_ESPCN(generate_feature_2)
            
            loss_G = D_net(generate_feature_1, z1).mean(0).view(1)
            
            loss_res = torch.FloatTensor([0.0]).cuda()
            reg_mono = torch.FloatTensor([0.0]).cuda()
            loss_gen = torch.FloatTensor([0.0]).cuda()
            gen_count = 0
            loss_res_count = 0
            reg_mono_count = 0
            res_feature_1 = generate_feature_1 - featuremap_1
            
            P_current_1_t = torch.softmax(model_2_generate(featuremap_1, torch.cat(
                [feature_1,grad_1.float().view(data_2.size(0), -1)], 1)),
                                          dim=1)
            P_current_2_T = torch.softmax(model_2_generate(featuremap_2, torch.cat(
                [feature_2, grad_2.float().view(data_2.size(0), -1)], 1)),
                                          dim=1)
            P_residue_1_i = torch.softmax(model_2_res(res_feature_1, feature_1.view(data_2.size(0), -1),
                                                      idx1=(grad_2 - grad_1 - 1).view(
                                                          data_2.size(0), ).detach().cpu().numpy()), dim=1)
            pred_gen_1 = model_2_generate(generate_feature_1, None)
            if args.train_ori_data:
                pred_cur_i = model_2_generate(featuremap_1)
                pred_cur_j = model_2_generate(featuremap_2)
            # example
            # output_cur_res_1 = P_current_1_t[:, 1] + P_current_1_t[:, 0] * P_residue_1_i[:, 1]
            # output_cur_res = torch.cat([1 - output_cur_res_1.unsqueeze(1), output_cur_res_1.unsqueeze(1)], dim=1)
            # 确认后使用
            # P_yj_1 = P_current_1_t[:, 1] + P_current_1_t[:, 0] * P_residue_1_i[:, 1]# 通过our method 预测的y为1的probability
            # loss_res = torch.add(loss_res,
            #                         F.nll_loss(
            #                             torch.log(eps + torch.cat([1 - P_yj_1.unsqueeze(1), P_yj_1.unsqueeze(1)],
            #                                                     dim=1)), target_ppa[i]))
            # reg_mono = torch.add(reg_mono, torch.max(P_current_1_t[:, 1] - P_current_2_T[:, 1] + args.delta1,
            #                                             torch.FloatTensor([0.0]).cuda()))
            # loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1, target_ppa))


            for i in range(data_1.size(0)):
                # if int(grad_1[i].detach().cpu().numpy()) == 6 and int(grad_2[i].detach().cpu().numpy()) == 6:
                #     print('it should not be appear')
                #     exit(1)
                # if int(grad_1[i].detach().cpu().numpy()) == 1 and int(grad_2[i].detach().cpu().numpy()) == 6:
                #     loss_res_count += 1
                #     reg_mono_count += 1
                #     gen_count += 1
                #     loss_res = torch.add(loss_res,
                #                          F.nll_loss(torch.log(eps + P_residue_1_i[i].unsqueeze(0)), target_ppa[i]))
                #     reg_mono = torch.add(reg_mono, torch.max(P_current_1_t[i][1] - P_current_2_T[i][1] + args.delta1,
                #                                              torch.FloatTensor([0.0]).cuda()))
                #     loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
                #     if args.train_ori_data:
                #         if target_ppa[i] == 0:
                #             gen_count += 2
                #             loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_i[i].unsqueeze(0), target_ppa[i]))
                #             loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_j[i].unsqueeze(0), target_ppa[i]))
                #         else:
                #             gen_count += 2
                #             loss_gen = torch.add(loss_gen,
                #                                  F.cross_entropy(pred_cur_i[i].unsqueeze(0), 1 - target_ppa[i]))
                #             loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_j[i].unsqueeze(0), target_ppa[i]))
                
                # elif int(grad_1[i].detach().cpu().numpy()) != 1 and int(grad_2[i].detach().cpu().numpy()) == 6:
                loss_res_count += 1
                reg_mono_count += 1
                gen_count += 1
                P_yj_1 = P_current_1_t[i][1] + P_current_1_t[i][0] * P_residue_1_i[i][1]# 通过our method 预测的y为1的probability
                loss_res = torch.add(loss_res,
                                        F.nll_loss(
                                            torch.log(eps + torch.cat([1 - P_yj_1.unsqueeze(0), P_yj_1.unsqueeze(0)],
                                                                    dim=0).unsqueeze(0)), target_ppa[i]))
                reg_mono = torch.add(reg_mono, torch.max(P_current_1_t[i][1] - P_current_2_T[i][1] + args.delta1,
                                                            torch.FloatTensor([0.0]).cuda()))
                loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
                
                if args.train_ori_data:
                    if target_ppa[i] == 0:
                        gen_count += 2
                        loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_i[i].unsqueeze(0), target_ppa[i]))
                        loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_j[i].unsqueeze(0), target_ppa[i]))
                    else:
                        gen_count += 1
                        loss_gen = torch.add(loss_gen, F.cross_entropy(pred_cur_j[i].unsqueeze(0), target_ppa[i]))
                # else:
                #     reg_mono_count += 1
                #     gen_count += 1
                #     reg_mono = torch.add(reg_mono, torch.max(P_current_1_t[i][1] - P_current_2_T[i][1] + args.delta1,
                #                                              torch.FloatTensor([0.0]).cuda()))
                #     loss_gen = torch.add(loss_gen, F.cross_entropy(pred_gen_1[i].unsqueeze(0), target_ppa[i]))
            
            if reg_mono_count != 0:
                reg_mono = reg_mono / reg_mono_count
            if loss_res_count != 0:
                loss_res = loss_res / loss_res_count
            if gen_count != 0:
                loss_gen = loss_gen / gen_count
            
            loss_temp = args.lambda2 * loss_res + args.alpha * reg_mono + args.gamma * loss_gen
            loss_G_mse = args.beta * F.mse_loss(generate_feature_1, featuremap_2)
            loss = loss_G + loss_temp + loss_G_mse
            
            optimizer_M_2_generate.zero_grad()
            optimizer_M_2_res.zero_grad()
            optimizer_G.zero_grad()
            # optimizer_ESPCN.zero_grad()
            loss.backward(retain_graph=False)
            # optimizer_ESPCN.step()
            optimizer_G.step()
            optimizer_M_2_generate.step()
            optimizer_M_2_res.step()
            
            if reg_mono_count > 0:
                train_loss_M2_reg_mono.update(args.alpha * reg_mono.item(), reg_mono_count)
            if loss_res_count > 0:
                train_loss_M2_res.update(loss_res.item(), loss_res_count)
                train_loss_M2.update(loss.item(), loss_res_count)
            if gen_count > 0:
                train_loss_M2_gen_cls.update(args.gamma * loss_gen.item(), gen_count)
            train_loss_G.update(loss_G.item(), data_1.size(0))
        
        args.logger.info('Model Train Epoch: {} [{}/{} ({:.0f}%)] loss_res: {:.6f}, '
                         'reg_mono: {:.6f}, gen_cls: {:.6f}, overall: {:.6f}'.format(
            epoch, batch_idx * len(data_1), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), train_loss_M2_res.avg,
            train_loss_M2_reg_mono.avg, train_loss_M2_gen_cls.avg, train_loss_M2.avg))
        
        args.logger.info('loss_D is real pred RA loss: {}'.format(train_loss_D.avg))
    
    loss = {
        'loss_M_current_reg': train_loss_M2_reg_mono.avg,
        'loss_M_minus': train_loss_M2_res.avg,
        'loss_M_gen_cls': train_loss_M2_gen_cls.avg,
        'loss': train_loss_M2.avg,
        'loss_D': train_loss_D.avg,
        'loss_G': train_loss_G.avg,
    }
    return loss


def evaluate(args,
             model_1,
             model_2_generate,
             model_2_res,
             G_net,
             # model_ESPCN,
             test_loader,
             epoch):
    model_1.eval()
    model_2_generate.eval()
    model_2_res.eval()
    G_net.eval()
    # model_ESPCN.eval()
    
    pred_result_current = np.zeros((len(test_loader.dataset), args.class_num))
    pred_result_minus = np.zeros((len(test_loader.dataset), args.class_num))
    correct_generate = 0
    correct_minus = 0
    target = np.zeros((len(test_loader.dataset),))
    pred_label_generate = np.zeros((len(test_loader.dataset), 1))
    pred_label_minus = np.zeros((len(test_loader.dataset), 1))
    name = []
    test_loss_generate = 0
    test_loss_minus = 0
    
    pred_result_cur_res = np.zeros((len(test_loader.dataset), args.class_num))
    pred_label_cur_res = np.zeros((len(test_loader.dataset), 1))
    
    pred_result_average_all = np.zeros((len(test_loader.dataset), args.class_num))
    pred_label_average_all = np.zeros((len(test_loader.dataset), 1))
    
    pred_result_gen = np.zeros((len(test_loader.dataset), args.class_num))
    pred_label_gen = np.zeros((len(test_loader.dataset), 1))
    with torch.no_grad():
        batch_begin = 0
        
        for batch_idx, (data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(test_loader):
            data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
                data_1.cuda(), data_2.cuda(), ppa_t1.cuda(), target_ppa.cuda(), feature_1.cuda(), feature_2.cuda(), \
                grad_1.cuda(), grad_2.cuda()
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            
            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            attr_1 = feature_1.view(data_2.size(0), -1, 1, 1)
            z1_attr_1 = torch.cat((z1, attr_1, grad_1.float().view(data_2.size(0), -1, 1, 1)), 1)
            for idx in range(data_1.size(0)):
                all_generated_feature = G_net(feature1=featuremap_1[idx].unsqueeze(0),
                                              z1=z1_attr_1[idx].unsqueeze(0),
                                              grad_1=grad_1[idx].detach().cpu().numpy() - 1,
                                              grad_2=grad_2[idx].detach().cpu().numpy() - 1,
                                              return_all=1)
                if idx == 0:
                    generate_feature_1 = all_generated_feature[0]
                else:
                    generate_feature_1 = torch.cat([generate_feature_1,
                                                    all_generated_feature[0]], dim=0)
            # generate_feature_1 = model_ESPCN(generate_feature_1)
            res_feature_1 = generate_feature_1 - featuremap_1
            
            P_current_1_t = torch.softmax(model_2_generate(featuremap_1, torch.cat(
                [feature_1,grad_1.float().view(data_2.size(0), -1)], 1)),
                                          dim=1)
            P_residue_1_i = torch.softmax(model_2_res(res_feature_1, feature_1.view(data_2.size(0), -1),
                                                      idx1=(grad_2 - grad_1 - 1).view(
                                                          data_2.size(0), ).detach().cpu().numpy()), dim=1)
            gen_pred = torch.softmax(model_2_generate(generate_feature_1, None), dim=1)
            
            pred_result_gen[batch_begin:batch_begin + data_1.size(0), :] = gen_pred.detach().cpu().numpy()
            pred_label_gen[batch_begin:batch_begin + data_1.size(0)] = gen_pred.argmax(dim=1,
                                                                                       keepdim=True).detach().cpu().numpy()
            pred_current = P_current_1_t.argmax(dim=1, keepdim=True)
            pred_minus = P_residue_1_i.argmax(dim=1, keepdim=True)
            correct_generate += pred_current.eq(target_ppa.view_as(pred_current)).sum().item()
            correct_minus += pred_minus.eq(target_ppa.view_as(pred_minus)).sum().item()
            
            pred_result_current[batch_begin:batch_begin + data_1.size(0), :] = F.softmax(P_current_1_t,
                                                                                         dim=1).detach().cpu().numpy()
            pred_result_minus[batch_begin:batch_begin + data_1.size(0), :] = F.softmax(P_residue_1_i,
                                                                                       dim=1).detach().cpu().numpy()
            pred_label_generate[batch_begin:batch_begin + data_1.size(0)] = pred_current.detach().cpu().numpy()
            pred_label_minus[batch_begin:batch_begin + data_1.size(0)] = pred_minus.detach().cpu().numpy()
            target[batch_begin:batch_begin + data_1.size(0)] = target_ppa.detach().cpu().numpy()
            
            output_cur_res_1 = P_current_1_t[:, 1] + P_current_1_t[:, 0] * P_residue_1_i[:, 1]
            output_cur_res = torch.cat([1 - output_cur_res_1.unsqueeze(1), output_cur_res_1.unsqueeze(1)], dim=1)
            
            pred_cur_res = output_cur_res.argmax(dim=1, keepdim=True)
            pred_result_cur_res[batch_begin:batch_begin + data_1.size(0), :] = output_cur_res.detach().cpu().numpy()
            pred_label_cur_res[batch_begin:batch_begin + data_1.size(0)] = pred_cur_res.detach().cpu().numpy()
            
            output_average_temp = (P_current_1_t[:, 1] + P_current_1_t[:, 0] * P_residue_1_i[:, 1] + gen_pred[:,
                                                                                                     1]) / 2.
            output_average_all = torch.cat([1 - output_average_temp.unsqueeze(1), output_average_temp.unsqueeze(1)],
                                           dim=1)
            
            pred_average_all = output_average_all.argmax(dim=1, keepdim=True)
            pred_result_average_all[batch_begin:batch_begin + data_1.size(0),
            :] = output_average_all.detach().cpu().numpy()
            pred_label_average_all[batch_begin:batch_begin + data_1.size(0)] = pred_average_all.detach().cpu().numpy()
            
            for i in range(data_1.size(0)):
                name.append(test_loader.dataset.image_path_all_1[batch_begin + i])
            
            batch_begin = batch_begin + data_1.size(0)
            
            if args.is_plot == 1:
                print('draw featuremap {} / {}'.format(batch_begin, len(test_loader.dataset)))
                for i in range(data_1.size(0)):
                    image_sequence = batch_begin - data_1.size(0) + i  ##range the image and path_name
                    if target_ppa[i] == 1:
                        
                        save_dir = '../Feature_map/Ecur_%s-Label1/' % (args.sequence,)
                        
                        unet_feature1_1 = generate_feature_1[i, :, :, :]  # add the unnomarlization
                        unet_feature1_1 = torch.unsqueeze(unet_feature1_1, 0)
                        if args.plot_sigmoid:
                            unet_feature1_1 = torch.sigmoid(unet_feature1_1)
                        unet_feature1_1 = unet_feature1_1.cpu().detach().numpy()
                        save_dir1 = (save_dir + 'unet/')
                        draw_features(16, 8, unet_feature1_1, save_dir1, image_sequence, test_loader, 'unet_',
                                      args.batch_size, args.dpi)
                        
                        minus_feature1_1 = res_feature_1[i, :, :, :]
                        minus_feature1_1 = torch.unsqueeze(minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_feature1_1 = torch.sigmoid(minus_feature1_1)
                        minus_feature1_1 = minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_feature1_1, save_dir2, image_sequence, test_loader, 'minus_',
                                      args.batch_size, args.dpi)
                        
                        minus_minus_feature1_1 = res_feature_1[i, :, :, :].mul(-1)
                        minus_minus_feature1_1 = torch.unsqueeze(minus_minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_minus_feature1_1 = torch.sigmoid(minus_minus_feature1_1)
                        minus_minus_feature1_1 = minus_minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_minus_feature1_1, save_dir2, image_sequence, test_loader,
                                      'minus_minus_',
                                      args.batch_size, args.dpi)
                        
                        featuremap_1_1 = featuremap_1[i, :, :, :]
                        featuremap_1_1 = torch.unsqueeze(featuremap_1_1, 0)
                        if args.plot_sigmoid:
                            featuremap_1_1 = torch.sigmoid(featuremap_1_1)
                        featuremap_1_1 = featuremap_1_1.cpu().detach().numpy()
                        save_dir3 = (save_dir + 'real1_')
                        draw_features(16, 8, featuremap_1_1, save_dir3, image_sequence, test_loader, 'real1_',
                                      args.batch_size, args.dpi)
                        
                        featuremap_2_1 = featuremap_2[i, :, :, :]
                        featuremap_2_1 = torch.unsqueeze(featuremap_2_1, 0)
                        if args.plot_sigmoid:
                            featuremap_2_1 = torch.sigmoid(featuremap_2_1)
                        featuremap_2_1 = featuremap_2_1.cpu().detach().numpy()
                        save_dir4 = (save_dir + 'real2_')
                        draw_features(16, 8, featuremap_2_1, save_dir4, image_sequence, test_loader, 'real2_',
                                      args.batch_size, args.dpi)
                    
                    
                    elif target_ppa[i] == 0:
                        # image_sequence = batch_begin - data_1.size(0) + i
                        save_dir = '../Feature_map/Ecur_%s-Label0/' % (args.sequence,)
                        
                        unet_feature1_1 = generate_feature_1[i, :, :, :]  # add the unnomarlization
                        unet_feature1_1 = torch.unsqueeze(unet_feature1_1, 0)
                        if args.plot_sigmoid:
                            unet_feature1_1 = torch.sigmoid(unet_feature1_1)
                        unet_feature1_1 = unet_feature1_1.cpu().detach().numpy()
                        save_dir1 = (save_dir + 'unet/')
                        draw_features(16, 8, unet_feature1_1, save_dir1, image_sequence, test_loader, 'unet_',
                                      args.batch_size, args.dpi)
                        
                        minus_feature1_1 = res_feature_1[i, :, :, :]
                        minus_feature1_1 = torch.unsqueeze(minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_feature1_1 = torch.sigmoid(minus_feature1_1)
                        minus_feature1_1 = minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_feature1_1, save_dir2, image_sequence, test_loader, 'minus_',
                                      args.batch_size, args.dpi)
                        
                        minus_minus_feature1_1 = res_feature_1[i, :, :, :].mul(-1)
                        minus_minus_feature1_1 = torch.unsqueeze(minus_minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_minus_feature1_1 = torch.sigmoid(minus_minus_feature1_1)
                        minus_minus_feature1_1 = minus_minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_minus_feature1_1, save_dir2, image_sequence, test_loader,
                                      'minus_minus_',
                                      args.batch_size, args.dpi)
                        
                        featuremap_1_1 = featuremap_1[i, :, :, :]
                        featuremap_1_1 = torch.unsqueeze(featuremap_1_1, 0)
                        if args.plot_sigmoid:
                            featuremap_1_1 = torch.sigmoid(featuremap_1_1)
                        featuremap_1_1 = featuremap_1_1.cpu().detach().numpy()
                        save_dir3 = (save_dir + 'real1_')
                        draw_features(16, 8, featuremap_1_1, save_dir3, image_sequence, test_loader, 'real1_',
                                      args.batch_size, args.dpi)
                        
                        featuremap_2_1 = featuremap_2[i, :, :, :]
                        featuremap_2_1 = torch.unsqueeze(featuremap_2_1, 0)
                        if args.plot_sigmoid:
                            featuremap_2_1 = torch.sigmoid(featuremap_2_1)
                        featuremap_2_1 = featuremap_2_1.cpu().detach().numpy()
                        save_dir4 = (save_dir + 'real2_')
                        draw_features(16, 8, featuremap_2_1, save_dir4, image_sequence, test_loader, 'real2_',
                                      args.batch_size, args.dpi)
    
    AUC_minus = sklearn.metrics.roc_auc_score(target, pred_result_minus[:, 1])
    acc_minus = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_minus, axis=1))
    cm_minus = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_minus, axis=1))
    sensitivity_minus = cm_minus[0, 0] / (cm_minus[0, 0] + cm_minus[0, 1])
    specificity_minus = cm_minus[1, 1] / (cm_minus[1, 0] + cm_minus[1, 1])

    # AUC_gen = sklearn.metrics.roc_auc_score(target, pred_result_gen[:, 1])
    # acc_gen = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_gen, axis=1))
    # cm_gen = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_gen, axis=1))
    # sensitivity_gen = cm_gen[0, 0] / (cm_gen[0, 0] + cm_gen[0, 1])
    # specificity_gen = cm_gen[1, 1] / (cm_gen[1, 0] + cm_gen[1, 1])

    AUC_cur_res = sklearn.metrics.roc_auc_score(target, pred_result_cur_res[:, 1])
    acc_cur_res = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_cur_res, axis=1))
    cm_cur_res = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_cur_res, axis=1))
    sensitivity_cur_res = cm_cur_res[0, 0] / (cm_cur_res[0, 0] + cm_cur_res[0, 1])
    specificity_cur_res = cm_cur_res[1, 1] / (cm_cur_res[1, 0] + cm_cur_res[1, 1])
    
    AUC_average_all = sklearn.metrics.roc_auc_score(target, pred_result_average_all[:, 1])
    acc_average_all = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_average_all, axis=1))
    
    AUC_gen = sklearn.metrics.roc_auc_score(target, pred_result_gen[:, 1])
    acc_gen = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_gen, axis=1))
    
    args.logger.info('In epoch {} for generate, AUC is {}, acc is {}.'.format(epoch, AUC_gen, acc_gen))
    args.logger.info(
        'In epoch {} for minus, AUC is {}, acc is {}, loss is {}'.format(epoch, AUC_minus, acc_minus, test_loss_minus))
    args.logger.info('In epoch {} for cur_res, AUC is {}, acc is {}'.format(epoch, AUC_cur_res, acc_cur_res))
    args.logger.info(
        'In epoch {} for average all with gen, AUC is {}, acc is {}'.format(epoch, AUC_average_all, acc_average_all))
    args.logger.info('      ')
    
    results = {
        'AUC_minus': AUC_minus,
        'acc_minus': acc_minus,
        'sensitivity_minus': sensitivity_minus,
        'specificity_minus': specificity_minus,
        'pred_result_minus': pred_result_minus,
        'pred_label_minus': pred_label_minus,
        
        'AUC_cur_res': AUC_cur_res,
        'acc_cur_res': acc_cur_res,
        'sensitivity_cur_res': sensitivity_cur_res,
        'specificity_cur_res': specificity_cur_res,
        'pred_result_cur_res': pred_result_cur_res,
        'pred_label_cur_res': pred_label_cur_res,
        
        'AUC_average_all': AUC_average_all,
        'acc_average_all': acc_average_all,
        'pred_result_average_all': pred_result_average_all,
        'pred_label_average_all': pred_label_average_all,
        
        'pred_result_gen': pred_result_gen,
        'AUC_gen': AUC_gen,
        'acc_gen': acc_gen,
        
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
    
    train_loader, val_loader_list, test_loader_list = get_all_dataloader_order1(args)
    
    model_1 = RN18_front(final_tanh=args.final_tanh).cuda()
    model_2_res = RN18_last_attr_ind(num_classes=args.class_num,
                                     feature_dim=len(args.feature_list),
                                     concat_num=1,
                                     task_num=5).cuda()
    model_2_generate = RN18_last_attr(num_classes=args.class_num,
                                          feature_dim=len(args.feature_list) + 1,
                                      dropout=args.dropout, dp=args.dp).cuda()
    
    # old model
    # load_pytorch_model(model_1,
    #                    r'../results/Ecur_200616_BS/2020-06-29-23-50-10_Mixed_0.010000_30.000000_size_256_ep_120_0_R_Maculae/best_val_auc_model_1.pt')
    #load_pytorch_model(model_2_generate,
                       #r'../results/Ecur_200616_BS/2020-06-29-23-50-10_Mixed_0.010000_30.000000_size_256_ep_120_0_R_Maculae/best_val_auc_model_2_generate.pt')
    load_pytorch_model(model_1,
                       r'../results/Ecur_211217_BS_Extractor/2024-06-21-09-21-49_Mixed_0.002000_50.000000_size_256_ep_80_0_R_Maculae/best_val_auc_model_1.pt')


    if args.G_net_type == 'G_net':
        G_net = Generator_LSTM_1_ind(feature_num=len(args.feature_list) + 1,
                                    final_tanh=args.final_tanh,
                                    in_channel=128,
                                    RNN_hidden=args.RNN_hidden).cuda()
    elif args.G_net_type == 'U_net':
        G_net = UNet(n_channels=128,
                     n_classes=128,
                     bilinear=args.bi_linear,
                     feature_num=len(args.feature_list) + 1,
                     final_tanh=args.final_tanh,
                     is_ESPCN=args.is_ESPCN, scale_factor=args.scale_factor, mid_channel=args.dw_midch,
                     dw_type=args.dw_type).cuda()
    
    D_net = Discriminator(128).cuda()

    print('-' * 20)
    print('-' * 20)
    print('print the number of model param :')
    print_model_parm_nums(model_1, logger=args.logger)
    print_model_parm_nums(model_2_res, logger=args.logger)
    print_model_parm_nums(model_2_generate, logger=args.logger)
    print_model_parm_nums(G_net, logger=args.logger)
    print_model_parm_nums(D_net, logger=args.logger)
    print('-' * 20)
    print('-' * 20)

    optimizer_M_2_res = optim.SGD([{'params': model_2_res.parameters(), 'lr': args.lr2,
                                    'weight_decay': args.wd2, 'momentum': args.momentum}
                                   ])
    optimizer_M_2_generate = optim.SGD(
        [{'params': model_2_generate.parameters(), 'lr': args.lr2,
          'weight_decay': args.wd2, 'momentum': args.momentum}])
    optimizer_G = optim.RMSprop([{'params': G_net.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
    optimizer_D = optim.RMSprop(D_net.parameters(), lr=args.lr, weight_decay=args.wd)

    
    full_results = {}
    args = init_metric(args)
    
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_results = train(args, model_1, model_2_generate, model_2_res, G_net, D_net,
                                  train_loader, optimizer_M_2_generate, optimizer_M_2_res, optimizer_G, optimizer_D,
                                  epoch)
            test_results_list = []
            val_results_list = []
            for ss in range(len(val_loader_list)):
                test_results_list.append(
                    evaluate(args, model_1, model_2_generate, model_2_res, G_net,
                             test_loader_list[ss], epoch))
                val_results_list.append(
                    evaluate(args, model_1, model_2_generate, model_2_res, G_net,
                             val_loader_list[ss], epoch))
            
            adjust_learning_rate(optimizer_M_2_generate, epoch, args)
            adjust_learning_rate(optimizer_M_2_res, epoch, args)
            
            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results_ours_order1(args,
                         model_1,
                         model_2_generate,
                         model_2_res,
                         G_net,
                         D_net,
                         train_results,
                         val_results_list,
                         test_results_list,
                         full_results,
                         optimizer_M_2_generate,
                         optimizer_M_2_res,
                         optimizer_G,
                         optimizer_D,
                         epoch)
    finally:
        args.logger.info('save_results_path: %s' % args.save_dir)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)


if __name__ == '__main__':
    main()
