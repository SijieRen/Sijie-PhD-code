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
from src.model_2d import *
from src.util import *
# from src.main_pair import train, evaluate


# train_baseline_LSOR(args, config, optimizer_lsor,model_lsor, scheduler_lsor ,
                                # train_loader, epoch)
def evaluate(config, model, dataloader, phase='val', set='val', save_res=True, info='', epoch=0):
    model.eval()
    loader = dataloader
    # if phase == 'val':
    #     loader = valDataLoader
    # else:
    #     if set == 'train':
    #         loader = trainDataLoader
    #     elif set == 'val':
    #         loader = valDataLoader
    #     elif set == 'test':
    #         loader = testDataLoader
    #     else:
    #         raise ValueError('Undefined loader')

    res_path = os.path.join("./lror_res", 'result_'+set)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    path = os.path.join(res_path, 'results_all'+info+'.h5')
    if os.path.exists(path):
        os.remove(path)

    loss_all_dict = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0}
    input1_list = []
    input2_list = []
    label_list = []
    age_list = []
    interval_list = []
    recon1_list = []
    recon2_list = []
    recon_zq_list = []
    ze1_list = []
    ze2_list = []
    ze_diff_list = []
    zq_list = []
    k1_list = []
    k2_list = []
    sim1_list = []
    sim2_list = []
    subj_id_list = []
    case_order_list = []
    global_iter = 0
    with torch.no_grad():
        recon_emb = model.recon_embeddings()

        for batch_idx, (data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(loader):
            data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
                data_1.cuda(), data_2.cuda(), ppa_t1.cuda(), target_ppa.cuda(), feature_1.cuda(), \
                feature_2.cuda(), grad_1.cuda(), grad_2.cuda()
            # LSOR
            loss_all_dict = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0.}
            global_iter0 = global_iter
            global_iter += 1
            # run LSOR model
            image1 = data_1
            image2 = data_2
            # if len(image1.shape) == 4:
            #     image1 = image1.unsqueeze(1)
            #     image2 = image2.unsqueeze(1)
            # label = sample['label'].to(config['device'], dtype=torch.float)
            # interval = sample['interval'].to(config['device'], dtype=torch.float)
            interval = grad_2 - grad_1
            label = target_ppa
            recons, zs = model.forward_pair_z(image1, image2, interval)

            recon_ze1, recon_ze2 = recons[0][0], recons[0][1]
            recon_zq1, recon_zq2 = recons[1][0], recons[1][1]
            z_e1, z_e2, z_e_diff = zs[0][0], zs[0][1], zs[0][2]
            z_q1, z_q2 = zs[1][0], zs[1][1]
            k1, k2 = zs[2][0], zs[2][1]
            sim1, sim2 = zs[3][0], zs[3][1]

                        # loss
            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = 0.5 * (model.compute_recon_loss(image1, recon_ze1) + model.compute_recon_loss(image2, recon_ze2))
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)

            if config['lambda_recon_zq'] > 0 and epoch >= config['warmup_epochs']:
                loss_recon_zq = 0.5 * (model.compute_recon_loss(image1, recon_zq1) + model.compute_recon_loss(image2, recon_zq2))
                loss += config['lambda_recon_zq'] * loss_recon_zq
            else:
                loss_recon_zq = torch.tensor(0.)

            if config['lambda_commit'] > 0 and epoch >= config['warmup_epochs']:
                loss_commit = 0.5 * (model.compute_commit_loss(z_e1, z_q1) + model.compute_commit_loss(z_e2, z_q2))
                loss += config['lambda_commit'] * loss_commit
            else:
                loss_commit = torch.tensor(0.)

            if config['lambda_som'] > 0 and epoch >= config['warmup_epochs']:
                loss_som = 0.5 * (model.compute_som_loss(z_e1, k1) + model.compute_som_loss(z_e2, k2))
                loss += config['lambda_som'] * loss_som
            else:
                loss_som = torch.tensor(0.)

            if config['lambda_dir'] > 0 and (config['dir_reg'] == 'LSSL' or epoch >= config['warmup_epochs']):
                if config['dir_reg'] == 'LSSL':
                    if epoch < config['warmup_epochs']:
                        loss_dir = model.compute_lssl_direction_loss(z_e_diff)
                    else:
                        loss_dir = model.compute_lssl_direction_loss(z_e_diff) + config['emb_dir_ratio'] * model.compute_emb_lssl_direction_loss()
                elif config['dir_reg'] == 'LNE':
                    if config['is_grid_ema']:
                        loss_dir = model.compute_lne_grid_ema_direction_loss(z_e_diff, k1)
                        model.update_grid_dz_ema(z_e_diff, k1)
                    else:
                        loss_dir = model.compute_lne_direction_loss(z_e_diff, sim1)
                else:
                    raise ValueError('Do not support this direction regularization method!')
                loss += config['lambda_dir'] * loss_dir
            else:
                loss_dir = torch.tensor(0.)


            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['recon_zq'] += loss_recon_zq.item()
            loss_all_dict['commit'] += loss_commit.item()
            loss_all_dict['som'] += loss_som.item()
            loss_all_dict['dir'] += loss_som.item()

            # if phase == 'test' and save_res:
            #     input1_list.append(image1.detach().cpu().numpy())
            #     input2_list.append(image2.detach().cpu().numpy())
            #     label_list.append(label.detach().cpu().numpy())
            #     recon1_list.append(recon_ze1.detach().cpu().numpy())
            #     recon2_list.append(recon_ze2.detach().cpu().numpy())
            #     recon_zq_list.append(recon_zq1.detach().cpu().numpy())
            #     ze1_list.append(z_e1.detach().cpu().numpy())
            #     ze2_list.append(z_e2.detach().cpu().numpy())
            #     ze_diff_list.append(z_e_diff.detach().cpu().numpy())
            #     age_list.append(sample['age'].numpy())
            #     interval_list.append(interval.detach().cpu().numpy())
            #     subj_id_list.append(sample['subj_id'])
            #     case_order_list.append(np.stack([sample['case_order'][0].numpy(), sample['case_order'][1].numpy()], 1))
            #     zq_list.append(z_q1.detach().cpu().numpy())
            #     sim1_list.append(sim1.detach().cpu().numpy())
            #     sim2_list.append(sim2.detach().cpu().numpy())
            k1_list.append(k1.detach().cpu().numpy())
            k2_list.append(k2.detach().cpu().numpy())

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (batch_idx + 1)

        if phase == 'test' and save_res:
            input1_list = np.concatenate(input1_list, axis=0)
            input2_list = np.concatenate(input2_list, axis=0)
            label_list = np.concatenate(label_list, axis=0)
            interval_list = np.concatenate(interval_list, axis=0)
            age_list = np.concatenate(age_list, axis=0)
            subj_id_list = np.concatenate(subj_id_list, axis=0)
            case_order_list = np.concatenate(case_order_list, axis=0)
            recon1_list = np.concatenate(recon1_list, axis=0)
            recon2_list = np.concatenate(recon2_list, axis=0)
            recon_zq_list = np.concatenate(recon_zq_list, axis=0)
            ze1_list = np.concatenate(ze1_list, axis=0)
            ze2_list = np.concatenate(ze2_list, axis=0)
            ze_diff_list = np.concatenate(ze_diff_list, axis=0)
            zq_list = np.concatenate(zq_list, axis=0)
            k1_list = np.concatenate(k1_list, axis=0)
            k2_list = np.concatenate(k2_list, axis=0)
            sim1_list = np.concatenate(sim1_list, axis=0)
            sim2_list = np.concatenate(sim2_list, axis=0)

            h5_file = h5py.File(path, 'w')
            # h5_file.create_dataset('subj_id', data=subj_id_list)
            h5_file.create_dataset('case_order', data=case_order_list)
            h5_file.create_dataset('input1', data=input1_list)
            h5_file.create_dataset('input2', data=input2_list)
            h5_file.create_dataset('label', data=label_list)
            h5_file.create_dataset('interval', data=interval_list)
            h5_file.create_dataset('age', data=age_list)
            h5_file.create_dataset('recon1', data=recon1_list)
            h5_file.create_dataset('recon2', data=recon2_list)
            h5_file.create_dataset('recon_zq', data=recon_zq_list)
            h5_file.create_dataset('ze1', data=ze1_list)
            h5_file.create_dataset('ze2', data=ze2_list)
            h5_file.create_dataset('ze_diff', data=ze_diff_list)
            h5_file.create_dataset('zq', data=zq_list)
            h5_file.create_dataset('k1', data=k1_list)
            h5_file.create_dataset('k2', data=k2_list)
            h5_file.create_dataset('sim1', data=sim1_list)
            h5_file.create_dataset('sim2', data=sim2_list)
            h5_file.create_dataset('embeddings', data=model.embeddings.detach().cpu().numpy())
            h5_file.create_dataset('recon_emb', data=recon_emb.detach().cpu().numpy())
        else:
            k1_list = np.concatenate(k1_list, axis=0)
            k2_list = np.concatenate(k2_list, axis=0)

        print('Number of used embeddings:', np.unique(k1_list).shape, np.unique(k2_list).shape)
        loss_all_dict['k1'] = np.unique(k1_list).shape[0]
        loss_all_dict['k2'] = np.unique(k2_list).shape[0]

    return loss_all_dict

def train_baseline_LSOR(args, config,
                   model_lsor,optimizer_lsor,scheduler_lsor,
                   train_loader, test_loader_list, epoch, 
                    monitor_metric_best = 100
                   ):
    global_iter = 0
    # model_1.train()
    # model_2_generate.train()
    model_lsor.train()
    
    train_loss_M2_generate = AverageMeter()
    image_sequence = 0
    correct_generate = 0
    for batch_idx, (data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(train_loader):
        data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
            data_1.cuda(), data_2.cuda(), ppa_t1.cuda(), target_ppa.cuda(), feature_1.cuda(), \
            feature_2.cuda(), grad_1.cuda(), grad_2.cuda()
        # LSOR
        loss_all_dict = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0.}
        global_iter0 = global_iter
        global_iter += 1
        interval = grad_2 - grad_1
        # run LSOR model
        image1 = data_1
        image2 = data_2
        recons, zs = model_lsor.forward_pair_z(image1, image2, interval)

        recon_ze1, recon_ze2 = recons[0][0], recons[0][1]
        recon_zq1, recon_zq2 = recons[1][0], recons[1][1]
        z_e1, z_e2, z_e_diff = zs[0][0], zs[0][1], zs[0][2]
        z_q1, z_q2 = zs[1][0], zs[1][1]
        k1, k2 = zs[2][0], zs[2][1]
        sim1, sim2 = zs[3][0], zs[3][1]

        # loss
        loss = 0
        if config['lambda_recon'] > 0:
            loss_recon = 0.5 * (model_lsor.compute_recon_loss(image1, recon_ze1) + model_lsor.compute_recon_loss(image2, recon_ze2))
            loss += config['lambda_recon'] * loss_recon
        else:
            loss_recon = torch.tensor(0.)

        if config['lambda_recon_zq'] > 0 and epoch >= config['warmup_epochs']:
            loss_recon_zq = 0.5 * (model_lsor.compute_recon_loss(image1, recon_zq1) + model_lsor.compute_recon_loss(image2, recon_zq2))
            loss += config['lambda_recon_zq'] * loss_recon_zq
        else:
            loss_recon_zq = torch.tensor(0.)

        if config['lambda_commit'] > 0 and epoch >= config['warmup_epochs']:
            loss_commit = 0.5 * (model_lsor.compute_commit_loss(z_e1, z_q1) + model_lsor.compute_commit_loss(z_e2, z_q2))
            loss += config['lambda_commit'] * loss_commit
        else:
            loss_commit = torch.tensor(0.)

        if config['lambda_som'] > 0 and epoch >= config['warmup_epochs']:
            loss_som = 0.5 * (model_lsor.compute_som_loss(z_e1, k1, global_iter-config['warmup_epochs']*len(train_loader), len(train_loader)) + \
                            model_lsor.compute_som_loss(z_e2, k2, global_iter-config['warmup_epochs']*len(train_loader), len(train_loader)))
            loss += config['lambda_som'] * loss_som
        else:
            loss_som = torch.tensor(0.)

        if config['lambda_dir'] > 0 and (config['dir_reg'] == 'LSSL' or epoch >= config['warmup_epochs']):
            if config['dir_reg'] == 'LSSL':
                if epoch < config['warmup_epochs']:
                    loss_dir = model_lsor.compute_lssl_direction_loss(z_e_diff)
                else:
                    loss_dir = model_lsor.compute_lssl_direction_loss(z_e_diff) + config['emb_dir_ratio'] * model_lsor.compute_emb_lssl_direction_loss()
            elif config['dir_reg'] == 'LNE':
                if config['is_grid_ema']:
                    loss_dir = model_lsor.compute_lne_grid_ema_direction_loss(z_e_diff, k1)
                    model_lsor.update_grid_dz_ema(z_e_diff, k1)
                else:
                    loss_dir = model_lsor.compute_lne_direction_loss(z_e_diff, sim1)
            else:
                raise ValueError('Do not support this direction regularization method!')
            loss += config['lambda_dir'] * loss_dir
        else:
            loss_dir = torch.tensor(0.)

        loss_all_dict['all'] += loss.item()
        loss_all_dict['recon'] += loss_recon.item()
        loss_all_dict['recon_zq'] += loss_recon_zq.item()
        loss_all_dict['commit'] += loss_commit.item()
        loss_all_dict['som'] += loss_som.item()
        loss_all_dict['dir'] += loss_dir.item()

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for name, param in model_lsor.named_parameters():
            try:
                if not torch.isfinite(param.grad).all():
                    pdb.set_trace()
            except:
                continue

        emb_old = model_lsor.embeddings.detach().cpu().numpy()
        enc_old = torch.cat([param.view(-1) for param in model_lsor.encoder.parameters()]).detach().cpu().numpy()
        dec_old = torch.cat([param.view(-1) for param in model_lsor.decoder.parameters()]).detach().cpu().numpy()

        optimizer_lsor.step()
        optimizer_lsor.zero_grad()

        emb_new = model_lsor.embeddings.detach().cpu().numpy()
        enc_new = torch.cat([param.view(-1) for param in model_lsor.encoder.parameters()]).detach().cpu().numpy()
        dec_new = torch.cat([param.view(-1) for param in model_lsor.decoder.parameters()]).detach().cpu().numpy()
        # print(np.abs(emb_new-emb_old).mean(), np.abs(enc_new-enc_old).mean(), np.abs(dec_new-dec_old).mean())
        # pdb.set_trace()

        if global_iter % 100 == 0:
            # pdb.set_trace()
            print('Epoch[%3d], batch_idx[%3d]: loss=[%.4f], recon=[%.4f], recon_zq=[%.4f], commit=[%.4f], som=[%.4f], dir=[%.4f]' \
                    % (epoch, batch_idx, loss.item(), loss_recon.item(), loss_recon_zq.item(), loss_commit.item(), loss_som.item(), loss_dir.item()))
            print('Num. of k:', torch.unique(k1).shape[0], torch.unique(k2).shape[0])

    # save train result
    num_iter = global_iter - global_iter0
    for key in loss_all_dict.keys():
        loss_all_dict[key] /= num_iter
    if 'k1' not in loss_all_dict:
        loss_all_dict['k1'] = 0
    if 'k2' not in loss_all_dict:
        loss_all_dict['k2'] = 0

    # save_result_stat(loss_all_dict, config, info='epoch[%2d]'%(epoch))
    print(loss_all_dict)

        # validation
        # pdb.set_trace()
    for ss in range(len(test_loader_list)):

        stat = evaluate(config, model_lsor, test_loader_list[ss], phase='val', set='val', save_res=False, epoch=epoch)
        if ss ==0:
            monitor_metric = stat['recon']
        else:
            monitor_metric += stat['recon']

    # monitor_metric = stat['all']
    
    scheduler_lsor.step(monitor_metric/5)
    # save_result_stat(stat, config, info='val')
    print("stat", stat)

    # save ckp
    is_best = False
    if monitor_metric <= monitor_metric_best:
        is_best = True
        monitor_metric_best = monitor_metric if is_best == True else monitor_metric_best
    state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
            'optimizer': optimizer_lsor.state_dict(), 'scheduler': scheduler_lsor.state_dict(), \
            'model': model_lsor.state_dict()}
    print(optimizer_lsor.param_groups[0]['lr'])
    # save_checkpoint(state, is_best, "miccai-2023-lror/epoch{%s}}.pth.tar"%(str(epoch).zfill(3)))

    # return monitor_metric_best


#train_baseline_CLS(args, config,optimizer_lsor,
                    # model_lsor, scheduler_lsor, model_cls, 
                    # optimizer_cls,epoch)

def train_baseline_CLS(args, config,optimizer_lsor,
                       model_lsor, scheduler_lsor, model_cls,
                       optimizer_cls,train_loader,
                                               test_loader_list,epoch
                       ):
    global_iter = 0
    # model_1.train()
    # model_2_generate.train()
    model_lsor.eval()
    model_cls.train()
    
    train_loss_M2_generate = AverageMeter()
    
    image_sequence = 0
    correct_generate = 0
    for batch_idx, (data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(train_loader):
        data_1, data_2, ppa_t1, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
            data_1.cuda(), data_2.cuda(), ppa_t1.cuda(), target_ppa.cuda(), feature_1.cuda(), \
            feature_2.cuda(), grad_1.cuda(), grad_2.cuda()
        # LSOR
        loss_all_dict = {'all': 0, 'recon': 0., 'recon_zq': 0., 'som': 0., 'commit': 0., 'dir': 0.}
        global_iter0 = global_iter
        global_iter += 1
        interval = grad_2 - grad_1
        # run LSOR model
        image1 = data_1
        image2 = data_2
        recons, zs = model_lsor.forward_pair_z(image1, image2, interval)

        # recon_ze1, recon_ze2 = recons[0][0], recons[0][1]
        # recon_zq1, recon_zq2 = recons[1][0], recons[1][1]
        # z_e1, z_e2, z_e_diff = zs[0][0], zs[0][1], zs[0][2]
        # z_q1, z_q2 = zs[1][0], zs[1][1]
        # k1, k2 = zs[2][0], zs[2][1]
        # sim1, sim2 = zs[3][0], zs[3][1]
        output_generate_1 = model_cls(zs[0])
        # loss
        loss_sum = torch.FloatTensor([0.0]).cuda()
        loss = F.CrossEntropy(F.softmax(output_generate_1, dim=1), target_ppa=target_ppa)

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for name, param in model_lsor.named_parameters():
            try:
                if not torch.isfinite(param.grad).all():
                    pdb.set_trace()
            except:
                continue

        enc_old = torch.cat([param.view(-1) for param in model_cls.encoder.parameters()]).detach().cpu().numpy()
        cls_old = torch.cat([param.view(-1) for param in model_cls.classifier.parameters()]).detach().cpu().numpy()
        optimizer_cls.step()
        optimizer_cls.zero_grad()

        enc_new = torch.cat([param.view(-1) for param in model_cls.encoder.parameters()]).detach().cpu().numpy()
        cls_new = torch.cat([param.view(-1) for param in model_cls.classifier.parameters()]).detach().cpu().numpy()
        print(np.abs(enc_new-enc_old).sum(), np.abs(cls_new-cls_old).sum())

        if global_iter % 100 == 0:
            # pdb.set_trace()
            print('Epoch[%3d], iter[%3d]: loss=[%.4f], cls=[%.4f]' % (epoch, batch_idx, loss.item(), loss.item()))


def evaluate_baseline_CLS(args,
                      model_lsor,
                      model_cls,
                      model_1,
                      model_2_generate,
                      test_loader,
                      epoch):
    model_1.eval()
    model_2_generate.eval()
    model_lsor.eval()
    model_cls.eval()
    
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
            
            interval = grad_2 - grad_1
            recons, zs = model_lsor.forward_pair_z(data_1, data_2, interval)

            output_generate_1 = model_cls(zs[0])
            # loss
            # loss_sum = torch.FloatTensor([0.0]).cuda()
            # loss = F.CrossEntropy(F.softmax(output_generate_1, dim=1), target_ppa=target_ppa)

            # featuremap_1 = model_1(data_1)
            # if args.model == 'MM_F':
            #     output_generate_1 = model_2_generate(featuremap_1, torch.cat(
            #         [feature_1, grad_1.float().view(data_2.size(0), -1)], 1))
            # else:
            #     output_generate_1 = model_2_generate(featuremap_1)
            
            output_generate_1 = F.softmax(output_generate_1, dim=1)# SoftMax()=tensor([[0.3,0.7],[0.4,0.6],......]) 
            pred_generate = output_generate_1.argmax(dim=1, keepdim=True)
            # #增加ppa_t1 的判断
            # # rulebase
            # pred_generate = pred_generate.detach().cpu()
            # for i in range(len(pred_generate.size(0))):
            #     if ppa_t1[i].detach().detach().cpu().numpy()[0] == 1:
            #         pred_generate[i] = 1
            # pred_generate = pred_generate.cuda()
            # #  rulebase
            correct_generate += pred_generate.eq(target_ppa.view_as(pred_generate)).sum().item()
            test_loss_generate += F.cross_entropy(output_generate_1, target_ppa, reduction='sum').item()
            pred_result_generate[batch_begin:batch_begin + data_1.size(0), :] = output_generate_1.cpu().numpy()
            pred_label_generate[batch_begin:batch_begin + data_1.size(0)] = pred_generate.cpu().numpy()
            
            ppa_1[batch_begin:batch_begin + data_1.size(0)] = ppa_t1.cpu().numpy()
            target[batch_begin:batch_begin + data_1.size(0)] = target_ppa.cpu().numpy()
            
            for i in range(data_1.size(0)):
                name.append(test_loader.dataset.image_path_all_1[batch_begin + i])
            
            batch_begin = batch_begin + data_1.size(0)
    
    test_loss_generate /= len(test_loader.dataset)
    
    AUC_generate = sklearn.metrics.roc_auc_score(target,
                                                 pred_result_generate[:, 1])  # pred -> Softmax(model output), pred_label -> prediction
    acc_generate = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_generate, axis=1))
    cm_generate = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_generate, axis=1))
    sensitivity_generate = cm_generate[0, 0] / (cm_generate[0, 0] + cm_generate[0, 1])
    specificity_generate = cm_generate[1, 1] / (cm_generate[1, 0] + cm_generate[1, 1])
    
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
    # LSOR
    _, config = load_config_yaml('src/config_pair.yaml')
    config['device'] = torch.device('cuda:'+ config['gpu'])
    args = get_opts()


    # if config['model_name'] in ['SOMPairVisit']:
    #     model = SOMPairVisit(config=config).to(config['device'])
    # else:
    #     raise ValueError('Not support other models yet!')

    # define model
    if config['model_name'] in ['SOMPairVisit']:
        model_lsor = SOMPairVisit(config=config).cuda()
        model_cls = LSOR_cls(latent_size=256, inter_num_ch=1024, inter_num_ch2=512)
    else:
        raise ValueError('Not support other models yet!')

    optimizer_lsor = torch.optim.Adam(model_lsor.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd, amsgrad=True)
    scheduler_lsor = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lsor, mode='min', factor=0.1, patience=10, min_lr=1e-5)
    start_epoch = -1

    optimizer_cls = optim.SGD(
        [{'params': model_cls.parameters(), 'lr': args.lr2,
          'weight_decay': args.wd2, 'momentum': args.momentum}])

    # Training settings
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader_list, test_loader_list = get_all_dataloader_order1(args)
    
    model_1 = RN18_front().cuda()
    if args.model == 'MM_F':
        model_2_generate = RN18_last_attr_e(num_classes=args.class_num,
                                          feature_dim=len(args.feature_list) + 1, dropout=args.dropout, dp=args.dp).cuda()
    else:
        model_2_generate = RN18_last_e(num_classes = args.class_num, dropout=args.dropout, dp=args.dp).cuda()
    
    print('-' * 20)
    print('-' * 20)
    print('print the number of model param :')
    print_model_parm_nums(model_1, logger=args.logger)
    print_model_parm_nums(model_2_generate, logger=args.logger)
    print_model_parm_nums(model_lsor, logger=args.logger)
    print_model_parm_nums(model_cls, logger=args.logger)
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
        # train the LROE
        for epoch in range(1, args.epochs):
            args.logger.info('LROR main model')
            train_baseline_LSOR(args, config, model_lsor, optimizer_lsor,scheduler_lsor ,
                                train_loader, test_loader_list, epoch)
            pass
        # model_lsor = load_pretrained_model_lsor(model_lsor), "./miccai-2023-lror/model_best.pth.tar"
        # model_lsor = load_pytorch_model(model_lsor, "./miccai-2023-lror/model_best.pth.tar")
        monitor_metric_best = 100
        # train the classifier
        for epoch in range(1, args.epochs + 1):
            args.logger.info('LROR Classifier')
            start_time = time.time()
            train_baseline_CLS(args, config,optimizer_lsor,
                                               model_lsor, scheduler_lsor, model_2_generate, 
                                               optimizer_cls,
                                               train_loader,
                                               test_loader_list, epoch)
            # [model], start_epoch = load_checkpoint_by_key(
            #     [model], config['ckpt_path'], ['model'], config['device'], config['ckpt_name'])
            # zheli jixue xie
            # model = load_pytorch_model(model_lsor, "")
            test_results_list = []
            val_results_list = []
            for ss in range(len(val_loader_list)):
                test_results_list.append(
                    evaluate_baseline_CLS(args, config,model_lsor, model_1, model_2_generate, test_loader_list[ss], epoch))
                val_results_list.append(
                    evaluate_baseline_CLS(args, config,model_lsor, model_1, model_2_generate, val_loader_list[ss], epoch))
            
            # adjust_learning_rate(optimizer_M_1, epoch, args)
            # adjust_learning_rate(optimizer_M_2_generate, epoch, args)
            
            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results_baseline_order1(
                args,
                model_1,
                model_2_generate,
                # train_results,
                val_results_list,
                test_results_list,
                full_results,
                optimizer_M_1,
                optimizer_M_2_generate,
                epoch)
    finally:
        args.logger.info('save_results_path: %s' % args.save_dir)
        args.logger.info('LSOR BS-order1')
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
if __name__ == '__main__':
    main()
