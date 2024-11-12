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


class Clas_ppa_train(data.Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 fold='train',
                 eye=0,
                 center=0,
                 feature_list=[8, 9],
                 ):
        super(Clas_ppa_train, self).__init__()
        self.root = root
        self.transform = transform
        self.eye = eye
        self.center = center
        self.image_path_all_1 = []
        self.image_path_all_2 = []
        self.target_ppa = []
        self.feature_list = feature_list
        self.feature_all_1 = []
        self.feature_all_2 = []
        self.base_grade_num = []
        self.base_grade_num_2 = []
        self.feature_mask_1 = np.zeros(72, ).astype('bool')
        self.feature_mask_2 = np.zeros(72, ).astype('bool')
        for ids in feature_list:
            self.feature_mask_1[ids] = True
            self.feature_mask_2[ids + 32] = True
        
        workbook1 = xlrd.open_workbook(
            r"../ppa-classi-dataset-onuse/ppa_06-29-order1-std-8.xls")
        sheet1 = workbook1.sheet_by_index(0)
        
        for rows in range(1, sheet1.nrows):
            if sheet1.row_values(rows)[5] == fold:
                if sheet1.row_values(rows)[3] in self.eye:
                    if str(sheet1.row_values(rows)[4]) in self.center:
                        self.image_path_all_1.append(os.path.join(self.root, sheet1.row_values(rows)[1]))
                        self.image_path_all_2.append(os.path.join(self.root, sheet1.row_values(rows)[2]))
                        self.target_ppa.append(sheet1.row_values(rows)[6])
                        self.feature_all_1.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))
                        self.feature_all_2.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_2].astype('float32'))
                        self.base_grade_num.append(int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                        self.base_grade_num_2.append(int(sheet1.row_values(rows)[2].split('/')[0][-1]))
    
    def __getitem__(self, index):
        
        img_path_1, img_path_2 = self.image_path_all_1[index], self.image_path_all_2[index]
        target_ppa = self.target_ppa[index]
        
        img_1 = Image.open(img_path_1)
        img_2 = Image.open(img_path_2)
        base_target = [-1, -1]
        if (target_ppa == 0 or target_ppa == 1) and self.base_grade_num[index] == 1:
            base_target[0] = 0
        
        if (target_ppa == 0 or target_ppa == 1) and self.base_grade_num_2[index] == 6:
            base_target[1] = target_ppa
        
        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        
        return img_1, \
               img_2, \
               torch.from_numpy(np.array(target_ppa).astype('int')), \
               torch.from_numpy(np.array(self.feature_all_1[index]).astype('float32')), \
               torch.from_numpy(np.array(self.feature_all_2[index]).astype('float32')), \
               torch.from_numpy(np.array(self.base_grade_num[index]).astype('int')), \
               torch.from_numpy(np.array(self.base_grade_num_2[index]).astype('int'))
    
    def __len__(self):
        return len(self.image_path_all_1)


def get_all_dataloader(args):
    test_loader_list = []
    val_loader_list = []
    kwargs = {'num_workers': args.works, 'pin_memory': True}
    train_loader = DataLoaderX(
        Clas_ppa_train(args.data_root, fold='train', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.RandomRotation(30),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test2', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val2', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test3', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val3', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test4', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val4', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test5', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val5', eye=args.eye, center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    return train_loader, val_loader_list, test_loader_list


def evaluate(args,
             model_1,
             model_2_generate,
             model_2_res,
             G_net,
             test_loader,
             epoch):
    model_1.eval()
    model_2_generate.eval()
    model_2_res.eval()
    G_net.eval()
    
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
        
        for batch_idx, (data_1, data_2, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(test_loader):
            data_1, data_2, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
                data_1.cuda(), data_2.cuda(), target_ppa.cuda(), feature_1.cuda(), feature_2.cuda(), \
                grad_1.cuda(), grad_2.cuda()
            
            featuremap_1 = model_1(data_1)
            featuremap_2 = model_1(data_2)
            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            attr_1 = feature_1.view(data_2.size(0), -1, 1, 1)
            z1_attr_1 = torch.cat((z1, attr_1, grad_1.float().view(data_2.size(0), -1, 1, 1)),
                                  1)  # cat the extra feature random z and specific extra labels
            generate_feature_1 = G_net(z1_attr_1, featuremap_1)
            res_feature_1 = generate_feature_1 - featuremap_1
            
            P_current_1_i = torch.softmax(model_2_generate(featuremap_1), dim=1)
            P_residue_1_i = torch.softmax(model_2_res(res_feature_1), dim=1)
            gen_pred = torch.softmax(model_2_generate(generate_feature_1), dim=1)
            
            pred_result_gen[batch_begin:batch_begin + data_1.size(0), :] = gen_pred.detach().cpu().numpy()
            pred_label_gen[batch_begin:batch_begin + data_1.size(0)] = gen_pred.argmax(dim=1,
                                                                                       keepdim=True).detach().cpu().numpy()
            pred_current = P_current_1_i.argmax(dim=1, keepdim=True)
            pred_minus = P_residue_1_i.argmax(dim=1, keepdim=True)
            correct_generate += pred_current.eq(target_ppa.view_as(pred_current)).sum().item()
            correct_minus += pred_minus.eq(target_ppa.view_as(pred_minus)).sum().item()
            
            pred_result_current[batch_begin:batch_begin + data_1.size(0), :] = F.softmax(P_current_1_i,
                                                                                         dim=1).detach().cpu().numpy()
            pred_result_minus[batch_begin:batch_begin + data_1.size(0), :] = F.softmax(P_residue_1_i,
                                                                                       dim=1).detach().cpu().numpy()
            pred_label_generate[batch_begin:batch_begin + data_1.size(0)] = pred_current.detach().cpu().numpy()
            pred_label_minus[batch_begin:batch_begin + data_1.size(0)] = pred_minus.detach().cpu().numpy()
            target[batch_begin:batch_begin + data_1.size(0)] = target_ppa.detach().cpu().numpy()
            
            output_cur_res_1 = P_current_1_i[:, 1] + P_current_1_i[:, 0] * P_residue_1_i[:, 1]
            output_cur_res = torch.cat([1 - output_cur_res_1.unsqueeze(1), output_cur_res_1.unsqueeze(1)], dim=1)
            
            pred_cur_res = output_cur_res.argmax(dim=1, keepdim=True)
            pred_result_cur_res[batch_begin:batch_begin + data_1.size(0), :] = output_cur_res.detach().cpu().numpy()
            pred_label_cur_res[batch_begin:batch_begin + data_1.size(0)] = pred_cur_res.detach().cpu().numpy()
            
            output_average_temp = (P_current_1_i[:, 1] + P_current_1_i[:, 0] * P_residue_1_i[:, 1] + gen_pred[:,
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
                                      args.batch_size, args.dpi, args.is_local_normal)
                        
                        minus_feature1_1 = res_feature_1[i, :, :, :]
                        minus_feature1_1 = torch.unsqueeze(minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_feature1_1 = torch.sigmoid(minus_feature1_1)
                        minus_feature1_1 = minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_feature1_1, save_dir2, image_sequence, test_loader, 'minus_',
                                      args.batch_size, args.dpi, args.is_local_normal)
                        
                        minus_minus_feature1_1 = res_feature_1[i, :, :, :].mul(-1)
                        minus_minus_feature1_1 = torch.unsqueeze(minus_minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_minus_feature1_1 = torch.sigmoid(minus_minus_feature1_1)
                        minus_minus_feature1_1 = minus_minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_minus_feature1_1, save_dir2, image_sequence, test_loader,
                                      'minus_minus_',
                                      args.batch_size, args.dpi, args.is_local_normal)
                        
                        featuremap_1_1 = featuremap_1[i, :, :, :]
                        featuremap_1_1 = torch.unsqueeze(featuremap_1_1, 0)
                        if args.plot_sigmoid:
                            featuremap_1_1 = torch.sigmoid(featuremap_1_1)
                        featuremap_1_1 = featuremap_1_1.cpu().detach().numpy()
                        save_dir3 = (save_dir + 'real1_')
                        draw_features(16, 8, featuremap_1_1, save_dir3, image_sequence, test_loader, 'real1_',
                                      args.batch_size, args.dpi, args.is_local_normal)
                        
                        featuremap_2_1 = featuremap_2[i, :, :, :]
                        featuremap_2_1 = torch.unsqueeze(featuremap_2_1, 0)
                        if args.plot_sigmoid:
                            featuremap_2_1 = torch.sigmoid(featuremap_2_1)
                        featuremap_2_1 = featuremap_2_1.cpu().detach().numpy()
                        save_dir4 = (save_dir + 'real2_')
                        draw_features(16, 8, featuremap_2_1, save_dir4, image_sequence, test_loader, 'real2_',
                                      args.batch_size, args.dpi, args.is_local_normal)
                    
                    
                    elif target_ppa[i] == 0:
                        
                        save_dir = '../Feature_map/Ecur_%s-Label0/' % (args.sequence,)
                        
                        unet_feature1_1 = generate_feature_1[i, :, :, :]  # add the unnomarlization
                        unet_feature1_1 = torch.unsqueeze(unet_feature1_1, 0)
                        if args.plot_sigmoid:
                            unet_feature1_1 = torch.sigmoid(unet_feature1_1)
                        unet_feature1_1 = unet_feature1_1.cpu().detach().numpy()
                        save_dir1 = (save_dir + 'unet/')
                        draw_features(16, 8, unet_feature1_1, save_dir1, image_sequence, test_loader, 'unet_',
                                      args.batch_size, args.dpi, args.is_local_normal)
                        
                        minus_feature1_1 = res_feature_1[i, :, :, :]
                        minus_feature1_1 = torch.unsqueeze(minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_feature1_1 = torch.sigmoid(minus_feature1_1)
                        minus_feature1_1 = minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_feature1_1, save_dir2, image_sequence, test_loader, 'minus_',
                                      args.batch_size, args.dpi, args.is_local_normal)
                        
                        minus_minus_feature1_1 = res_feature_1[i, :, :, :].mul(-1)
                        minus_minus_feature1_1 = torch.unsqueeze(minus_minus_feature1_1, 0)
                        if args.plot_sigmoid:
                            minus_minus_feature1_1 = torch.sigmoid(minus_minus_feature1_1)
                        minus_minus_feature1_1 = minus_minus_feature1_1.cpu().detach().numpy()
                        save_dir2 = (save_dir + 'minus_')
                        draw_features(16, 8, minus_minus_feature1_1, save_dir2, image_sequence, test_loader,
                                      'minus_minus_',
                                      args.batch_size, args.dpi, args.is_local_normal)
                        
                        featuremap_1_1 = featuremap_1[i, :, :, :]
                        featuremap_1_1 = torch.unsqueeze(featuremap_1_1, 0)
                        if args.plot_sigmoid:
                            featuremap_1_1 = torch.sigmoid(featuremap_1_1)
                        featuremap_1_1 = featuremap_1_1.cpu().detach().numpy()
                        save_dir3 = (save_dir + 'real1_')
                        draw_features(16, 8, featuremap_1_1, save_dir3, image_sequence, test_loader, 'real1_',
                                      args.batch_size, args.dpi, args.is_local_normal)
                        
                        featuremap_2_1 = featuremap_2[i, :, :, :]
                        featuremap_2_1 = torch.unsqueeze(featuremap_2_1, 0)
                        if args.plot_sigmoid:
                            featuremap_2_1 = torch.sigmoid(featuremap_2_1)
                        featuremap_2_1 = featuremap_2_1.cpu().detach().numpy()
                        save_dir4 = (save_dir + 'real2_')
                        draw_features(16, 8, featuremap_2_1, save_dir4, image_sequence, test_loader, 'real2_',
                                      args.batch_size, args.dpi, args.is_local_normal)
    
    AUC_minus = sklearn.metrics.roc_auc_score(target, pred_result_minus[:, 1])
    acc_minus = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_minus, axis=1))
    cm_minus = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_minus, axis=1))
    sensitivity_minus = cm_minus[0, 0] / (cm_minus[0, 0] + cm_minus[0, 1])
    specificity_minus = cm_minus[1, 1] / (cm_minus[1, 0] + cm_minus[1, 1])
    
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


def save_results(args,
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
                 epoch):
    val_auc_average = (val_results_list[0]['AUC_average_all'] + val_results_list[1]['AUC_average_all'] +
                       val_results_list[2]['AUC_average_all'] + val_results_list[3]['AUC_average_all'] +
                       val_results_list[4]['AUC_average_all']) / 5
    val_acc_average = (val_results_list[0]['acc_average_all'] + val_results_list[1]['acc_average_all'] +
                       val_results_list[2]['acc_average_all'] + val_results_list[3]['acc_average_all'] +
                       val_results_list[4]['acc_average_all']) / 5
    test_auc_average = (test_results_list[0]['AUC_average_all'] + test_results_list[1]['AUC_average_all'] +
                        test_results_list[2]['AUC_average_all'] + test_results_list[3]['AUC_average_all'] +
                        test_results_list[4]['AUC_average_all']) / 5
    test_acc_average = (test_results_list[0]['acc_average_all'] + test_results_list[1]['acc_average_all'] +
                        test_results_list[2]['acc_average_all'] + test_results_list[3]['acc_average_all'] +
                        test_results_list[4]['acc_average_all']) / 5
    
    if args.best_test_acc < test_acc_average:
        args.best_test_acc = copy.deepcopy(test_acc_average)
        args.best_test_acc_epoch = copy.deepcopy(epoch)
    
    if args.best_test_auc < test_auc_average:
        args.best_test_auc = copy.deepcopy(test_auc_average)
        args.best_test_auc_epoch = copy.deepcopy(epoch)
    
    if args.best_val_acc < val_acc_average:
        args.best_val_acc = copy.deepcopy(val_acc_average)
        args.best_val_acc_epoch = copy.deepcopy(epoch)
    
    if args.best_val_auc < val_auc_average:
        args.best_val_auc = copy.deepcopy(val_auc_average)
        args.best_val_auc_epoch = copy.deepcopy(epoch)
    
    if epoch == args.best_test_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_model_1.pt'))
        torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_model_2_res.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_D_net.pt'))
        torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_G_net.pt'))
    if epoch == args.best_test_auc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_1.pt'))
        torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_2_res.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_D_net.pt'))
        torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_G_net.pt'))
    if epoch == args.best_val_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_1.pt'))
        torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_2_res.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_D_net.pt'))
        torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_G_net.pt'))
    if epoch == args.best_val_auc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_1.pt'))
        torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_2_res.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_D_net.pt'))
        torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_G_net.pt'))
    
    args.logger.info(
        'Utill now the best test acc epoch is : {},  acc is {}'.format(args.best_test_acc_epoch, args.best_test_acc))
    args.logger.info(
        'Utill now the best test AUC epoch is : {}, AUC is {}'.format(args.best_test_auc_epoch, args.best_test_auc))
    full_results[epoch] = {
        'train_results': copy.deepcopy(train_results),
        'test_results_list': copy.deepcopy(test_results_list),
        'val_results_list': copy.deepcopy(val_results_list),
    }
    pickle.dump(full_results, open(os.path.join(args.save_dir, 'results.pkl'), 'wb'))
    
    strs = 'cur_res'
    test_acc_mean = 0.0
    args.logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (args.best_val_acc_epoch, args.best_val_acc))
    for ss in range(len(test_results_list)):
        test_acc_mean = test_acc_mean + full_results[args.best_val_acc_epoch]['test_results_list'][ss][
            'acc_%s' % strs]
        args.logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_%s' % strs]))
    args.logger.info('mean test acc: %0.4f' % (test_acc_mean / len(test_results_list)))
    
    test_auc_mean = 0.0
    args.logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (args.best_val_auc_epoch, args.best_val_auc))
    for ss in range(len(test_results_list)):
        test_auc_mean = test_auc_mean + full_results[args.best_val_auc_epoch]['test_results_list'][ss][
            'AUC_%s' % strs]
        args.logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_%s' % strs]))
    args.logger.info('mean test auc: %0.4f' % (test_auc_mean / len(test_results_list)))
    
    strs = 'gen'
    test_acc_mean = 0.0
    args.logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (args.best_val_acc_epoch, args.best_val_acc))
    for ss in range(len(test_results_list)):
        test_acc_mean = test_acc_mean + full_results[args.best_val_acc_epoch]['test_results_list'][ss][
            'acc_%s' % strs]
        args.logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_%s' % strs]))
    args.logger.info('mean test acc: %0.4f' % (test_acc_mean / len(test_results_list)))
    
    test_auc_mean = 0.0
    args.logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (args.best_val_auc_epoch, args.best_val_auc))
    for ss in range(len(test_results_list)):
        test_auc_mean = test_auc_mean + full_results[args.best_val_auc_epoch]['test_results_list'][ss][
            'AUC_%s' % strs]
        args.logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_%s' % strs]))
    args.logger.info('mean test auc: %0.4f' % (test_auc_mean / len(test_results_list)))
    
    strs = 'average_all'
    test_acc_mean = 0.0
    args.logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (args.best_val_acc_epoch, args.best_val_acc))
    for ss in range(len(test_results_list)):
        test_acc_mean = test_acc_mean + full_results[args.best_val_acc_epoch]['test_results_list'][ss][
            'acc_%s' % strs]
        args.logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_%s' % strs]))
    args.logger.info('mean test acc: %0.4f' % (test_acc_mean / len(test_results_list)))
    
    test_auc_mean = 0.0
    args.logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (args.best_val_auc_epoch, args.best_val_auc))
    for ss in range(len(test_results_list)):
        test_auc_mean = test_auc_mean + full_results[args.best_val_auc_epoch]['test_results_list'][ss][
            'AUC_%s' % strs]
        args.logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_%s' % strs]))
    args.logger.info('mean test auc: %0.4f' % (test_auc_mean / len(test_results_list)))
    
    is_best = 1
    if args.save_checkpoint > 0:
        save_checkpoint({
            'epoch': copy.deepcopy(epoch),
            'model_1': model_1.state_dict(),
            'model_2_generate': model_2_generate.state_dict(),
            'model_2_minus': model_2_res.state_dict(),
            'G_net': G_net.state_dict(),
            'D_net': D_net.state_dict(),
            'best_test_acc': args.best_test_acc,
            'optimizer_M_2_generate': optimizer_M_2_generate.state_dict(),
            'optimizer_M_2_res': optimizer_M_2_res.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        }, is_best, base_dir=args.save_dir)
        

def main():
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader_list, test_loader_list = get_all_dataloader(args)
    
    model_1 = RN18_front().cuda()
    model_2_res = RN18_last(num_classes = args.class_num, dropout=args.dropout, dp=args.dp).cuda()
    model_2_generate = RN18_last(num_classes = args.class_num, dropout=0).cuda()
    
    load_pytorch_model(model_1,
                       r'../results/Ecur_200616_BS/2020-06-29-23-50-10_Mixed_0.010000_30.000000_size_256_ep_120_0_R_Maculae/best_val_auc_model_1.pt')
    load_pytorch_model(model_2_generate,
                       r'../results/Ecur_200616_BS/2020-06-29-23-50-10_Mixed_0.010000_30.000000_size_256_ep_120_0_R_Maculae/best_val_auc_model_2_generate.pt')
    
    G_net = Generator(feature_num=len(args.feature_list)+1, is_ESPCN=args.is_ESPCN, mid_channel=args.dw_midch).cuda()
    D_net = Discriminator(128).cuda()
    
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
                                  train_loader,optimizer_M_2_generate, optimizer_M_2_res, optimizer_G, optimizer_D,
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
            save_results(args,
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
        args.logger.info('save_results_path: %s'%args.save_dir)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
if __name__ == '__main__':
    main()
