# -*- coding: UTF-8 -*-
import torch
import torch.utils.data as data
import numpy as np
import os
import xlrd
from PIL import Image
import copy
import os
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import sklearn.metrics
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
from torchvision import transforms
import datetime
import torch
from torchvision import models
import cv2
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision import utils
import torch.autograd as autograd
import argparse

class Clas_ppa_train(data.Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 fold='train',
                 eye=0,
                 center=0,
                 label=0,
                 filename='std',
                 feature_list=[8, 9],
                 is_endwith6 = 0,
                 ):
        super(Clas_ppa_train, self).__init__()
        self.root = root  # excel path
        self.transform = transform
        self.eye = eye
        self.center = center
        self.label = label
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
        
        if filename == 'no_normal':
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_06-19-order1-no_normalize-8.xls")
        if filename == 'minmax':
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_06-19-order1-minmax-8.xls")
        if filename == 'std':
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_08-09-order1-std-deletetrain1234-1.xls")
        if filename == 'std-456':
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_08-16-order1-std-45to6.xls")
        if filename == 'std-3456':
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_08-16-order1-std-345to6.xls")
        if filename == 'std-23456':
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_08-16-order1-std-2345to6.xls")
        if filename == 'std-ori':
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_06-29-order1-std-8.xls")
        
        sheet1 = workbook1.sheet_by_index(0)
        
        for rows in range(1, sheet1.nrows):
            if sheet1.row_values(rows)[5] == fold:
                if sheet1.row_values(rows)[3] in self.eye:
                    if str(sheet1.row_values(rows)[4]) in self.center:
                        if is_endwith6 == 1:
                            if int(sheet1.row_values(rows)[2].split('/')[0][5]) == 6:
                                self.image_path_all_1.append(os.path.join(self.root, sheet1.row_values(rows)[1]))
                                self.image_path_all_2.append(os.path.join(self.root, sheet1.row_values(rows)[2]))
                                self.target_ppa.append(sheet1.row_values(rows)[6])
                                # print(np.array(sheet1.row_values(rows))[self.feature_mask_1])
                                self.feature_all_1.append(
                                    np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))
                                self.feature_all_2.append(
                                    np.array(sheet1.row_values(rows))[self.feature_mask_2].astype('float32'))
                                self.base_grade_num.append(int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                                self.base_grade_num_2.append(int(sheet1.row_values(rows)[2].split('/')[0][-1]))
                        else:
                            self.image_path_all_1.append(os.path.join(self.root, sheet1.row_values(rows)[1]))
                            self.image_path_all_2.append(os.path.join(self.root, sheet1.row_values(rows)[2]))
                            self.target_ppa.append(sheet1.row_values(rows)[6])
                            # print(np.array(sheet1.row_values(rows))[self.feature_mask_1])
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
        Clas_ppa_train(args.data_root, fold='train', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           #transforms.RandomHorizontalFlip(p=0.5),
                           #transforms.RandomVerticalFlip(p=0.5),
                           transforms.RandomRotation(30),
                           transforms.ToTensor(),

                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test2', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val2', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test3', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val3', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test4', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val4', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test5', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val5', eye=args.eye, center=args.center, label=args.label,
                       filename=args.filename, feature_list=args.feature_list, is_endwith6=args.is_endwith6,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    
    return train_loader, val_loader_list, test_loader_list




def get_opts():
    parser = argparse.ArgumentParser(description='PyTorch PrimarySchool Fundus photography RA-Regression')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--lr2', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default= -1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=24, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='')
    parser.add_argument('--lr_controler', type=int, default=30, )
    parser.add_argument('--wd', type=float, default=0.0001, help='')
    parser.add_argument('--wd2', type=float, default=0.0001, help='')
    
    parser.add_argument('--isplus', type=int, default=0)
    parser.add_argument('--D_epoch', type=int, default=4)
    parser.add_argument('--dataset_type', type=str, default='baseline')
    parser.add_argument('--load_model_1', type=str, default='nbase_2')
    parser.add_argument('--load_model_pre', type=str, default='baseline')
    parser.add_argument('--G_net_type', type=str, default='G_net')
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--final_tanh', type=int, default=0)
    parser.add_argument('--train_ori_data', type=int, default=0)
    
    parser.add_argument('--pred_future', type=int, default=1)
    parser.add_argument('--xlsx_name', type=str, default='')
    
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--works', type=int, default=2)
    parser.add_argument('--is_plot', type=int, default=0)
    parser.add_argument('--plot_sigmoid', type=int, default=0)

    
    # Dataset
    parser.add_argument('--data_root', type=str, default='../data/JPG256SIZE-95')
    parser.add_argument('--filename', type=str, default='std_all')
    parser.add_argument('--feature_list', type=int, nargs='+', default=[8, 9], help='')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    # sijie
    parser.add_argument('--eye', type=str, default='R',
                        help='choose the right or left eye')
    parser.add_argument('--center', type=str, default='Maculae',
                        help='choose the center type pf the picture,input(Maculae)(Disc)(Maculae&Disc) ')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='set the error margin')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='choose the optimizer function')
    parser.add_argument('--label', type=str, default='RA',
                        help='choose the label we want to do research')
    parser.add_argument('--wcl', type=float, default=0.01,
                        help='control the clamp parameter of Dnet, weight_cliping_limit')
    parser.add_argument('--critic_iter', type=int, default=5,
                        help='control the critic iterations for updating Gnet')
    parser.add_argument('--noise_size', type=int, default=516,
                        help='control the size of the input of Gnet')
    parser.add_argument('--save_checkpoint', type=int, default=1,
                        help='save the check point of the model and optim')
    
    # params
    parser.add_argument('--alpha', type=float, default=1,
                        help='parameter of loss image')
    parser.add_argument('--beta', type=float, default=1,
                        help='parameter of fake loss RA prediction')
    parser.add_argument('--beta1', type=float, default=1,
                        help='parameter of fake loss RA prediction')
    parser.add_argument('--gamma', type=float, default=1,
                        help='parameter of real loss RA prediction')
    parser.add_argument('--delta1', type=float, default=0.001,
                        help='parameter of real loss RA prediction')
    parser.add_argument('--sequence', type=str, default='5-separate-1',
                        help='save the check point of the model and optim')
    parser.add_argument('--betas1', type=float, default='0.5',
                        help='hyper-parameter for Adam')
    parser.add_argument('--betas2', type=float, default='0.9',
                        help='hyper-parameter for Adam')
    parser.add_argument('--LAMBDA', type=float, default='10',
                        help='hyper-parameter for gradient panelty')
    parser.add_argument('--bi_linear', type=int, default=0,
                        help='whether use the bilinear mode in Unet')
    # sijie add for debugging biggan
    parser.add_argument('--gantype', type=str, default='Big',
                        help='hyper-parameter for gradient panelty')
    parser.add_argument('--discritype', type=str, default='normal',
                        help='hyper-parameter for gradient panelty')
    parser.add_argument('--SCR', type=int, default=0,
                        help='hyper-parameter for gradient panelty')
    # sijie for generate test
    parser.add_argument('--load_dir', type=str, default='/home')
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--dw_type', type=str, default='conv',
                        help='choose the downsample type of ESPCN')
    parser.add_argument('--dw_midch', type=int, default=1024,
                        help='choose the mid-channel type of ESPCN')
    parser.add_argument('--scale_factor', type=int, default=2,
                        help='choose the mid-channel type of ESPCN')
    parser.add_argument('--lr3', type=float, default=0.001,
                        help='lr3 for ESPCN')
    parser.add_argument('--is_ESPCN', type=int, default=0,
                        help='whether load model ESPCN')

    parser.add_argument('--is_endwith6', type=int, default=0,
                        help='if make the dataset end with 6')
    parser.add_argument('--lambda_R', type=float, default=0.01,
                        help='if make the dataset end with 6')
    parser.add_argument('--lambda_C', type=float, default=0.01,
                        help='if make the dataset end with 6')
    parser.add_argument('--lossC', type=float, default=1,
                        help='if make the dataset end with 6')
    parser.add_argument('--lossR', type=float, default=1,
                        help='if make the dataset end with 6')
    parser.add_argument('--modelR', type=str, default='half')
    parser.add_argument('--Rinch', type=int, default=32)
    parser.add_argument('--Routch', type=int, default=32)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--dp', type=float, default=0.5)
    
    
    
    args = parser.parse_args()
    # botong: save the full_results
    time_stamp = datetime.datetime.now()
    save_dir = '../results/Ecur_%s/%s_%s_%f_%f_size_%s_ep_%d_%d_%s_%s/' % (
        args.sequence,
        time_stamp.strftime(
            '%Y-%m-%d-%H-%M-%S'),
        args.optimizer,
        args.lr,
        args.lr_controler,
        args.image_size,
        args.epochs,
        args.wcl, args.eye,
        args.center)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # print(args.save_model)
    args.save_dir = save_dir
    args.logger = get_logger(args)
    
    return args

def load_pytorch_model(model, path):
    from collections import OrderedDict
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict,  strict=False)###parameter strict: do not load the part dont match
    return model

def get_logger(opt, fold=0):
    import logging
    logger = logging.getLogger('AL')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler('{}training_{}.log'.format(opt.save_dir, fold))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_metric(args):
    args.best_test_acc = -1
    args.best_test_auc = -1
    args.best_val_acc = -1
    args.best_val_auc = -1

    args.best_test_acc_1 = -1
    args.best_test_auc_1 = -1
    args.best_val_acc_1 = -1
    args.best_val_auc_1 = -1
    args.best_val_auc_1_epoch = 0
    args.best_val_acc_1_epoch = 0

    args.best_test_acc_2 = -1
    args.best_test_auc_2 = -1
    args.best_val_acc_2 = -1
    args.best_val_auc_2 = -1
    args.best_val_auc_2_epoch = 0
    args.best_val_acc_2_epoch = 0

    args.best_test_acc_3 = -1
    args.best_test_auc_3 = -1
    args.best_val_acc_3 = -1
    args.best_val_auc_3 = -1
    args.best_val_auc_3_epoch = 0
    args.best_val_acc_3_epoch = 0

    args.best_test_acc_4 = -1
    args.best_test_auc_4 = -1
    args.best_val_acc_4 = -1
    args.best_val_auc_4 = -1
    args.best_val_auc_4_epoch = 0
    args.best_val_acc_4_epoch = 0

    args.best_test_acc_5 = -1
    args.best_test_auc_5 = -1
    args.best_val_acc_5 = -1
    args.best_val_auc_5 = -1
    args.best_val_auc_5_epoch = 0
    args.best_val_acc_5_epoch = 0

    args.best_test_acc_list = [args.best_test_acc_1, args.best_test_acc_2, args.best_test_acc_3, args.best_test_acc_4,
                               args.best_test_acc_5]
    args.best_test_auc_list = [args.best_test_auc_1, args.best_test_auc_2, args.best_test_auc_3, args.best_test_auc_4,
                               args.best_test_auc_5]

    args.best_val_acc_list = [args.best_val_acc_1, args.best_val_acc_2, args.best_val_acc_3, args.best_val_acc_4,
                               args.best_val_acc_5]
    args.best_val_auc_list = [args.best_val_auc_1, args.best_val_auc_2, args.best_val_auc_3, args.best_val_auc_4,
                               args.best_val_auc_5]

    args.best_val_auc_epoch_list = [args.best_val_auc_1_epoch, args.best_val_auc_2_epoch, args.best_val_auc_3_epoch,
                                      args.best_val_auc_4_epoch, args.best_val_auc_5_epoch]
    args.best_val_acc_epoch_list = [args.best_val_acc_1_epoch, args.best_val_acc_2_epoch, args.best_val_acc_3_epoch,
                                    args.best_val_acc_4_epoch, args.best_val_acc_5_epoch]
    return args

def save_results(args,
                 model_C,
                 model_R,
                 G_net1, G_net2,
                 D_net,
                 train_results,
                 val_results_list,
                 test_results_list,
                 full_results,
                 optimizer_C,
                 optimizer_R,
                 optimizer_G1, optimizer_G2,
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
        torch.save(model_R.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_model_R.pt'))
        torch.save(model_C.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_D_net.pt'))
        torch.save(G_net1.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_G_net1.pt'))
        torch.save(G_net2.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_G_net2.pt'))
    if epoch == args.best_test_auc_epoch:
        torch.save(model_R.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_R.pt'))
        torch.save(model_C.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_D_net.pt'))
        torch.save(G_net1.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_G_net1.pt'))
        torch.save(G_net2.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_G_net2.pt'))
    if epoch == args.best_val_acc_epoch:
        torch.save(model_R.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_R.pt'))
        torch.save(model_C.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_D_net.pt'))
        torch.save(G_net1.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_G_net1.pt'))
        torch.save(G_net2.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_G_net2.pt'))
    if epoch == args.best_val_auc_epoch:
        torch.save(model_R.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_R.pt'))
        torch.save(model_C.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_D_net.pt'))
        torch.save(G_net1.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_G_net1.pt'))
        torch.save(G_net2.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_G_net2.pt'))
    
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

    
    test_acc_mean = 0.0
    args.logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (args.best_val_acc_epoch, args.best_val_acc))
    for ss in range(len(test_results_list)):
        test_acc_mean = test_acc_mean + full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_average_all']
        args.logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_average_all']))
    args.logger.info('mean test acc: %0.4f' % (test_acc_mean / 5))
    
    test_auc_mean = 0.0
    args.logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (args.best_val_auc_epoch, args.best_val_auc))
    for ss in range(len(test_results_list)):
        test_auc_mean = test_auc_mean + full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_average_all']
        args.logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_average_all']))
    args.logger.info('mean test auc: %0.4f' % (test_auc_mean / 5))
    



    is_best = 1
    if args.save_checkpoint > 0:
        save_checkpoint({
            'epoch': copy.deepcopy(epoch),
            'model_2_generate': model_C.state_dict(),
            'model_R': model_R.state_dict(),
            'G_net1': G_net1.state_dict(),
            'G_net2': G_net2.state_dict(),
            'D_net': D_net.state_dict(),
            'best_test_acc': args.best_test_acc,
            'optimizer_C': optimizer_C.state_dict(),
            'optimizer_R': optimizer_R.state_dict(),
            'optimizer_G1': optimizer_G1.state_dict(),
            'optimizer_G2': optimizer_G2.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        }, is_best, base_dir=args.save_dir)
        #torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'Final_model_1.pt'))
        #torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'Final_model_2_generate.pt'))
        #torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'Final_model_2_res.pt'))
        #torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'Final_G_net.pt'))
        #torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'Final_D_net.pt'))


def train(args,
          model_C,
          G_net1,
          G_net2,
          D_net,
          model_R,
          train_loader,
          optimizer_C,
          optimizer_G1, optimizer_G2,
          optimizer_D,
          optimizer_R,
          epoch):
    

    train_loss_D = AverageMeter()
    train_loss_G = AverageMeter()
    train_loss_M2_reg_mono = AverageMeter()
    train_loss_M2_reg = AverageMeter()
    train_loss_C = AverageMeter()
    train_loss_R = AverageMeter()
    train_loss_M2_gen_cls = AverageMeter()
    eps = 1e-5

    pred_result_minus = np.zeros((len(train_loader.dataset), args.class_num))
    image_sequence = 0
    acc_num = 0
    num = 0
    for batch_idx, (data_1, data_2, target_ppa, feature_1, feature_2, grad_1, grad_2) in enumerate(train_loader):
        data_1, data_2, target_ppa, feature_1, feature_2, grad_1, grad_2 = \
            data_1.cuda(), data_2.cuda(), target_ppa.cuda(), feature_1.cuda(), \
            feature_2.cuda(), grad_1.cuda(), grad_2.cuda()

        if batch_idx % 5 < args.D_epoch:
            image_sequence += data_1.size(0)

            for p in G_net2.parameters():
                p.requires_grad = False
            for p in G_net1.parameters():
                p.requires_grad = False
            for p in D_net.parameters():
                p.requires_grad = True
            for p in model_C.parameters():
                p.requires_grad = True
            for p in model_R.parameters():
                p.requires_grad = True

            

            target_ppa = target_ppa.reshape(-1, 1)

            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            y1 = torch.randn(data_1.size(0), 100).cuda()
            #y1 = torch.sign(y1)
            generate_feature_1, y_hat = G_net1(z1, y1)
            y_hat1 = torch.sign(y_hat).long()
            
            z2 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            y2 = torch.randn(data_1.size(0), 100).cuda()
            #y2 = torch.sign(y2)
            generate_feature_2, y_hat = G_net2(z2, y2)
            y_hat2 = torch.sign(y_hat).long()
            
            #reg_feature_real = model_R(data_1)
            #reg_feature_gene = model_R(generate_feature_1)

            #pred_t = model_C(data_1).argmax(dim=1, keepdim=True)
            pred_t1 = model_C(data_2)
            #pred_Gt = model_C(generate_feature_1)#.argmax(dim=1, keepdim=True)
            pred_Gt1 = model_C(generate_feature_2)

            real_loss_1 = D_net(data_1).mean(0).view(1)
            fake_loss_1 = D_net(generate_feature_1).mean(0).view(1)

            real_loss_2 = D_net(data_2).mean(0).view(1)
            fake_loss_2 = D_net(generate_feature_2).mean(0).view(1)

            loss_D = (real_loss_1 - fake_loss_1) + (real_loss_2 - fake_loss_2)
            optimizer_D.zero_grad()
            for p in D_net.parameters():
                p.data.clamp_(-args.wcl, args.wcl)
            loss_D.backward(retain_graph=False)
            optimizer_D.step()

            R_num = 0
            loss_R_1 = 0
            loss_R_2 = 0
            loss_C_data1 = 0
            for i in range(data_1.size(0)):
                R_num += 1
                num += 1
                deltaT = int(grad_2[i].detach().cpu().numpy()) - int(grad_1[i].detach().cpu().numpy())
                data_R_real = data_1[i].unsqueeze(0)
                data_R_gene = generate_feature_1[i].unsqueeze(0)
                for ii in range(deltaT):
                    data_R_real = model_R(data_R_real)
                    data_R_gene = model_R(data_R_gene)
                loss_R_1 += F.l1_loss(data_R_real, data_2[i].unsqueeze(0))
                loss_R_2 += F.l1_loss(generate_feature_2[i].unsqueeze(0), data_R_gene)
                pred = torch.softmax(model_C(data_R_real), dim=1).argmax(dim=1, keepdim=True)
                loss_C_data1 += F.cross_entropy(model_C(data_R_real), target_ppa[i])
                if pred.detach().cpu().numpy() == target_ppa[i].squeeze().detach().cpu().numpy():
                    acc_num += 1
            loss_C_data1 /= R_num
            loss_R_1 /= R_num
            loss_R_2 /= R_num


            loss_G =  fake_loss_1 + fake_loss_2
            loss_R = loss_R_1 + args.lambda_R * loss_R_2
            loss_C = F.cross_entropy(pred_t1, target_ppa.squeeze()) + args.lambda_C * F.cross_entropy(pred_Gt1, y_hat2.squeeze()) + loss_C_data1
            loss = args.lossC * loss_C + args.lossR * loss_R

            optimizer_C.zero_grad()
            optimizer_R.zero_grad()
            loss.backward(retain_graph=False)
            optimizer_C.step()
            optimizer_R.step()

            train_loss_G.update(loss_G.item(), data_1.size(0))
            train_loss_D.update(loss_D.item(), 2 * data_1.size(0))
            train_loss_C.update(loss_C.item(), data_1.size(0))
            train_loss_R.update(loss_C.item(), data_1.size(0))
            
        if batch_idx % 5 >= args.D_epoch:
            image_sequence += data_1.size(0)

            for p in G_net2.parameters():
                p.requires_grad = True
            for p in G_net1.parameters():
                p.requires_grad = True
            for p in D_net.parameters():
                p.requires_grad = False
            for p in model_C.parameters():
                p.requires_grad = True
            for p in model_R.parameters():
                p.requires_grad = True

            target_ppa = target_ppa.reshape(-1, 1)

            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            y1 = torch.randn(data_1.size(0), 100).cuda()
            #y1 = torch.sign(y1)
            generate_feature_1, y_hat = G_net1(z1, y1)
            y_hat1 = torch.sign(y_hat).long()
            
            z2 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            y2 = torch.randn(data_1.size(0), 100).cuda()
            #y2 = torch.sign(y2)
            generate_feature_2, y_hat = G_net2(z2, y2)
            y_hat2 = torch.sign(y_hat).long()
            
            #reg_feature_real = model_R(data_1)
            #reg_feature_gene = model_R(generate_feature_1)

            #pred_t = model_C(data_1)  # .argmax(dim=1, keepdim=True)
            pred_t1 = model_C(data_2)

            #pred_Gt = model_C(generate_feature_1)  # .argmax(dim=1, keepdim=True)
            pred_Gt1 = model_C(generate_feature_2)
            
            real_loss_1 = D_net(data_1).mean(0).view(1)
            fake_loss_1 = D_net(generate_feature_1).mean(0).view(1)

            real_loss_2 = D_net(data_2).mean(0).view(1)
            fake_loss_2 = D_net(generate_feature_2).mean(0).view(1)

            R_num = 0
            loss_R_1 = 0
            loss_R_2 = 0
            loss_C_data1 = 0
            for i in range(data_1.size(0)):
                R_num += 1
                num += 1
                deltaT = int(grad_2[i].detach().cpu().numpy()) - int(grad_1[i].detach().cpu().numpy())
                data_R_real = data_1[i].unsqueeze(0)
                data_R_gene = generate_feature_1[i].unsqueeze(0)
                for ii in range(deltaT):
                    data_R_real = model_R(data_R_real)
                    data_R_gene = model_R(data_R_gene)
                loss_R_1 += F.l1_loss(data_R_real, data_2[i].unsqueeze(0))
                loss_R_2 += F.l1_loss(generate_feature_2[i].unsqueeze(0), data_R_gene)
                # print('deltaT', deltaT)
                # print('data_1[i]', data_1[i].max(), data_1[i].min())
                # print('data_2[i]', data_2[i].max(), data_2[i].min())
                # print('data_R_real', data_R_real.max(), data_R_real.min(), F.l1_loss(data_R_real, data_2[i]))
                pred = torch.softmax(model_C(data_R_real), dim=1).argmax(dim=1, keepdim=True)
                loss_C_data1 += F.cross_entropy(model_C(data_R_real), target_ppa[i])
                if pred.detach().cpu().numpy() == target_ppa[i].squeeze().detach().cpu().numpy():
                    acc_num += 1
            loss_C_data1 /= R_num
            loss_R_1 /= R_num
            loss_R_2 /= R_num

            loss_D = (real_loss_1 - fake_loss_1) + (real_loss_2 - fake_loss_2)
            loss_R = loss_R_1 + args.lambda_R * loss_R_2
            loss_C = F.cross_entropy(pred_t1, target_ppa.squeeze()) + args.lambda_C * F.cross_entropy(pred_Gt1, y_hat2.squeeze()) + loss_C_data1
            loss_G = fake_loss_1 + fake_loss_2

            loss = args.lossC * loss_C + loss_G + args.lossR * loss_R
            #print(loss)
            optimizer_G1.zero_grad()
            optimizer_G2.zero_grad()
            optimizer_C.zero_grad()
            optimizer_R.zero_grad()
            loss.backward(retain_graph=False)
            optimizer_R.step()
            optimizer_C.step()
            optimizer_G2.step()
            optimizer_G1.step()

            train_loss_D.update(loss_D.item(), 2 * data_1.size(0))
            train_loss_G.update(loss_G.item(), data_1.size(0))
            train_loss_C.update(loss_C.item(), data_1.size(0))
            train_loss_R.update(loss_C.item(), data_1.size(0))

        print('acc for train is :', acc_num/num)
        args.logger.info('Model Train Epoch: {} [{}/{} ({:.0f}%)] loss_G: {:.4f}, '
                         'loss_D: {:.4f}, loss_R: {:.4f}, loss_C: {:.4f}'.format(
            epoch, batch_idx * len(data_1), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), train_loss_G.avg,
            train_loss_D.avg, train_loss_R.avg, train_loss_C.avg))
        
        #args.logger.info('loss_D is real pred RA loss: {}'.format(train_loss_D.avg))
    
    loss = {
        'loss_D': train_loss_D.avg,
        'loss_G': train_loss_G.avg,
        'loss_R': train_loss_R.avg,
        'loss_C': train_loss_C.avg,
    }
    return loss

def evaluate(args,
          model_R,
          model_C,
          G_net1,
          test_loader,
          epoch,
          deltaT):
    model_C.eval()
    model_R.eval()
    G_net1.eval()
    
    
    
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
            
            reg = data_1
            for ii in range(deltaT):
                #print('{} :{}'.format(ii, reg))
                reg = model_R(reg)
            '''
            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            y1 = torch.randn(data_1.size(0), 100).cuda()
            y1 = torch.sign(y1)
            generate_feature_1, y_hat = G_net1(z1, y1)
            y_hat1 = torch.sign(y_hat).long()
            '''
            
            P_residue_1_i = torch.softmax(model_C(reg), dim=1)
            #print('predict values',model_C(generate_feature_1))
            #print('predict softmax',torch.softmax(model_C(generate_feature_1), dim=1))
            #print('data',data_1)
            #print('generate',generate_feature_1)

            pred_minus = P_residue_1_i.argmax(dim=1, keepdim=True)

            correct_minus += pred_minus.eq(target_ppa.view_as(pred_minus)).sum().item()

            #pred_result_minus[batch_begin:batch_begin + data_1.size(0), :] = F.softmax(P_residue_1_i,
                                                                                       #dim=1).detach().cpu().numpy()
            pred_result_minus[batch_begin:batch_begin + data_1.size(0), :] = P_residue_1_i.detach().cpu().numpy()

            pred_label_minus[batch_begin:batch_begin + data_1.size(0)] = pred_minus.detach().cpu().numpy()
            target[batch_begin:batch_begin + data_1.size(0)] = target_ppa.detach().cpu().numpy()
            

            
            for i in range(data_1.size(0)):
                name.append(test_loader.dataset.image_path_all_1[batch_begin + i])
            
            batch_begin = batch_begin + data_1.size(0)
            
            
    print('acc for test using correct_num is :', correct_minus)
    AUC_minus = sklearn.metrics.roc_auc_score(target, pred_result_minus[:, 1])
    acc_minus = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_minus, axis=1))
    cm_minus = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_minus, axis=1))
    sensitivity_minus = cm_minus[0, 0] / (cm_minus[0, 0] + cm_minus[0, 1])
    specificity_minus = cm_minus[1, 1] / (cm_minus[1, 0] + cm_minus[1, 1])
    

    #args.logger.info('In epoch {} for generate, AUC is {}, acc is {}.'.format(epoch, AUC_gen, acc_gen))
    args.logger.info(
        'In epoch {} for minus, AUC is {}, acc is {}, loss is {}'.format(epoch, AUC_minus, acc_minus, test_loss_minus))

    args.logger.info('      ')
    
    results = {
        'AUC_average_all': AUC_minus,
        'acc_average_all': acc_minus,
        'sensitivity_minus': sensitivity_minus,
        'specificity_minus': specificity_minus,
        'pred_result_minus': pred_result_minus,
        'pred_label_minus': pred_label_minus,

        
        'target': target,
        'image_path': name,
        
    }
    return results


def save_results_baseline(args,
                          model_1,
                          model_2_generate,
                          train_results,
                          val_results_list,
                          test_results_list,
                          full_results,
                          optimizer_M_1,
                          optimizer_M_2_generate,
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
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_model_2_generate.pt'))
    if epoch == args.best_test_auc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_1.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_2_generate.pt'))
    if epoch == args.best_val_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_1.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_2_generate.pt'))
    if epoch == args.best_val_auc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_1.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_2_generate.pt'))
    
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
    
    if epoch == args.epochs:
        save_results_as_xlsx(root='', args=args)
        pass
    
    is_best = 1
    if args.save_checkpoint > 0:
        save_checkpoint({
            'epoch': copy.deepcopy(epoch),
            'model_1': model_1.state_dict(),
            'model_2_generate': model_2_generate.state_dict(),
            'best_test_acc': args.best_test_acc,
            'optimizer_M_2_generate': optimizer_M_2_generate.state_dict(),
            'optimizer_M_1': optimizer_M_1.state_dict(),
        }, is_best, base_dir=args.save_dir)
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'Final_model_1.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'Final_model_2_generate.pt'))

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_decay ** (epoch // args.lr_controler))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', base_dir='./'):
    torch.save(state, os.path.join(base_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(base_dir, filename), os.path.join(base_dir, 'model_best.pth.tar'))


import xlwt
import xlrd
from xlutils.copy import copy as x_copy


def save_results_as_xlsx(root, args=None):
    if not os.path.exists(os.path.dirname(args.xlsx_name)):
        os.makedirs(os.path.dirname(args.xlsx_name))
        pass
    
    if not os.path.exists(os.path.join(root, args.xlsx_name)):
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet('Sheet1')
        worksheet.write(0, 0, 'exp_index')
        worksheet.write(0, 1, 'lr')
        worksheet.write(0, 2, 'wd')
        worksheet.write(0, 3, 'epochs')
        # worksheet.write(0, 4, 'loss')
        worksheet.write(0, 5, 'lr_decay')
        worksheet.write(0, 6, 'lr_controler')
        worksheet.write(0, 7, 'alpha')
        worksheet.write(0, 8, 'lambda_1')
        worksheet.write(0, 9, 'lambda_2')
        worksheet.write(0, 10, 'theta')
        worksheet.write(0, 11, 'seed')
        worksheet.write(0, 12, 'val_acc')
        worksheet.write(0, 13, 'val_auc')
        worksheet.write(0, 14, 'test_acc')
        worksheet.write(0, 15, 'test_auc')
        worksheet.write(0, 16, 'results_root')
        worksheet.write(0, 17, 'optimizer')
        
        idx = 1
        worksheet.write(idx, 0, 1)
        worksheet.write(idx, 1, args.lr)
        worksheet.write(idx, 2, args.wd)
        worksheet.write(idx, 3, args.epochs)
        # worksheet.write(idx, 4, args.loss)
        worksheet.write(idx, 5, args.lr_decay)
        worksheet.write(idx, 6, args.lr_controler)
        worksheet.write(idx, 7, args.alpha)
        worksheet.write(idx, 8, args.gamma)
        #worksheet.write(idx, 9, args.lambda2)
        worksheet.write(idx, 10, args.delta1)
        worksheet.write(idx, 11, args.seed)
        worksheet.write(idx, 12, args.best_val_acc)
        worksheet.write(idx, 13, args.best_val_auc)
        worksheet.write(idx, 14, args.best_test_acc)
        worksheet.write(idx, 15, args.best_test_auc)
        worksheet.write(idx, 16, args.save_dir)
        worksheet.write(idx, 17, args.optimizer)
        
        workbook.save(os.path.join(root, args.xlsx_name))
    else:
        rb = xlrd.open_workbook(os.path.join(root, args.xlsx_name))
        wb = x_copy(rb)
        worksheet = wb.get_sheet(0)
        idx = len(worksheet.get_rows())
        
        worksheet.write(idx, 0, 1)
        worksheet.write(idx, 1, args.lr)
        worksheet.write(idx, 2, args.wd)
        worksheet.write(idx, 3, args.epochs)
        # worksheet.write(idx, 4, args.loss)
        worksheet.write(idx, 5, args.lr_decay)
        worksheet.write(idx, 6, args.lr_controler)
        worksheet.write(idx, 7, args.alpha)
        worksheet.write(idx, 8, args.gamma)
        #worksheet.write(idx, 9, args.lambda2)
        worksheet.write(idx, 10, args.delta1)
        worksheet.write(idx, 11, args.seed)
        worksheet.write(idx, 12, args.best_val_acc)
        worksheet.write(idx, 13, args.best_val_auc)
        worksheet.write(idx, 14, args.best_test_acc)
        worksheet.write(idx, 15, args.best_test_auc)
        worksheet.write(idx, 16, args.save_dir)
        worksheet.write(idx, 17, args.optimizer)
        
        wb.save(os.path.join(root, args.xlsx_name))


import numpy as np
import torch
from torch.autograd import Variable
import logging


def print_model_parm_nums(model, logger=None):
    if logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('test')
    total = sum([param.nelement() for param in model.parameters()])
    logger.info('  + Number of params: %.2fM' % (total / 1e6))


def print_model_parm_flops(model, logger=None):
    if logger is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('test')
    # prods = {}
    # def save_prods(self, input, output):
    # print 'flops:{}'.format(self.__class__.__name__)
    # print 'input:{}'.format(input)
    # print '_dim:{}'.format(input[0].dim())
    # print 'input_shape:{}'.format(np.prod(input[0].shape))
    # grads.append(np.prod(input[0].shape))
    
    prods = {}
    
    def save_hook(name):
        def hook_per(self, input, output):
            # print 'flops:{}'.format(self.__class__.__name__)
            # print 'input:{}'.format(input)
            # print '_dim:{}'.format(input[0].dim())
            # print 'input_shape:{}'.format(np.prod(input[0].shape))
            # prods.append(np.prod(input[0].shape))
            prods[name] = np.prod(input[0].shape)
            # prods.append(np.prod(input[0].shape))
        
        return hook_per
    
    list_1 = []
    
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    
    list_2 = {}
    
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)
    
    multiply_adds = False
    list_conv = []
    
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width, input_depth = input[0].size()
        output_channels, output_height, output_width, output_depth = output[0].size()
        
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * (
                    self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0
        
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width * output_depth
        
        list_conv.append(flops)
    
    list_linear = []
    
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()
        
        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
    
    list_bn = []
    
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())
    
    list_relu = []
    
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())
    
    list_pooling = []
    
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width, input_depth = input[0].size()
        output_channels, output_height, output_width, output_depth = output[0].size()
        
        kernel_ops = self.kernel_size * self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width * output_depth
        
        list_pooling.append(flops)
    
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv3d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.ConvTranspose3d):
                # net.register_forward_hook(save_hook(net.__class__.__name__))
                # net.register_forward_hook(simple_hook)
                # net.register_forward_hook(simple_hook2)
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm3d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool3d) or isinstance(net, torch.nn.AvgPool3d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)
    
    foo(model)
    input = Variable(torch.rand(3, 1, 64, 64, 64), requires_grad=True)
    out = model(input.cuda())
    
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    
    logger.info('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))