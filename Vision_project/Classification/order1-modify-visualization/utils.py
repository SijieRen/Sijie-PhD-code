# -*- coding: UTF-8 -*-
import torch
import numpy as np
import os
import xlrd
from PIL import Image
import copy
import os
import pickle
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
import sklearn.metrics
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
import datetime
import torch
from torchvision import models
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils
import torch.autograd as autograd
import argparse



def get_opts():
    parser = argparse.ArgumentParser(description='PyTorch PrimarySchool Fundus photography RA-Regression')
    # Params
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default= -1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=24, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='Mixed',
                        help='choose the optimizer function')
    
    # Dataset
    parser.add_argument('--data_root', type=str, default='../data/JPG256SIZE-95')
    parser.add_argument('--filename', type=str, default='std')
    parser.add_argument('--feature_list', type=int, nargs='+', default=[8, 9], help='')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--eye', type=str, default='R',
                        help='choose the right or left eye')
    parser.add_argument('--center', type=str, default='Maculae',
                        help='choose the center type pf the picture,input(Maculae)(Disc)(Maculae&Disc) ')
    
    # Model
    parser.add_argument('--load_dir', type=str, default='/home')
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--works', type=int, default=2)
    
    # Visulization
    parser.add_argument('--is_plot', type=int, default=0)
    parser.add_argument('--plot_sigmoid', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--is_local_normal', type=int, default=0)
    parser.add_argument('--what_plot', type=int, default=0)
    parser.add_argument('--log_factor', type=float, default=1)
    parser.add_argument('--color', type=str, default='JET')

    parser.add_argument('--load_visual', type=str, default='/home')
    
    
    #parser.add_argument('--margin', type=float, default=0.5,
                        #help='set the error margin')
    
    parser.add_argument('--label', type=str, default='RA',
                        help='choose the label we want to do research')
    
    # Train set
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--lr2', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--lr3', type=float, default=0.001,
                        help='lr3 for ESPCN')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='')
    parser.add_argument('--lr_controler', type=int, default=30, )
    parser.add_argument('--wd', type=float, default=0.0001, help='')
    parser.add_argument('--wd2', type=float, default=0.0001, help='')
    parser.add_argument('--dp', type=float, default=0.0)
    parser.add_argument('--is_ESPCN', type=int, default=0,
                        help='whether load model ESPCN')
    parser.add_argument('--dw_midch', type=int, default=1024,
                        help='choose the mid-channel type of ESPCN')
    parser.add_argument('--RNN_hidden', type=int, nargs='+', default=[256], help='')
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--dp_n', type=float, default=0.5)
    parser.add_argument('--wcl', type=float, default=0.01,
                        help='control the clamp parameter of Dnet, weight_cliping_limit')
    parser.add_argument('--isplus', type=int, default=0)
    parser.add_argument('--final_tanh', type=int, default=0)
    parser.add_argument('--train_ori_data', type=int, default=0)
    parser.add_argument('--pred_future', type=int, default=1)
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
    parser.add_argument('--D_epoch', type=int, default=4)

    #parser.add_argument('--is_endwith6', type=int, default=0,
                        #help='if make the dataset end with 6')
    # parser.add_argument('--dataset_type', type=str, default='baseline')
    # parser.add_argument('--load_model_1', type=str, default='nbase_2')
    # parser.add_argument('--load_model_pre', type=str, default='baseline')
    parser.add_argument('--G_net_type', type=str, default='G_net')
    # parser.add_argument('--dw_type', type=str, default='conv',
    # help='choose the downsample type of ESPCN')
    
    #parser.add_argument('--critic_iter', type=int, default=5,
                        #help='control the critic iterations for updating Gnet')

    parser.add_argument('--noise_size', type=int, default=516,
                        help='control the size of the input of Gnet')
    parser.add_argument('--save_checkpoint', type=int, default=1,
                        help='save the check point of the model and optim')
    
    #parser.add_argument('--betas1', type=float, default='0.5',
                        #help='hyper-parameter for Adam')
    #parser.add_argument('--betas2', type=float, default='0.9',
                        #help='hyper-parameter for Adam')
    #parser.add_argument('--LAMBDA', type=float, default='10',
                        #help='hyper-parameter for gradient panelty')
    #parser.add_argument('--bi_linear', type=int, default=0,
                        #help='whether use the bilinear mode in Unet')
    # sijie add for debugging biggan
    #parser.add_argument('--gantype', type=str, default='Big',
                        #help='hyper-parameter for gradient panelty')
    #parser.add_argument('--discritype', type=str, default='normal',
                        #help='hyper-parameter for gradient panelty')
    #parser.add_argument('--SCR', type=int, default=0,
                        #help='hyper-parameter for gradient panelty')
    # sijie for generate test
    
    
    #parser.add_argument('--scale_factor', type=int, default=2,
                        #help='choose the mid-channel type of ESPCN')
    
    
    args = parser.parse_args()
    args.final_tanh = 0
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
    model.load_state_dict(new_state_dict,  strict=False)###parameter strict: do not load the part dont match
    return model

def load_pytorch_model22(model, path, list=0):
    from collections import OrderedDict
    state_dict = torch.load(path)
    print('load model successfully!')
    #print(state_dict)
    #print(state_dict[0])
    #print(len(state_dict))
    new_state_dict = OrderedDict()
    for k, v in state_dict[list].items():
        if k[:7] == 'module.':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    # load params
    model.load_state_dict(new_state_dict)
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

def draw_features(width, height, feature, savepath, image_sequence, train_loader, a, batchsize, dpi, localnormal, args):
    str = a
    batch_size = batchsize
    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)


    num = image_sequence
    x = feature
    np_savepath = np_savename = os.path.join(savepath, 'npy/')
    if not os.path.exists(np_savepath):
        os.makedirs(np_savepath)
    np_savename = os.path.join(savepath, 'npy/',os.path.basename(
        train_loader.dataset.image_path_all_1[num])[:11] + '%s' % (str) + os.path.basename(
        train_loader.dataset.image_path_all_1[num])[10:-4] + '.npy')
    np.save(np_savename, x)
    if localnormal == 0:
        pmax = 0.0001
        pmin = -0.0001
        for i in range(width * height):
            img = x[0, i, :, :]
            pmin1 = np.min(img)
            pmax1 = np.max(img)
            if pmin1 <= pmin:
                pmin = copy.deepcopy(pmin1)
            if pmax1 >= pmax:
                pmax = copy.deepcopy(pmax1)
        for i in range(width * height):
            plt.subplot(height, width, i + 1)
            plt.axis('off')
            img = x[0, i, :, :]
            img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
            img = img.astype(np.uint8)
            img = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
            img = img[:, :, ::-1]
            plt.imshow(img)
        
    else:
        for i in range(width * height):
            plt.subplot(height, width, i + 1)
            plt.axis('off')
            img = x[0, i, :, :]
            #mu = np.mean(img)
            #std = np.std(img)
            #img = ( (img - mu) / std ) * 255
            pmin = np.min(img)
            pmax = np.max(img)
            #img = ((img - pmin) / (pmax - pmin + 0.000001)) * 10  ## a echencement factor
            img = ((img - pmin) / (pmax - pmin + 0.000001)) * args.log_factor
            img = np.exp(img)
            pmin = np.min(img)
            pmax = np.max(img)
            img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
            img = img.astype(np.uint8)
            if args.color == 'JET':
                img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            if args.color == 'HSV':
                img = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
            img = img[:, :, ::-1]
            plt.imshow(img)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savename = os.path.join(savepath,  os.path.basename(
                        train_loader.dataset.image_path_all_1[num])[:11] + '%s' % (str) + os.path.basename(
                        train_loader.dataset.image_path_all_1[num])[10:])
    fig.savefig(savename, dpi=dpi)
    fig.clf()
    plt.close()


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







def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', base_dir='./'):
    torch.save(state, os.path.join(base_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(base_dir, filename), os.path.join(base_dir, 'model_best.pth.tar'))



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_decay ** (epoch // args.lr_controler))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
