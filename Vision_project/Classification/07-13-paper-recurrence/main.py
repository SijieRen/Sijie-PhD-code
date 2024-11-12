from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models import *
from resnet import Resnet18
from utils import *
import time
import pickle
import copy
import datetime
import xlrd
import torchvision



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
                 is_endwith6=0,
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
        
        self.feature_mask_1 = np.zeros(74, ).astype('bool')
        self.feature_mask_2 = np.zeros(74, ).astype('bool')
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
        if filename == 'std-123456':
            workbook1 = xlrd.open_workbook(
            r"../ppa-classi-dataset-onuse/ppa_08-16-order1-std-2345to6.xls")
        if filename == 'std-ori':
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_06-29-order1-std-8.xls")
        elif filename == 'std_all':
            print('load minmax model')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2021-12-06-order1-std-8-withAllLabel.xls")
        
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
                            self.target_ppa.append(sheet1.row_values(rows)[73])
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
                           # transforms.RandomHorizontalFlip(p=0.5),
                           # transforms.RandomVerticalFlip(p=0.5),
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


def save_results_order1(args,
                 model_C,
                 model_R,
                 G_net1,# G_net2,
                 D_net,
                 train_results,
                 val_results_list,
                 test_results_list,
                 full_results,
                 optimizer_C,
                 optimizer_R,
                 optimizer_G1,# optimizer_G2,
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
        #torch.save(G_net2.state_dict(), os.path.join(args.save_dir, 'best_test_acc' + '_G_net2.pt'))
    if epoch == args.best_test_auc_epoch:
        torch.save(model_R.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_R.pt'))
        torch.save(model_C.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_D_net.pt'))
        torch.save(G_net1.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_G_net1.pt'))
        #torch.save(G_net2.state_dict(), os.path.join(args.save_dir, 'best_test_auc' + '_G_net2.pt'))
    if epoch == args.best_val_acc_epoch:
        torch.save(model_R.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_R.pt'))
        torch.save(model_C.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_D_net.pt'))
        torch.save(G_net1.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_G_net1.pt'))
        #torch.save(G_net2.state_dict(), os.path.join(args.save_dir, 'best_val_acc' + '_G_net2.pt'))
    if epoch == args.best_val_auc_epoch:
        torch.save(model_R.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_R.pt'))
        torch.save(model_C.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_model_2_generate.pt'))
        torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_D_net.pt'))
        torch.save(G_net1.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_G_net1.pt'))
        #torch.save(G_net2.state_dict(), os.path.join(args.save_dir, 'best_val_auc' + '_G_net2.pt'))
    
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
        test_acc_mean = test_acc_mean + full_results[args.best_val_acc_epoch]['test_results_list'][ss][
            'acc_average_all']
        args.logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_acc_epoch]['test_results_list'][ss]['acc_average_all']))
    args.logger.info('mean test acc: %0.4f' % (test_acc_mean / 5))
    
    test_auc_mean = 0.0
    args.logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (args.best_val_auc_epoch, args.best_val_auc))
    for ss in range(len(test_results_list)):
        test_auc_mean = test_auc_mean + full_results[args.best_val_auc_epoch]['test_results_list'][ss][
            'AUC_average_all']
        args.logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[args.best_val_auc_epoch]['test_results_list'][ss]['AUC_average_all']))
    args.logger.info('mean test auc: %0.4f' % (test_auc_mean / 5))
    
    if epoch == args.epochs:
        save_results_as_xlsx(root='', args=args)
        pass
    
    is_best = 1
    if args.save_checkpoint > 0:
        save_checkpoint({
            'epoch': copy.deepcopy(epoch),
            'model_2_generate': model_C.state_dict(),
            'model_R': model_R.state_dict(),
            'G_net1': G_net1.state_dict(),
            #'G_net2': G_net2.state_dict(),
            'D_net': D_net.state_dict(),
            'best_test_acc': args.best_test_acc,
            'optimizer_C': optimizer_C.state_dict(),
            'optimizer_R': optimizer_R.state_dict(),
            'optimizer_G1': optimizer_G1.state_dict(),
            #'optimizer_G2': optimizer_G2.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
        }, is_best, base_dir=args.save_dir)
        # torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'Final_model_1.pt'))
        # torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'Final_model_2_generate.pt'))
        # torch.save(model_2_res.state_dict(), os.path.join(args.save_dir, 'Final_model_2_res.pt'))
        # torch.save(G_net.state_dict(), os.path.join(args.save_dir, 'Final_G_net.pt'))
        # torch.save(D_net.state_dict(), os.path.join(args.save_dir, 'Final_D_net.pt'))


def train(args,
          model_C,
          G_net1,
          #G_net2,
          D_net,
          model_R,
          train_loader,
          optimizer_C,
          optimizer_G1,# optimizer_G2,
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
            
            #for p in G_net2.parameters():
                #p.requires_grad = False
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
            z1 = z1.view(z1.shape[0], 100, 1, 1)
            feature_1 = feature_1.view(feature_1.shape[0], feature_1.shape[1], 1, 1)
            z1 = torch.cat((z1, feature_1), 1)
            y1 = torch.randn(data_1.size(0), 100).cuda()
            # y1 = torch.sign(y1)
            generate_feature_1, generate_feature_2, y_hat1, y_hat2 = G_net1(z1, y1)
            
            y_hat1 = torch.sign(y_hat1).long()
            y_hat2 = torch.sign(y_hat2).long()

            # reg_feature_real = model_R(data_1)
            # reg_feature_gene = model_R(generate_feature_1)

            # pred_t = model_C(data_1).argmax(dim=1, keepdim=True)
            pred_t1 = model_C(data_2)
            # pred_Gt = model_C(generate_feature_1)#.argmax(dim=1, keepdim=True)
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
            loss_C_data1_num = 0
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
                # if int(grad_2[i].detach().cpu().numpy()) == 6:
                loss_C_data1 += F.cross_entropy(model_C(data_R_real), target_ppa[i])
                loss_C_data1_num +=1
                if pred.detach().cpu().numpy() == target_ppa[i].squeeze().detach().cpu().numpy():
                    acc_num += 1
            if not loss_C_data1_num == 0:
                loss_C_data1 /= loss_C_data1_num
            loss_R_1 /= R_num
            loss_R_2 /= R_num

            lossC_1 = 0
            lossC_1_num = 0
            # lossC_num = 0
            for i in range(data_1.size(0)):
                if int(grad_2[i].detach().cpu().numpy()) == 6:
                    lossC_1 += F.cross_entropy(pred_t1[i].unsqueeze(0), target_ppa[i])
                    lossC_1_num += 1

            if not lossC_1_num == 0:
                lossC_1 /= lossC_1_num

            #lossC_scale = 1
            #if not (int(data_1.size(0)) - lossC_num) == 0:
                #lossC_scale = int(data_1.size(0)) / (int(data_1.size(0)) - lossC_num)
            
            loss_G = args.lossG * (fake_loss_1 + fake_loss_2)
            loss_R = loss_R_1 + args.lambda_R * loss_R_2
            loss_C = lossC_1 + args.lambda_C * F.cross_entropy(pred_Gt1, y_hat2.squeeze())
            #loss_C *= lossC_scale
            loss_C += args.lossC_data1 * loss_C_data1

            loss = args.lossC * loss_C + args.lossR * loss_R
            
            optimizer_C.zero_grad()
            optimizer_R.zero_grad()
            loss.backward(retain_graph=False)
            optimizer_C.step()
            optimizer_R.step()
            
            train_loss_G.update(loss_G.item(), 2 * data_1.size(0))
            train_loss_D.update(loss_D.item(), 2 * data_1.size(0))
            train_loss_C.update(loss_C.item(), data_1.size(0))
            train_loss_R.update(loss_R.item(), data_1.size(0))
        
        if batch_idx % 5 >= args.D_epoch:
            image_sequence += data_1.size(0)
            
            #for p in G_net2.parameters():
                #p.requires_grad = True
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
            z1 = z1.view(z1.shape[0], 100, 1, 1)
            feature_1 = feature_1.view(feature_1.shape[0], feature_1.shape[1], 1, 1)
            z1 = torch.cat((z1, feature_1), 1)
            y1 = torch.randn(data_1.size(0), 100).cuda()
            # y1 = torch.sign(y1)
            generate_feature_1, generate_feature_2, y_hat1, y_hat2 = G_net1(z1, y1)

            y_hat1 = torch.sign(y_hat1).long()
            y_hat2 = torch.sign(y_hat2).long()

            # reg_feature_real = model_R(data_1)
            # reg_feature_gene = model_R(generate_feature_1)

            # pred_t = model_C(data_1)  # .argmax(dim=1, keepdim=True)
            pred_t1 = model_C(data_2)

            # pred_Gt = model_C(generate_feature_1)  # .argmax(dim=1, keepdim=True)
            pred_Gt1 = model_C(generate_feature_2)
            
            real_loss_1 = D_net(data_1).mean(0).view(1)
            fake_loss_1 = D_net(generate_feature_1).mean(0).view(1)
            
            real_loss_2 = D_net(data_2).mean(0).view(1)
            fake_loss_2 = D_net(generate_feature_2).mean(0).view(1)

            R_num = 0
            loss_R_1 = 0
            loss_R_2 = 0
            loss_C_data1 = 0
            loss_C_data1_num = 0
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
                # if int(grad_2[i].detach().cpu().numpy()) == 6:
                loss_C_data1 += F.cross_entropy(model_C(data_R_real), target_ppa[i])
                loss_C_data1_num += 1
                if pred.detach().cpu().numpy() == target_ppa[i].squeeze().detach().cpu().numpy():
                    acc_num += 1
            if not loss_C_data1_num == 0:
                loss_C_data1 /= loss_C_data1_num
            loss_R_1 /= R_num
            loss_R_2 /= R_num

            lossC_1 = 0
            lossC_1_num = 0
            # lossC_num = 0
            for i in range(data_1.size(0)):
                if int(grad_2[i].detach().cpu().numpy()) == 6:
                    lossC_1 += F.cross_entropy(pred_t1[i].unsqueeze(0), target_ppa[i])
                    lossC_1_num +=1

            if not lossC_1_num == 0:
                lossC_1 /= lossC_1_num
            # lossC_scale = 1
            # if not (int(data_1.size(0)) - lossC_num) == 0:
            # lossC_scale = int(data_1.size(0)) / (int(data_1.size(0)) - lossC_num)

            loss_G = args.lossG * (fake_loss_1 + fake_loss_2)
            loss_R = loss_R_1 + args.lambda_R * loss_R_2
            loss_C = lossC_1 + args.lambda_C * F.cross_entropy(pred_Gt1, y_hat2.squeeze())
            #loss_C *= lossC_scale
            loss_C += args.lossC_data1 * loss_C_data1
            loss = args.lossC * loss_C + loss_G + args.lossR * loss_R
            #print('lossC_data1 ori :', loss_C_data1.detach().cpu().numpy())
            #print('lossC_data1 coeffi:', args.lossC_data1)
            #print('lossC_data1 :', (args.lossC_data1 * loss_C_data1).detach().cpu().numpy())
            #print('lossC_T', lossC_1.detach().cpu().numpy())
            #print('lambda_C:', (args.lambda_C * F.cross_entropy(pred_Gt1, y_hat2.squeeze()).detach().cpu().numpy()))
            # print(loss)
            optimizer_G1.zero_grad()
            #optimizer_G2.zero_grad()
            optimizer_C.zero_grad()
            optimizer_R.zero_grad()
            loss.backward(retain_graph=False)
            optimizer_R.step()
            optimizer_C.step()
            #optimizer_G2.step()
            optimizer_G1.step()
            
            train_loss_D.update(loss_D.item(), 2 * data_1.size(0))
            train_loss_G.update(loss_G.item(), 2 * data_1.size(0))
            train_loss_C.update(loss_C.item(), data_1.size(0))
            train_loss_R.update(loss_R.item(), data_1.size(0))
        
        # print('acc for train is :', acc_num / num)
        args.logger.info('Model Train Epoch: {} [{}/{} ({:.0f}%)] loss_G: {:.4f}, '
                         'loss_D: {:.4f}, loss_R: {:.4f}, loss_C: {:.4f}'.format(
            epoch, batch_idx * len(data_1), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), train_loss_G.avg,
            train_loss_D.avg, train_loss_R.avg, train_loss_C.avg))

        # args.logger.info('loss_D is real pred RA loss: {}'.format(train_loss_D.avg))
    
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
             #G_net1,
             test_loader,
             epoch,
             deltaT):
    model_C.eval()
    model_R.eval()
    #G_net1.eval()
    
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
                # print('{} :{}'.format(ii, reg))
                reg = model_R(reg)
            '''
            z1 = torch.randn(data_1.size(0), 100, 1, 1).cuda()
            y1 = torch.randn(data_1.size(0), 100).cuda()
            y1 = torch.sign(y1)
            generate_feature_1, y_hat = G_net1(z1, y1)
            y_hat1 = torch.sign(y_hat).long()
            '''
            
            P_residue_1_i = torch.softmax(model_C(reg), dim=1)
            # print('predict values',model_C(generate_feature_1))
            # print('predict softmax',torch.softmax(model_C(generate_feature_1), dim=1))
            # print('data',data_1)
            # print('generate',generate_feature_1)
            
            pred_minus = P_residue_1_i.argmax(dim=1, keepdim=True)
            
            correct_minus += pred_minus.eq(target_ppa.view_as(pred_minus)).sum().item()

            # pred_result_minus[batch_begin:batch_begin + data_1.size(0), :] = F.softmax(P_residue_1_i,
            # dim=1).detach().cpu().numpy()
            pred_result_minus[batch_begin:batch_begin + data_1.size(0), :] = P_residue_1_i.detach().cpu().numpy()
            
            pred_label_minus[batch_begin:batch_begin + data_1.size(0)] = pred_minus.detach().cpu().numpy()
            target[batch_begin:batch_begin + data_1.size(0)] = target_ppa.detach().cpu().numpy()
            
            for i in range(data_1.size(0)):
                name.append(test_loader.dataset.image_path_all_1[batch_begin + i])
            
            batch_begin = batch_begin + data_1.size(0)
    
    # print('acc for test using correct_num is :', correct_minus)
    AUC_minus = sklearn.metrics.roc_auc_score(target, pred_result_minus[:, 1])
    acc_minus = sklearn.metrics.accuracy_score(target, np.argmax(pred_result_minus, axis=1))
    cm_minus = sklearn.metrics.confusion_matrix(target, np.argmax(pred_result_minus, axis=1))
    sensitivity_minus = cm_minus[0, 0] / (cm_minus[0, 0] + cm_minus[0, 1])
    specificity_minus = cm_minus[1, 1] / (cm_minus[1, 0] + cm_minus[1, 1])
    
    # args.logger.info('In epoch {} for generate, AUC is {}, acc is {}.'.format(epoch, AUC_gen, acc_gen))
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


def main():
    # Training settings
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader_list, test_loader_list = get_all_dataloader(args)
    
    #resnet18 = models.resnet18(pretrained=False)
    resnet18 = Resnet18(dropout=args.dropout, dp=args.dp,pretrained=False)
    
    model_C = resnet18.cuda()
    
    if args.modelR == 'half':
        model_R = UNet_half(n_channels = 3,
                         n_classes = 3,
                         bilinear = args.bi_linear,
                         feature_num=0,
                         final_tanh=args.final_tanh,
                         is_ESPCN=args.is_ESPCN,scale_factor=args.scale_factor, mid_channel=args.dw_midch,
                            dw_type=args.dw_type,
                            inch=args.Rinch, outch=args.Routch).cuda()
    elif args.modelR == 'half_minus1':
        model_R = UNet_half_minus1(n_channels=3,
                            n_classes=3,
                            bilinear=args.bi_linear,
                            feature_num=0,
                            final_tanh=args.final_tanh,
                            is_ESPCN=args.is_ESPCN, scale_factor=args.scale_factor, mid_channel=args.dw_midch,
                            dw_type=args.dw_type,
                            inch=args.Rinch, outch=args.Routch).cuda()
    elif args.modelR == 'half_minus2':
        model_R = UNet_half_minus2(n_channels=3,
                            n_classes=3,
                            bilinear=args.bi_linear,
                            feature_num=0,
                            final_tanh=args.final_tanh,
                            is_ESPCN=args.is_ESPCN, scale_factor=args.scale_factor, mid_channel=args.dw_midch,
                            dw_type=args.dw_type,
                            ch=args.Rinch).cuda()
    
    
   
    if args.G_net_type == 'G_net':
        G_net1 = Generator_paper1order1().cuda()
        #G_net2 = Generator_paper2().cuda()
    elif args.G_net_type == 'halfG_net':
        G_net1 = Generator_paper1order1_half().cuda()
    elif args.G_net_type == 'U_net':
        G_net = UNet(n_channels = 128,
                     n_classes = 128,
                     bilinear = args.bi_linear,
                     feature_num=len(args.feature_list)+1,
                     final_tanh=args.final_tanh,
                     is_ESPCN=args.is_ESPCN,scale_factor=args.scale_factor, mid_channel=args.dw_midch, dw_type=args.dw_type).cuda()
        
    elif 'Big' in args.gantype:
        G_net = BigGenerator(gantype=args.gantype, n_class=len(args.feature_list)).cuda()
    elif 'Half' in args.gantype:
        G_net = HalfBigGenerator(gantype=args.gantype, n_class=len(args.feature_list)).cuda()
    elif 'localadjust' in args.gantype:
        G_net = BigGenerator1(gantype=args.gantype, n_class=len(args.feature_list)).cuda()
    
    if 'Big' in args.discritype:
        D_net = BigDiscriminator1(n_class=len(args.feature_list), SCR=args.SCR).cuda()
    elif 'small' in args.discritype:
        D_net = Discriminator_paper(3).cuda()
    elif 'normal' in args.discritype:
        D_net = Discriminator(3).cuda()


    print('-' * 20)
    print('-' * 20)
    print('print the number of model param :')
    print_model_parm_nums(model_C, logger=args.logger)
    print_model_parm_nums(model_R, logger=args.logger)
    #print_model_parm_nums(model_2_generate, logger=args.logger)
    print_model_parm_nums(G_net1, logger=args.logger)
    print_model_parm_nums(D_net, logger=args.logger)
    print('-' * 20)
    print('-' * 20)
    
    if args.optimizer == 'Mixed':
        optimizer_C = optim.SGD([{'params': model_C.parameters(), 'lr': args.lr2,
                                        'weight_decay': args.wd2, 'momentum': args.momentum}])
        optimizer_R = optim.SGD([{'params': model_R.parameters(), 'lr': args.lr3,
                                        'weight_decay': args.wd3, 'momentum': args.momentum}])
        #optimizer_C = optim.Adam(model_C.parameters(), lr=args.lr2)
        optimizer_G1 = optim.RMSprop([{'params': G_net1.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
        #optimizer_G2 = optim.RMSprop([{'params': G_net2.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
        optimizer_D = optim.RMSprop(D_net.parameters(), lr=args.lr, weight_decay=args.wd)

    
    if args.optimizer == 'Mixed-Adam':
        optimizer_C = optim.Adam([{'params': model_C.parameters(), 'lr': args.lr2,
                                        'weight_decay': args.wd, 'betas': (args.betas1, args.betas2), 'eps': 0.00000001}])
        optimizer_R = optim.Adam([{'params': model_C.parameters(), 'lr': args.lr3,
                                        'weight_decay': args.wd, 'betas': (args.betas1, args.betas2), 'eps': 0.00000001}])
        optimizer_G1 = optim.RMSprop([{'params': G_net1.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
        optimizer_G2 = optim.RMSprop([{'params': G_net2.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
        optimizer_D = optim.RMSprop(D_net.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.optimizer == 'Mixed-SA':
        optimizer_C = optim.SGD([{'params': model_C.parameters(), 'lr': args.lr2,
                                        'weight_decay': args.wd, 'momentum': args.momentum}])
        optimizer_R = optim.Adam([{'params': model_C.parameters(), 'lr': args.lr3,
                                        'weight_decay': args.wd, 'betas': (args.betas1, args.betas2), 'eps': 0.00000001}])
        optimizer_G1 = optim.RMSprop([{'params': G_net1.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
        optimizer_G2 = optim.RMSprop([{'params': G_net2.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
        optimizer_D = optim.RMSprop(D_net.parameters(), lr=args.lr, weight_decay=args.wd)
    
    full_results = {}
    args = init_metric(args)

    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_results = train(args,
                                  model_C,
                                  G_net1,
                                  D_net,
                                  model_R,
                                  train_loader,
                                  optimizer_C,
                                  optimizer_G1,
                                  optimizer_D,
                                  optimizer_R,
                                  epoch)
            test_results_list = []
            val_results_list = []
            
            for ss in range(len(val_loader_list)):
                test_results_list.append(
                    evaluate(args,
                             model_R,
                             model_C,
                             #G_net1,
                             test_loader_list[ss],
                             epoch,
                             5-ss))
                val_results_list.append(
                    evaluate(args,
                             model_R,
                             model_C,
                             #G_net1,
                             val_loader_list[ss],
                             epoch,
                             5-ss))
    
            adjust_learning_rate(optimizer_C, epoch, args)
            adjust_learning_rate(optimizer_R, epoch, args)
    
            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results_order1(args,
                         model_C,
                         model_R,
                         G_net1,
                         #G_net2,
                         D_net,
                         train_results,
                         val_results_list,
                         test_results_list,
                         full_results,
                         optimizer_C,
                         optimizer_R,
                         optimizer_G1,# optimizer_G2,
                         optimizer_D,
                         epoch)
    finally:
        args.logger.info('save_results_path: %s'%args.save_dir)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
if __name__ == '__main__':
    main()
