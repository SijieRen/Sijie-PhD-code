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
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader


class Clas_ppa_train(data.Dataset):
    def __init__(self,
                 filename,
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
        self.target_RA_A = []
        self.target_RA_P = []
        self.feature_list = feature_list
        self.feature_all_1 = []
        self.feature_all_2 = []
        self.base_grade_num = []
        self.base_grade_num_2 = []

        print('load std data')
        if filename == 'std':
            workbook1 = xlrd.open_workbook(
                "../dataset/2020-12-24-new-dataset-std.xls")
        else:
            workbook1 = xlrd.open_workbook(
                "../dataset/2020-12-24-new-dataset.xls")
        sheet1 = workbook1.sheet_by_index(0)

        self.feature_mask_1 = np.zeros(sheet1.ncols, ).astype('bool')
        for ids in feature_list:
            self.feature_mask_1[ids] = True

        for rows in range(1, sheet1.nrows):
            if sheet1.row_values(rows)[4] == fold:
                if sheet1.row_values(rows)[2] in self.eye:
                    if str(sheet1.row_values(rows)[3]) in self.center:
                        self.image_path_all_1.append(os.path.join(self.root, sheet1.row_values(rows)[1]))

                        self.target_RA_A.append(sheet1.row_values(rows)[13])
                        self.feature_all_1.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))
                        self.target_RA_P.append(sheet1.row_values(rows)[46])

                        #self.base_grade_num.append(int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                        #self.base_grade_num_2.append(int(sheet1.row_values(rows)[2].split('/')[0][-1]))

    def __getitem__(self, index):

        img_path_1 = self.image_path_all_1[index]
        target_RA_A , target_RA_P= self.target_RA_A[index], self.target_RA_P[index]

        img_1 = Image.open(img_path_1)

        #base_target = [-1, -1]
        #if (target_ppa == 0 or target_ppa == 1) and self.base_grade_num[index] == 1:
            #base_target[0] = 0

        #if (target_ppa == 0 or target_ppa == 1) and self.base_grade_num_2[index] == 6:
            #base_target[1] = target_ppa

        if self.transform is not None:
            img_1 = self.transform(img_1)
            #img_2 = self.transform(img_2)

        return img_1, \
               torch.from_numpy(np.array(target_RA_A).astype('float32')), \
               torch.from_numpy(np.array(target_RA_P).astype('float32')), \
               torch.from_numpy(np.array(self.feature_all_1[index]).astype('float32')), \


    def __len__(self):
        return len(self.image_path_all_1)


def get_all_dataloader(args):
    test_loader_list = []
    val_loader_list = []
    kwargs = {'num_workers': args.works, 'pin_memory': True}
    train_loader = DataLoaderX(
        Clas_ppa_train(args.filename, args.data_root, fold='train', eye=args.eye,
                       center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.RandomRotation(30),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader_list = DataLoaderX(
        Clas_ppa_train(args.filename, args.data_root, fold='test', eye=args.eye,
                       center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    val_loader_list = DataLoaderX(
        Clas_ppa_train(args.filename, args.data_root, fold='val', eye=args.eye,
                       center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)


    return train_loader, val_loader_list, test_loader_list

def acc_calcu(a, b, margin):
    acc = 0
    for ii in range(len(a)):
        if abs(a[ii] - b[ii]) <= margin:
            acc += 1
    return acc


def train_baseline(phase,
                   args,
                   model_1,
                   model_2_generate,
                   train_loader,
                   optimizer_M_1,
                   optimizer_M_2_generate,
                   epoch):
    if phase == 'train':
        model_1.train()
        model_2_generate.train()
    else:
        model_1.eval()
        model_2_generate.eval()

    train_loss = AverageMeter()

    image_sequence = 0
    correct_num_ave = 0
    correct_num_minus = 0
    correct_num_pred = 0

    correct_num_ave_2 = 0
    correct_num_minus_2 = 0
    correct_num_pred_2 = 0

    correct_num_ave_3 = 0
    correct_num_minus_3 = 0
    correct_num_pred_3 = 0
    for batch_idx, (data_1, target_RA, before_RA, feature_1,) in enumerate(train_loader):
        data_1, target_RA, before_RA, feature_1 = data_1.cuda(), target_RA.cuda(), before_RA.cuda(),feature_1.cuda(),

        image_sequence += data_1.size(0)

        featuremap_1 = model_1(data_1)

        if args.model == 'MM_F':
            output_generate_1, output_generate_2 = model_2_generate(featuremap_1, feature_1)

        else:
            output_generate_1, output_generate_2 = model_2_generate(featuremap_1)


        if args.filename == 'std':
            target = target_RA
            pred = output_generate_1
            loss = F.mse_loss(pred, target)
            target = (target_RA * 1.6863335 + 0.013257576)
            # target = torch.tensor(target, dtype=torch.float32).cuda()
            pred = (output_generate_1 * 1.6863335 + 0.013257576)
            acc = acc_calcu(target, pred, args.margin)
            correct_num_1 += acc
        else:
            target = target_RA.reshape(-1, 1)
            before_RA = before_RA.reshape(-1, 1)
            pred_RA = output_generate_1
            pred_minus = output_generate_2
            loss = F.smooth_l1_loss(pred_RA, target) + F.smooth_l1_loss(pred_minus, (target - before_RA))

            acc_1 = acc_calcu(pred_RA, target, args.margin)
            acc_2 = acc_calcu(pred_minus + before_RA, target, args.margin)
            acc = (acc_1 + acc_2) / 2
            correct_num_ave += acc
            correct_num_pred += acc_1
            correct_num_minus += acc_2

            acc_1 = acc_calcu(pred_RA, target, 2*args.margin)
            acc_2 = acc_calcu(pred_minus + before_RA, target, 2*args.margin)
            acc = (acc_1 + acc_2) / 2
            correct_num_ave_2 += acc
            correct_num_pred_2 += acc_1
            correct_num_minus_2 += acc_2

            acc_1 = acc_calcu(pred_RA, target, 4*args.margin)
            acc_2 = acc_calcu(pred_minus + before_RA, target, 4*args.margin)
            acc = (acc_1 + acc_2) / 2
            correct_num_ave_3 += acc
            correct_num_pred_3 += acc_1
            correct_num_minus_3 += acc_2


            

        if phase == 'train':
            optimizer_M_1.zero_grad()
            optimizer_M_2_generate.zero_grad()
            loss.backward(retain_graph=False)
            optimizer_M_1.step()
            optimizer_M_2_generate.step()
        else:
            pass

        train_loss.update(loss.item(), data_1.size(0))  ####add loss_MG
        args.logger.info('Model {} Epoch: {} [{}/{} ({:.0f}%)] Loss_M1: {:.6f}'.format(phase,
            epoch, batch_idx * len(data_1), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), train_loss.avg))
    args.logger.info('In epoch {}, acc for margin {} is : {}'.format(epoch,
                                                                     args.margin,
                                                                    correct_num_ave / len(train_loader.dataset)))

    loss = {
        'loss': train_loss.avg,
        'acc_ave': correct_num_ave / len(train_loader.dataset),
        'acc_pred': correct_num_pred / len(train_loader.dataset),
        'acc_minus': correct_num_minus / len(train_loader.dataset),

        'acc_ave_2': correct_num_ave_2 / len(train_loader.dataset),
        'acc_pred_2': correct_num_pred_2 / len(train_loader.dataset),
        'acc_minus_2': correct_num_minus_2 / len(train_loader.dataset),

        'acc_ave_3': correct_num_ave_3 / len(train_loader.dataset),
        'acc_pred_3': correct_num_pred_3 / len(train_loader.dataset),
        'acc_minus_3': correct_num_minus_3 / len(train_loader.dataset),


    }
    return loss


def test_baseline(phase,
                   args,
                   model_1,
                   model_2_generate,
                   train_loader,
                   optimizer_M_1,
                   optimizer_M_2_generate,
                   epoch):
    if phase == 'train':
        model_1.train()
        model_2_generate.train()
    else:
        model_1.eval()
        model_2_generate.eval()
    
    train_loss = AverageMeter()
    
    image_sequence = 0
    correct_num_ave = 0
    correct_num_minus = 0
    correct_num_pred = 0

    correct_num_ave_2 = 0
    correct_num_minus_2 = 0
    correct_num_pred_2 = 0

    correct_num_ave_3 = 0
    correct_num_minus_3 = 0
    correct_num_pred_3 = 0
    with torch.no_grad():
        for batch_idx, (data_1, target_RA, before_RA, feature_1,) in enumerate(train_loader):
            data_1, target_RA, before_RA, feature_1 = data_1.cuda(), target_RA.cuda(), before_RA.cuda(), feature_1.cuda(),
            
            image_sequence += data_1.size(0)
            
            featuremap_1 = model_1(data_1)
            
            if args.model == 'MM_F':
                output_generate_1, output_generate_2 = model_2_generate(featuremap_1, feature_1)
            
            else:
                output_generate_1, output_generate_2 = model_2_generate(featuremap_1)
            
            if args.filename == 'std':
                target = target_RA
                pred = output_generate_1
                loss = F.mse_loss(pred, target)
                target = (target_RA * 1.6863335 + 0.013257576)
                # target = torch.tensor(target, dtype=torch.float32).cuda()
                pred = (output_generate_1 * 1.6863335 + 0.013257576)
                acc = acc_calcu(target, pred, args.margin)
                correct_num_1 += acc
            else:
                target = target_RA.reshape(-1, 1)
                before_RA = before_RA.reshape(-1, 1)
                pred_RA = output_generate_1
                pred_minus = output_generate_2
                #print('target', target)
                #print('pred_minus: ', pred_minus)
                #print('before RA: ', before_RA)
                #print('pred minus + before RA:', pred_minus + before_RA)
                loss = F.smooth_l1_loss(pred_RA, target) + F.smooth_l1_loss(pred_minus, (target - before_RA))

                acc_1 = acc_calcu(pred_RA, target, args.margin)
                acc_2 = acc_calcu(pred_minus + before_RA, target, args.margin)
                acc = (acc_1 + acc_2) / 2
                correct_num_ave += acc
                correct_num_pred += acc_1
                correct_num_minus += acc_2

                acc_1 = acc_calcu(pred_RA, target, 2 * args.margin)
                acc_2 = acc_calcu(pred_minus + before_RA, target, 2 * args.margin)
                acc = (acc_1 + acc_2) / 2
                correct_num_ave_2 += acc
                correct_num_pred_2 += acc_1
                correct_num_minus_2 += acc_2

                acc_1 = acc_calcu(pred_RA, target, 4 * args.margin)
                acc_2 = acc_calcu(pred_minus + before_RA, target, 4 * args.margin)
                acc = (acc_1 + acc_2) / 2
                correct_num_ave_3 += acc
                correct_num_pred_3 += acc_1
                correct_num_minus_3 += acc_2
            
            if phase == 'train':
                optimizer_M_1.zero_grad()
                optimizer_M_2_generate.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_M_1.step()
                optimizer_M_2_generate.step()
            else:
                pass
            
            train_loss.update(loss.item(), data_1.size(0))  ####add loss_MG
            args.logger.info('Model {} Epoch: {} [{}/{} ({:.0f}%)] Loss_M1: {:.6f}'.format(phase,
                                                                                           epoch, batch_idx * len(data_1),
                                                                                           len(train_loader.dataset),
                                                                                           100. * batch_idx / len(
                                                                                               train_loader),
                                                                                           train_loss.avg))
        args.logger.info('In epoch {}, acc for margin {} is : {}'.format(epoch,
                                                                         args.margin,
                                                                         correct_num_ave / len(train_loader.dataset)))
    
    loss = {
        'loss': train_loss.avg,
        'acc_ave': correct_num_ave / len(train_loader.dataset),
        'acc_pred': correct_num_pred / len(train_loader.dataset),
        'acc_minus': correct_num_minus / len(train_loader.dataset),

        'acc_ave_2': correct_num_ave_2 / len(train_loader.dataset),
        'acc_pred_2': correct_num_pred_2 / len(train_loader.dataset),
        'acc_minus_2': correct_num_minus_2 / len(train_loader.dataset),

        'acc_ave_3': correct_num_ave_3 / len(train_loader.dataset),
        'acc_pred_3': correct_num_pred_3 / len(train_loader.dataset),
        'acc_minus_3': correct_num_minus_3 / len(train_loader.dataset),
    }
    return loss



def save_results_baseline(args,
                          model_1,
                          model_2_generate,
                          train_results,
                          val_results,
                          test_results,
                          full_results,
                          optimizer_M_1,
                          optimizer_M_2_generate,
                          epoch):

    val_acc_1 = val_results['acc_ave']
    test_acc_1 = test_results['acc_ave']

    if args.best_test_acc < test_acc_1:
        args.best_test_acc = copy.deepcopy(test_acc_1)
        args.best_test_acc_epoch = copy.deepcopy(epoch)


    if args.best_val_acc < val_acc_1:
        args.best_val_acc = copy.deepcopy(val_acc_1)
        args.best_val_acc_epoch = copy.deepcopy(epoch)


    if epoch == args.best_test_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_test_acc_1_' + '_model_1.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_test_acc_1_' + '_model_2_generate.pt'))
    if epoch == args.best_val_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_val_acc_1_' + '_model_1.pt'))
        torch.save(model_2_generate.state_dict(), os.path.join(args.save_dir, 'best_val_acc_1_' + '_model_2_generate.pt'))

    args.logger.info(
        'Utill now the best test acc epoch is : {},  ave acc is {}'.format(args.best_test_acc_epoch, args.best_test_acc))
    args.logger.info(
        'Utill now the best val acc epoch is : {},  ave acc is {}'.format(args.best_val_acc_epoch, args.best_val_acc))

    full_results[epoch] = {
        'train_results': copy.deepcopy(train_results),
        'test_results': copy.deepcopy(test_results),
        'val_results': copy.deepcopy(val_results),
    }
    pickle.dump(full_results, open(os.path.join(args.save_dir, 'results.pkl'), 'wb'))
    args.logger.info('for model average :')
    args.logger.info(
        'In this epoch : {}, test average acc is {}'.format(args.best_val_acc_epoch,
                                                full_results[args.best_val_acc_epoch]['test_results']['acc_ave']))
    args.logger.info(
        'In this epoch : {}, test average acc is {}'.format(args.best_val_acc_epoch,
                                                full_results[args.best_val_acc_epoch]['test_results']['acc_ave_2']))
    args.logger.info(
        'In this epoch : {}, test average acc is {}'.format(args.best_val_acc_epoch,
                                                full_results[args.best_val_acc_epoch]['test_results']['acc_ave_3']))

    args.logger.info('for model pred :')
    args.logger.info(
        'In this epoch : {}, test pred acc is {}'.format(args.best_val_acc_epoch,
                                                full_results[args.best_val_acc_epoch]['test_results']['acc_pred']))
    args.logger.info(
        'In this epoch : {}, test pred acc is {}'.format(args.best_val_acc_epoch,
                                                full_results[args.best_val_acc_epoch]['test_results']['acc_pred_2']))
    args.logger.info(
        'In this epoch : {}, test pred acc is {}'.format(args.best_val_acc_epoch,
                                                full_results[args.best_val_acc_epoch]['test_results']['acc_pred_3']))

    args.logger.info('for model pred minus :')
    args.logger.info(
        'In this epoch : {}, test pred minus acc is {}'.format(args.best_val_acc_epoch,
                                                full_results[args.best_val_acc_epoch]['test_results']['acc_minus']))
    args.logger.info(
        'In this epoch : {}, test pred minus acc is {}'.format(args.best_val_acc_epoch,
                                                full_results[args.best_val_acc_epoch]['test_results']['acc_minus_2']))
    args.logger.info(
        'In this epoch : {}, test pred minus acc is {}'.format(args.best_val_acc_epoch,
                                                full_results[args.best_val_acc_epoch]['test_results']['acc_minus_3']))

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


def main():
    # Training settings
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader, test_loader = get_all_dataloader(args)

    model_1 = RN18_front().cuda()
    if args.double_pred == 1:
        if args.model == 'MM_F':
            model_2_generate = RN18_last_attr_double(num_classes=args.class_num,
                                              feature_dim=len(args.feature_list), dropout=args.dropout,
                                              dp=args.dp).cuda()
        else:
            model_2_generate = RN18_last_double(num_classes=args.class_num, dropout=args.dropout, dp=args.dp).cuda()
    else:
        if args.model == 'MM_F':
            model_2_generate = RN18_last_attr(num_classes=args.class_num,
                                              feature_dim=len(args.feature_list), dropout=args.dropout,
                                              dp=args.dp).cuda()
        else:
            model_2_generate = RN18_last(num_classes=args.class_num, dropout=args.dropout, dp=args.dp).cuda()

    if args.optimizer == 'SGD':
        optimizer_M_1 = optim.SGD([{'params': model_1.parameters(), 'lr': args.lr,
                                    'weight_decay': args.wd, 'momentum': args.momentum}])
        optimizer_M_2_generate = optim.SGD(
            [{'params': model_2_generate.parameters(), 'lr': args.lr,
              'weight_decay': args.wd, 'momentum': args.momentum}])
    elif args.optimizer == 'Adam':
        optimizer_M_1 = optim.Adam(model_1.parameters(), lr=args.lr, weight_decay=args.wd)
        optimizer_M_2_generate = optim.Adam(model_2_generate.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'RMSprop':
        optimizer_M_1 = optim.RMSprop(model_1.parameters(), lr=args.lr, alpha=args.alpha)
        optimizer_M_2_generate = optim.RMSprop(model_2_generate.parameters(), lr=args.lr, alpha=args.alpha)

    full_results = {}
    args = init_metric(args)
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_results = train_baseline('train', args, model_1, model_2_generate, train_loader,
                                           optimizer_M_1, optimizer_M_2_generate, epoch)
            test_results = test_baseline('test', args, model_1, model_2_generate, test_loader,
                                           optimizer_M_1, optimizer_M_2_generate, epoch)
            val_results = test_baseline('val', args, model_1, model_2_generate, val_loader,
                                          optimizer_M_1, optimizer_M_2_generate, epoch)


            adjust_learning_rate(optimizer_M_1, epoch, args)
            adjust_learning_rate(optimizer_M_2_generate, epoch, args)

            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results_baseline(
                args,
                model_1,
                model_2_generate,
                train_results,
                val_results,
                test_results,
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
