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
from efficientnet_pytorch import EfficientNet
import torchvision.models as torchmodel


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
                        if filename == 'std':
                            self.target_RA_A.append(sheet1.row_values(rows)[13])###target RA_A-13 /// RA_P-16
                        else:
                            self.target_RA_A.append(sheet1.row_values(rows)[43])  ###target RA_A-43 /// RA_P-46
                        self.feature_all_1.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))

                        #self.base_grade_num.append(int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                        #self.base_grade_num_2.append(int(sheet1.row_values(rows)[2].split('/')[0][-1]))

    def __getitem__(self, index):

        img_path_1 = self.image_path_all_1[index]
        target_RA_A = self.target_RA_A[index]

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
                           transforms.RandomHorizontalFlip(),
                           # transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.4),
                           transforms.RandomRotation(30),
                           transforms.ToTensor(),
                           # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader_list = DataLoaderX(
        Clas_ppa_train(args.filename, args.data_root, fold='test', eye=args.eye,
                       center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    val_loader_list = DataLoaderX(
        Clas_ppa_train(args.filename, args.data_root, fold='val', eye=args.eye,
                       center=args.center, feature_list=args.feature_list,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)


    return train_loader, val_loader_list, test_loader_list

def acc_calcu(a, b, margin):
    acc = 0
    for ii in range(len(a)):
        if not abs(a[ii] - b[ii]) > margin:
            acc += 1
    return acc


def train_baseline(phase,
                   args,
                   model_1,
                   model_2,

                   train_loader,
                   optimizer_M_1,
                   optimizer_M_2,

                   epoch):
    phase = phase
    
    model_1.train()
    model_2.train()

    


    train_loss = AverageMeter()

    image_sequence = 0
    correct_num_1 = 0
    correct_num_2 = 0
    correct_num_3 = 0
    correct_num_4 = 0
    for batch_idx, (data_1, target_RA, feature_1,) in enumerate(train_loader):
        data_1, target_RA, feature_1 = data_1.cuda(), target_RA.cuda(), feature_1.cuda(),
        # print('train', target_RA)
        image_sequence += data_1.size(0)

        if args.modality == 'MMF':
            output_generate_1 = args.beta1 * model_1(data_1)
            output_generate_1 += model_2(feature_1)
        elif args.modality == 'MLP':
            output_generate_1 = model_2(feature_1)
        elif args.modality == 'img':
            output_generate_1 = model_1(data_1)

		
        if args.filename == 'std':
            target = target_RA.reshape(-1, 1)
            pred = output_generate_1.reshape(-1, 1)
            loss = F.smooth_l1_loss(pred, target)

            if args.modality == 'MMF':
                optimizer_M_1.zero_grad()
                optimizer_M_2.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_M_1.step()
                optimizer_M_2.step()
            elif args.modality == 'img':
                optimizer_M_1.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_M_1.step()
            elif args.modality == 'MLP':
                optimizer_M_2.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_M_2.step()
            
            target = (target_RA * 1.6863335 + 0.013257576)  ###RA_A
            #target = (target * 1.5460724 - 0.9562774)  ###RA_P
            pred = (output_generate_1 * 1.6863335 + 0.013257576)  ###RA_A
            #pred = (pred * 1.5460724 - 0.9562774)  ###RA_P
            acc = acc_calcu(target, pred, args.margin)
            correct_num_1 += acc
            acc = acc_calcu(pred, target, args.margin * 2)
            correct_num_2 += acc
            acc = acc_calcu(pred, target, args.margin * 4)
            correct_num_3 += acc
            acc = acc_calcu(pred, target, args.margin * 8)
            correct_num_4 += acc
        else:
            target = target_RA.reshape(-1, 1)
            pred = output_generate_1.reshape(-1, 1)
            loss = F.smooth_l1_loss(pred, target)

            if args.modality == 'MMF':
                optimizer_M_1.zero_grad()
                optimizer_M_2.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_M_1.step()
                optimizer_M_2.step()
            elif args.modality == 'img':
                optimizer_M_1.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_M_1.step()
            elif args.modality == 'MLP':
                optimizer_M_2.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_M_2.step()
                
            acc = acc_calcu(pred, target, args.margin)
            correct_num_1 += acc
            acc = acc_calcu(pred, target, args.margin * 2)
            correct_num_2 += acc
            acc = acc_calcu(pred, target, args.margin * 4)
            correct_num_3 += acc
            acc = acc_calcu(pred, target, args.margin * 8)
            correct_num_4 += acc

        train_loss.update(loss.item(), data_1.size(0))  ####add loss_MG
        args.logger.info('Model {} Epoch: {} [{}/{} ({:.0f}%)] Loss_M1: {:.6f}'.format(phase,
            epoch, batch_idx * len(data_1), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), train_loss.avg))
    args.logger.info('In epoch {}, acc for margin {} is : {}'.format(epoch,
                                                                     args.margin,
                                                                    correct_num_1 / len(train_loader.dataset)))

    loss = {
        'loss': train_loss.avg,
        'acc_1': correct_num_1 / len(train_loader.dataset),
        'acc_2': correct_num_2 / len(train_loader.dataset),
        'acc_3': correct_num_3 / len(train_loader.dataset),
        'acc_4': correct_num_4 / len(train_loader.dataset),
    }
    return loss


def test_baseline(phase,
                   args,
                   model_1,
                    model_2,
                   train_loader,
                   optimizer_M_1,
                  optimizer_M_2,
                   epoch):
    model_1.eval()
    model_2.eval()
    train_loss = AverageMeter()
    
    image_sequence = 0
    correct_num_1 = 0
    correct_num_2 = 0
    correct_num_3 = 0
    correct_num_4 = 0
    with torch.no_grad():
        for batch_idx, (data_1, target_RA, feature_1,) in enumerate(train_loader):
            data_1, target_RA, feature_1 = data_1.cuda(), target_RA.cuda(), feature_1.cuda(),
            # print('test', target_RA)
            image_sequence += data_1.size(0)
    
            output_generate_1 = 0
            #output_2 = model_2(feature_1)
            if args.modality == 'MMF':
                output_generate_1 = args.beta1 * model_1(data_1)
                output_generate_1 += model_2(feature_1)
            elif args.modality == 'MLP':
                output_generate_1 = model_2(feature_1)
            elif args.modality == 'img':
                output_generate_1 = model_1(data_1)
            
            if args.filename == 'std':
                target = target_RA.reshape(-1, 1)
                pred = output_generate_1.reshape(-1, 1)
                loss = F.smooth_l1_loss(pred, target)
                target = (target_RA * 1.6863335 + 0.013257576)###RA_A
                #target = (target * 1.5460724 - 0.9562774)  ###RA_P
                pred = (output_generate_1 * 1.6863335 + 0.013257576)###RA_A
                #pred = (pred * 1.5460724 - 0.9562774)  ###RA_P
                acc = acc_calcu(target, pred, args.margin)
                correct_num_1 += acc
                acc = acc_calcu(pred, target, args.margin * 2)
                correct_num_2 += acc
                acc = acc_calcu(pred, target, args.margin * 4)
                correct_num_3 += acc
                acc = acc_calcu(pred, target, args.margin * 8)
                correct_num_4 += acc
                #print('target :', target.reshape(1, -1))
                #print('pred : ', pred.reshape(1, -1))
                #print(' ')
            else:
                target = target_RA.reshape(-1, 1)
                pred = output_generate_1.reshape(-1, 1)
                loss = F.smooth_l1_loss(pred, target)
                acc = acc_calcu(pred, target, args.margin)
                correct_num_1 += acc
                acc = acc_calcu(pred, target, args.margin * 2)
                correct_num_2 += acc
                acc = acc_calcu(pred, target, args.margin * 4)
                correct_num_3 += acc
                acc = acc_calcu(pred, target, args.margin * 8)
                correct_num_4 += acc
                #print('target :', target.reshape(1, -1))
                #print('pred : ', pred.reshape(1, -1))
                #print(' ')

            

            
            train_loss.update(loss.item(), data_1.size(0))  ####add loss_MG
            args.logger.info('Model {} Epoch: {} [{}/{} ({:.0f}%)] Loss_M1: {:.6f}'.format(phase,
                                                                                           epoch, batch_idx * len(data_1),
                                                                                           len(train_loader.dataset),
                                                                                           100. * batch_idx / len(
                                                                                               train_loader),
                                                                                           train_loss.avg))
        args.logger.info('In epoch {}, acc for margin {} is : {}'.format(epoch,
                                                                         args.margin,
                                                                         correct_num_1 / len(train_loader.dataset)))
    
    loss = {
        'loss': train_loss.avg,
        'acc_1': correct_num_1 / len(train_loader.dataset),
        'acc_2': correct_num_2 / len(train_loader.dataset),
        'acc_3': correct_num_3 / len(train_loader.dataset),
        'acc_4': correct_num_4 / len(train_loader.dataset),
    }
    return loss

def save_results_baseline(args,
                          model_1,
                          model_2,

                          train_results,
                          val_results,
                          test_results,
                          full_results,
                          optimizer_M_1,
                          optimizer_M_2,

                          epoch):

    val_acc_1 = val_results['acc_1']
    test_acc_1 = test_results['acc_1']

    if args.best_test_acc < test_acc_1:
        args.best_test_acc = copy.deepcopy(test_acc_1)
        args.best_test_acc_epoch = copy.deepcopy(epoch)


    if args.best_val_acc < val_acc_1:
        args.best_val_acc = copy.deepcopy(val_acc_1)
        args.best_val_acc_epoch = copy.deepcopy(epoch)


    if epoch == args.best_test_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_test_acc_1_' + '_model_1.pt'))
        torch.save(model_2.state_dict(), os.path.join(args.save_dir, 'best_test_acc_1_' + '_model_2_MLP.pt'))
    if epoch == args.best_val_acc_epoch:
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'best_val_acc_1_' + '_model_1.pt'))
        torch.save(model_2.state_dict(), os.path.join(args.save_dir, 'best_val_acc_1_' + '_model_2_MLP.pt'))

    args.logger.info(
        'Utill now the best test acc epoch is : {},  acc is {}'.format(args.best_test_acc_epoch, args.best_test_acc))
    args.logger.info(
        'Utill now the best val acc epoch is : {},  acc is {}'.format(args.best_val_acc_epoch, args.best_val_acc))

    full_results[epoch] = {
        'train_results': copy.deepcopy(train_results),
        'test_results': copy.deepcopy(test_results),
        'val_results': copy.deepcopy(val_results),
    }
    pickle.dump(full_results, open(os.path.join(args.save_dir, 'results.pkl'), 'wb'))
    args.logger.info(
        'In this epoch : {}, for margin {},test acc is {}'.format(args.best_val_acc_epoch, args.margin,
                                                full_results[args.best_val_acc_epoch]['test_results']['acc_1']))
    args.logger.info(
        'In this epoch : {}, for margin {}, test acc is {}'.format(args.best_val_acc_epoch, args.margin * 2,
                                                                   full_results[args.best_val_acc_epoch][
                                                                       'test_results']['acc_2']))
    args.logger.info(
        'In this epoch : {}, for margin {}, test acc is {}'.format(args.best_val_acc_epoch, args.margin * 4,
                                                                   full_results[args.best_val_acc_epoch][
                                                                       'test_results']['acc_3']))
    args.logger.info(
        'In this epoch : {}, for margin {}, test acc is {}'.format(args.best_val_acc_epoch, args.margin * 8,
                                                                   full_results[args.best_val_acc_epoch][
                                                                       'test_results']['acc_4']))

    is_best = 1
    if args.save_checkpoint > 0:
        save_checkpoint({
            'epoch': copy.deepcopy(epoch),
            'model_1': model_1.state_dict(),
            'model_2': model_2.state_dict(),
            'best_test_acc': args.best_test_acc,
            'optimizer_M_2': optimizer_M_2.state_dict(),
            'optimizer_M_1': optimizer_M_1.state_dict(),
        }, is_best, base_dir=args.save_dir)
        torch.save(model_1.state_dict(), os.path.join(args.save_dir, 'Final_model_1.pt'))
        torch.save(model_2.state_dict(), os.path.join(args.save_dir, 'Final_model_2_MLP.pt'))


def main():
    # Training settings
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader, test_loader = get_all_dataloader(args)
    if args.model == 'efficient-0':
        model_1 = EfficientNet.from_pretrained('efficientnet-b0').cuda()
        feature = model_1._fc.in_features
        model_1._fc = nn.Linear(in_features=feature, out_features=args.class_num, bias=True)
    if args.model == 'efficient-1':
	    model_1 = EfficientNet.from_pretrained('efficientnet-b1').cuda()
	    feature = model_1._fc.in_features
	    model_1._fc = nn.Linear(in_features=feature, out_features=args.class_num, bias=True)
    if args.model == 'efficient-2':
        model_1 = EfficientNet.from_pretrained('efficientnet-b2').cuda()
        feature = model_1._fc.in_features
        model_1._fc = nn.Linear(in_features=feature, out_features=args.class_num, bias=True)
    if args.model == 'efficient-3':
        model_1 = EfficientNet.from_pretrained('efficientnet-b3').cuda()
        feature = model_1._fc.in_features
        model_1._fc = nn.Linear(in_features=feature, out_features=args.class_num, bias=True)
    if args.model == 'efficient-4':
        model_1 = EfficientNet.from_pretrained('efficientnet-b4').cuda()
        feature = model_1._fc.in_features
        model_1._fc = nn.Linear(in_features=feature, out_features=args.class_num, bias=True)
    elif args.model == 'rn34':
        resnet34 = torchmodel.resnet34(pretrained=True)
        resnet34.fc = nn.Linear(512, args.class_num)
        resnet34 = resnet34.cuda()
        model_1 = resnet34
    elif args.model == 'rn50':
        resnet50 = torchmodel.resnet50(pretrained=True)
        resnet50.fc = nn.Linear(2048, args.class_num)
        resnet50 = resnet50.cuda()
        model_1 = resnet50
    elif args.model == 'rn101':
        resnet101 = torchmodel.resnet101(pretrained=False)
        resnet101.fc = nn.Linear(2048, args.class_num)
        resnet101 = resnet101.cuda()
        model_1 = resnet101
    elif args.model == 'rn152':
        resnet152 = torchmodel.resnet152(pretrained=True)
        resnet152.fc = nn.Linear(2048, args.class_num)
        resnet152 = resnet152.cuda()
        model_1 = resnet152

	
    #model_1 = RN18_front().cuda()
    model_2 = MLP(in_channels=len(args.feature_list), out_channel=args.class_num).cuda()

    if args.optimizer == 'SGD':
        optimizer_M_1 = optim.SGD([{'params': model_1.parameters(), 'lr': args.lr,
                                    'weight_decay': args.wd, 'momentum': args.momentum}])
        optimizer_M_2 = optim.SGD([{'params': model_2.parameters(), 'lr': args.lr2,
                                    'weight_decay': args.wd2, 'momentum': args.momentum}])
    elif args.optimizer == 'Adam':
        optimizer_M_1 = optim.Adam(model_1.parameters(), lr=args.lr, weight_decay=args.wd)
        optimizer_M_2 = optim.Adam(model_2.parameters(), lr=args.lr2, weight_decay=args.wd2)
    elif args.optimizer == 'RMSprop':
        optimizer_M_1 = optim.RMSprop(model_1.parameters(),lr=args.lr, alpha=args.alpha)
        optimizer_M_2 = optim.RMSprop(model_2.parameters(), lr=args.lr2, alpha=args.alpha)
   #optimizer_M_2_generate = optim.SGD(
        #[{'params': model_2_generate.parameters(), 'lr': args.lr2,
          #'weight_decay': args.wd2, 'momentum': args.momentum}])
    
    model_1 = model_1.cuda()
    full_results = {}
    args = init_metric(args)
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_results = train_baseline('train', args, model_1, model_2, train_loader,
                                           optimizer_M_1, optimizer_M_2,epoch)
            test_results = test_baseline('test', args, model_1, model_2,test_loader,
                                           optimizer_M_1,  optimizer_M_2,epoch)
            val_results = test_baseline('val', args, model_1, model_2,val_loader,
                                          optimizer_M_1, optimizer_M_2,epoch)


            adjust_learning_rate(optimizer_M_1, epoch, args, args.lr)
            adjust_learning_rate(optimizer_M_2, epoch, args, args.lr2)

            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results_baseline(
                args,
                model_1,
                model_2,
                train_results,
                val_results,
                test_results,
                full_results,
                optimizer_M_1,
                optimizer_M_2,
                epoch)
    finally:
        args.logger.info('save_results_path: %s' % args.save_dir)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)


if __name__ == '__main__':
    main()
