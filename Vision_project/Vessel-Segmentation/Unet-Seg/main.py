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
from PIL import Image


class Clas_ppa_train(data.Dataset):
    def __init__(self,
                 root,
                 transform1=None,
                 transform2=None,
                 fold='train',
                 ):
        super(Clas_ppa_train, self).__init__()
        self.root = root  # excel path
        self.transform1 = transform1
        self.transform2 = transform2
        self.image_path = []
        self.mask_path = []
        self.target_path = []

        #workbook1 = xlrd.open_workbook(
            #r"../Retina-Seg-Dataset/retina-seg-dataset-12-08.xls")
        workbook1 = xlrd.open_workbook(
            r"../Retina-Seg-Dataset/retina-seg-dataset-12-15-traintestval.xls")

        sheet1 = workbook1.sheet_by_index(0)
        for rows in range(1, sheet1.nrows):
            if sheet1.row_values(rows)[1] == fold:
                if sheet1.row_values(rows)[2] == 'image':
                    #print(sheet1.row_values(rows)[3], 'and', os.path.join(self.root, sheet1.row_values(rows)[3]))
                    self.image_path.append('..' + sheet1.row_values(rows)[3])
                elif sheet1.row_values(rows)[2] == 'mask':
                    self.mask_path.append('..' + sheet1.row_values(rows)[3])
                elif sheet1.row_values(rows)[2] == 'target':
                    self.target_path.append('..' + sheet1.row_values(rows)[3])


    def __getitem__(self, index):
        #print('in train datadet the len :', len(self.image_path), len(self.mask_path), len(self.target_path))
        img_path_, img_path_target, img_path_mask = self.image_path[index], self.target_path[index], self.mask_path[index]
        
        
        img_ = Image.open(img_path_)
        img_ = img_.convert('RGB')

        img_target = Image.open(img_path_target)
        img_target = img_target.convert('RGB')
        #print('1111111',img_target)
        #print(img_target.size)
        img_target = binarize_image(img_target, 200)
        #print('2222222',img_target)
        #print(img_target.size)
        img_target = Image.fromarray(img_target)
        
        img_mask = Image.open(img_path_mask)
        img_mask = img_mask.convert('RGB')

        if self.transform1 is not None:
            img_ = self.transform1(img_)
            img_target = self.transform2(img_target)
            img_mask = self.transform2(img_mask)


        return img_, img_target, img_mask

    def __len__(self):
        return len(self.image_path)


def get_all_dataloader(args):
    kwargs = {'num_workers': args.works, 'pin_memory': True}
    train_loader = DataLoaderX(
        Clas_ppa_train(args.data_root, fold='train',
                       transform1=transforms.Compose([
                           # transforms.RandomHorizontalFlip(p=0.5),
                           # transforms.RandomVerticalFlip(p=0.5),
                           #transforms.RandomRotation(30),
                           transforms.CenterCrop((576,560)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        
                       ]),
                       transform2 = transforms.Compose([
                            # transforms.RandomHorizontalFlip(p=0.5),
                            # transforms.RandomVerticalFlip(p=0.5),
                            # transforms.RandomRotation(30),
                            transforms.CenterCrop((576, 560)),
                            transforms.ToTensor(),
                            #transforms.Normalize((0.5), (0.5))
                       ])
                       ),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    args.logger.info('Load train data')
    test_loader = (DataLoaderX(
        Clas_ppa_train(args.data_root, fold='test',
                       transform1=transforms.Compose([
                           transforms.CenterCrop((576, 560)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           
                       ]),
                       transform2=transforms.Compose([
                           transforms.CenterCrop((576, 560)),
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5), (0.5))
                       ])
                       ),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    args.logger.info('Load test data')
    val_loader = (DataLoaderX(
        Clas_ppa_train(args.data_root, fold='val',
                       transform1=transforms.Compose([
                           transforms.CenterCrop((576, 560)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                       ]),
                       transform2=transforms.Compose([
                           transforms.CenterCrop((576, 560)),
                           transforms.ToTensor(),
                           #transforms.Normalize((0.5), (0.5))
                       ])
                       ),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    args.logger.info('Load val data')

    return train_loader, test_loader, val_loader


def train(args,
          model,
          train_loader,
          optimizer,
          epoch):
    model.train()
    train_loss = AverageMeter()
    eps = 1e-5

    image_sequence = 0
    for batch_idx, (image, target, mask) in enumerate(train_loader):
        image, target, mask = image.cuda(), target.cuda(), mask.cuda()

        criterion1 = nn.BCEWithLogitsLoss()
        Dice_criterion = SoftDiceLoss()
        target1 = target.detach().cpu().numpy()
        #print('target min and max',np.min(target1), np.max(target1))
        #print('target', target1)
        pred = model(image)
        pred1 = pred.detach().cpu().numpy()
        #print('pred min and max', np.min(pred1), np.max(pred1))
        #print('predt', pred1)

        #loss = criterion1(pred, target)
        loss = Dice_criterion(pred, target)
        optimizer.zero_grad()
        if args.clip == 1:
            for p in model.parameters():
                p.data.clamp_(-args.wcl, args.wcl)
        else:
            pass
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), image.size(0))
        image_sequence += len(image)
        
        args.logger.info('Model Train Epoch: {} [{}/{} ({:.0f}%)] Dice loss: {:.6f}, '.format(
            epoch, image_sequence, len(train_loader.dataset),
                   100. * image_sequence / len(train_loader.dataset), train_loss.avg,))

    loss = {
        'loss': train_loss.avg,
    }
    return loss


def evaluate(args,
             model,
             # model_ESPCN,
             test_loader,
             epoch):
    model.eval()

    test_loss = AverageMeter()
    #test_loss = 0
    batch_begin = 0
    with torch.no_grad():

        for batch_idx, (image, target, mask) in enumerate(test_loader):
            image, target, mask = image.cuda(), target.cuda(), mask.cuda()

            criterion1 = nn.BCEWithLogitsLoss()
            Dice_criterion = SoftDiceLoss()
            
            
            pred = model(image)

            #print('the pred :', pred)
            #print('-'*20)
            #loss = criterion1(pred, pred)
            loss = Dice_criterion(pred, target)
            
            #test_loss += loss
            batch_begin = batch_begin + image.size(0)
            test_loss.update(loss.item(), image.size(0))
            
            if epoch % args.plot_ep == 0:  ##########save the fake images in every 5 epochs
                save_image(args, batch_begin, pred, test_loader, epoch)
                

   

    args.logger.info('In epoch {}, test loss AUC is {:.6f}.'.format(epoch, test_loss.avg))

    args.logger.info('-'*20)

    results = {
        'loss': test_loss.avg,
    
    }
    return results


def save_results(args,
                 model,
                 train_results,
                 test_results,
                 val_results,
                 full_results,
                 optimizer,
                 epoch):
    
    test_loss = test_results['loss']
    val_loss = val_results['loss']

    if args.best_test_loss > test_loss:
        args.best_test_loss = copy.deepcopy(test_loss)
        args.best_test_loss_epoch = copy.deepcopy(epoch)

    if args.best_val_loss > val_loss:
        args.best_val_loss = copy.deepcopy(val_loss)
        args.best_val_loss_epoch = copy.deepcopy(epoch)


    if epoch == args.best_val_loss_epoch:
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_val_loss' + '_model.pt'))
    


    args.logger.info(
        'Utill now the best test loss epoch is : {},  test Dice loss is {}, test dice-coff is {}'\
            .format(args.best_test_loss_epoch, args.best_test_loss, 1-args.best_test_loss))
    
    full_results[epoch] = {
        'train_results': copy.deepcopy(train_results),
        'test_results': copy.deepcopy(test_results),
        'val_results': copy.deepcopy(val_results),
    }
    pickle.dump(full_results, open(os.path.join(args.save_dir, 'results.pkl'), 'wb'))
    args.logger.info(
        'Utill now the best val loss epoch is : {},  val Dice loss is {}, val dice-coff is {}' \
            .format(args.best_val_loss_epoch, args.best_val_loss, 1 - args.best_val_loss))
    args.logger.info(
        'It this epoch : {},  test Dice loss is {}, test dice-coff is {}' \
            .format(args.best_val_loss_epoch, full_results[args.best_val_loss_epoch]['test_results']['loss'],
                    1 - full_results[args.best_val_loss_epoch]['test_results']['loss']))

    is_best = 1
    if args.save_checkpoint > 0:
        save_checkpoint({
            'epoch': copy.deepcopy(epoch),
            'model': model.state_dict(),
            'best_test_loss': full_results[args.best_val_loss_epoch]['test_results']['loss'],
            'optimizer': optimizer.state_dict(),
        }, is_best, base_dir=args.save_dir)



def main():
    # Training settings
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, val_loader = get_all_dataloader(args)
    if args.U_net_type == 'normal':
        model = UNet(n_channels=3,
                         n_classes=1,
                         bilinear=args.bi_linear).cuda()
    if args.U_net_type == 'SmaAt':
        model = UNet_SmaAt(n_channels=3,
                     n_classes=1,
                     bilinear=args.bi_linear).cuda()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD([{'params': model.parameters(), 'lr': args.lr,
                                    'weight_decay': args.wd, 'momentum': args.momentum}
                                   ])
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': args.lr,
                                  'weight_decay': args.wd, 'betas': (0.9, 0.999)}])

    if args.optimizer == 'Rmsprop':
        #optimizer_G = optim.RMSprop([{'params': G_net.parameters(), 'lr': args.lr, 'weight_decay': args.wd}])
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    full_results = {}
    args = init_metric(args)

    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_results = train(args,
                                  model,
                                  train_loader,
                                  optimizer,
                                  epoch)

            test_results = evaluate(args,
                                         model,
                                         test_loader,
                                         epoch)
            val_results = evaluate(args,
                                    model,
                                    val_loader,
                                    epoch)


            adjust_learning_rate(optimizer, epoch, args)


            one_epoch_time = time.time() - start_time
            args.logger.info('one epoch time is %f' % (one_epoch_time))
            save_results(args,
                         model,
                         train_results,
                         test_results,
                         val_results,
                         full_results,
                         optimizer,
                         epoch)
    finally:
        args.logger.info('save_results_path: %s' % args.save_dir)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)
        args.logger.info('-' * 50)


if __name__ == '__main__':
    main()
