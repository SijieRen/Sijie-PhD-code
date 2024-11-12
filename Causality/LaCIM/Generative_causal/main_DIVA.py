# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from torchvision import transforms
from models import *
import torch.utils.data as data
import xlrd
import torch.nn.functional as F
import sys
import torch.distributions as dist
import torch.nn as nn


def train(epoch, model, optimizer, dataloader, args):
    all_zs = np.zeros((len(dataloader.dataset), args.zs_dim))
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        if args.cuda:
            if len(batch) == 5:
                x, u, us, target, env = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[4].cuda()
            else:
                x, target, env, u = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
        # print(u, env, target)
        
        env_onehot = torch.FloatTensor(x.size(0), args.env_num).cuda()
        env_onehot.zero_()
        env_onehot.scatter_(1, env.unsqueeze(1), 1)
        env = env_onehot.float()

        y_onehot = torch.FloatTensor(x.size(0), args.num_classes).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, target.unsqueeze(1), 1)
        y_onehot = y_onehot.float()
        
        optimizer.zero_grad()
        loss, class_y_loss = model.loss_function(env, x, y_onehot)
        loss.backward()
        optimizer.step()
        
        all_loss.update(loss.item(), x.size(0))
        
        if batch_idx % 10 == 0:
            print(
                'epoch [{}/{}], batch: {}, rec_loss:{:.4f}, kld_loss:{:.4f} cls_loss:{:.4f}, overall_loss:{:.4f},acc:{:.4f}'
                    .format(epoch,
                            args.epochs,
                            batch_idx,
                            0,
                            0,
                            0,
                            all_loss.avg,
                            accuracy.avg * 100))
    
    if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' or \
            args.model == 'VAE_f' or args.model == 'sVAE_f':
        all_zs = all_zs[:batch_begin]
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, all_loss.avg))
    
    return all_zs, accuracy.avg


def evaluate(model, dataloader, args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    for batch_idx, batch in enumerate(dataloader):
        if args.cuda:
            if len(batch) == 5:
                x, u, us, target, env = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[
                    4].cuda()
            else:
                x, target, env, u = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
        
        pred_d, pred_y = model.classifier(x)
        pred[batch_begin:batch_begin + x.size(0), :] = pred_y.detach().cpu()
        batch_begin = batch_begin + x.size(0)
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
    return pred, accuracy.avg


def evaluate_only(model, dataloader, args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    label = np.zeros((dataloader.dataset.__len__(), ))
    batch_begin = 0
    for batch_idx, batch in enumerate(dataloader):
        if args.cuda:
            if len(batch) == 5:
                x, u, us, target, env = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda(), batch[
                    4].cuda()
            else:
                x, target, env, u = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
        
        pred_d, pred_y = model.classifier(x)
        pred[batch_begin:batch_begin + x.size(0), :] = pred_y.detach().cpu().numpy()
        label[batch_begin:batch_begin + x.size(0), ] = target.detach().cpu().numpy()
        batch_begin = batch_begin + x.size(0)
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
    return pred, label, accuracy.avg

class get_dataset(data.Dataset):
    def __init__(self,
                 root=None,
                 fold='train',
                 args=None,
                 aug=1):
        self.root = root
        self.args = args
        self.image_path_list = []
        self.u = []
        self.us = []
        self.train_u = []
        self.train_us = []
        self.y = []
        self.env = []
        self.aug = aug
        fold_map = {'train': 0, 'val': 1, 'test': 2}
        Label_map = {'AD': 2, 'MCI': 1, 'NC': 0}
        if args.dataset == 'AD':
            if self.root is None:
                self.root = '/home/botong/'
            
            if args.dataset_type == 'gene_1':
                workbook = xlrd.open_workbook(r"/home/botong/Dataset/ADNI_dataset_APOE4.xlsx")
                sheet = workbook.sheet_by_index(0)
                for rows in range(1, sheet.nrows):
                    if sheet.row_values(rows)[6] == fold_map['train']:
                        self.train_u.append(sheet.row_values(rows)[8:12])
                        self.train_us.append(float(sheet.row_values(rows)[12]))
                    
                    if sheet.row_values(rows)[6] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        self.us.append(float(sheet.row_values(rows)[12]))
            
            elif 'gene_3' in args.dataset_type:
                if args.fold == 1:
                    workbook = xlrd.open_workbook(r"/home/botong/Dataset/ADNI_dataset_APOE4.xlsx")
                    print('load data excel 1')
                elif args.fold == 2:
                    workbook = xlrd.open_workbook(r"/home/botong/Dataset/ADNI_dataset_APOE4_2.xlsx")
                    print('load data excel 2')
                elif args.fold == 3:
                    workbook = xlrd.open_workbook(r"/home/botong/Dataset/ADNI_dataset_APOE4_3.xlsx")
                    print('load data excel 3')
                if 'SEX' in args.dataset_type:
                    sheet = workbook.sheet_by_index(10)
                elif 'EDU' in args.dataset_type:
                    sheet = workbook.sheet_by_index(7)
                elif 'AGE2' in args.dataset_type:
                    sheet = workbook.sheet_by_index(11)
                elif 'AGE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(5)
                elif 're' in args.dataset_type:
                    sheet = workbook.sheet_by_index(3)
                else:
                    sheet = workbook.sheet_by_index(1)
                for rows in range(1, sheet.nrows):
                    if sheet.row_values(rows)[6] == fold_map['train']:
                        self.train_u.append(sheet.row_values(rows)[8:12])
                        if 'm' in args.dataset_type:
                            self.train_us.append(sheet.row_values(rows)[12:17])
                        else:
                            self.train_us.append(sheet.row_values(rows)[12:15])
                    
                    if sheet.row_values(rows)[6] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        if 'm' in args.dataset_type:
                            self.us.append(sheet.row_values(rows)[12:17])
                        else:
                            self.us.append(sheet.row_values(rows)[12:15])
            
            elif 'gene_5' in args.dataset_type:
                if args.fold == 1:
                    workbook = xlrd.open_workbook(r"%sDataset/ADNI_dataset_APOE4.xlsx" % args.root)
                    print('load data excel 1')
                elif args.fold == 2:
                    workbook = xlrd.open_workbook(r"%sDataset/ADNI_dataset_APOE4_2.xlsx" % args.root)
                    print('load data excel 2')
                elif args.fold == 3:
                    workbook = xlrd.open_workbook(r"%sDataset/ADNI_dataset_APOE4_3.xlsx" % args.root)
                    print('load data excel 3')
                if 'SEX' in args.dataset_type:
                    sheet = workbook.sheet_by_index(10)
                elif 'AV45' in args.dataset_type:
                    sheet = workbook.sheet_by_index(12)
                elif 'ABE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(13)
                elif 'TAU' in args.dataset_type:
                    sheet = workbook.sheet_by_index(14)
                elif 'APOE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(15)
                    print('workbook.sheet_by_index(15)')
                elif 'EDU' in args.dataset_type:
                    sheet = workbook.sheet_by_index(8)
                elif 'AGE2' in args.dataset_type:
                    sheet = workbook.sheet_by_index(11)
                elif 'AGE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(6)
                elif 're' in args.dataset_type:
                    sheet = workbook.sheet_by_index(4)
                else:
                    sheet = workbook.sheet_by_index(2)
                for rows in range(1, sheet.nrows):
                    if sheet.row_values(rows)[6] == fold_map['train']:
                        self.train_u.append(sheet.row_values(rows)[8:12])
                        if 'm' in args.dataset_type:
                            self.train_us.append(sheet.row_values(rows)[12:21])
                        else:
                            self.train_us.append(sheet.row_values(rows)[12:17])
                    
                    if sheet.row_values(rows)[6] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        if 'm' in args.dataset_type:
                            self.us.append(sheet.row_values(rows)[12:21])
                        else:
                            self.us.append(sheet.row_values(rows)[12:17])
                        
                        self.env.append(int(sheet.row_values(rows)[21]))
            else:
                if args.dataset_type == 'test':
                    workbook = xlrd.open_workbook(r"/home/botong/Dataset/ADNI_dataset_test.xlsx")
                elif args.dataset_type == '80':
                    workbook = xlrd.open_workbook(r"/home/botong/Dataset/ADNI_dataset_80_test.xlsx")
                elif args.dataset_type == 'EDU':
                    workbook = xlrd.open_workbook(r"/home/botong/Dataset/ADNI_dataset_EDU.xlsx")
                
                sheet = workbook.sheet_by_index(0)
                
                for rows in range(1, sheet.nrows):
                    if sheet.row_values(rows)[6] == fold_map['train']:
                        self.train_u.append(sheet.row_values(rows)[8:12])
                        self.train_us.append(sheet.row_values(rows)[14:22])
                    
                    if sheet.row_values(rows)[6] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        self.us.append(sheet.row_values(rows)[14:22])
        
        self.u = np.array(self.u).astype('float32')
        self.us = np.array(self.us).astype('float32')
        self.train_u = np.array(self.train_u).astype('float32')
        self.train_us = np.array(self.train_us).astype('float32')
        if args.data_process == 'fill':
            for ss in range(self.u.shape[1]):
                self.u[self.u[:, ss] < 0, ss] = self.train_u[self.train_u[:, ss] >= 0, ss].mean()
            for ss in range(self.us.shape[1]):
                self.us[self.us[:, ss] < 0, ss] = self.train_us[self.train_us[:, ss] >= 0, ss].mean()
        elif args.data_process == 'fill_std':
            for ss in range(self.u.shape[1]):
                self.u[self.u[:, ss] < 0, ss] = self.train_u[self.train_u[:, ss] >= 0, ss].mean()
                self.train_u[self.train_u[:, ss] < 0, ss] = self.train_u[self.train_u[:, ss] >= 0, ss].mean()
                self.u[:, ss] = (self.u[:, ss] - self.train_u[:, ss].mean()) / self.train_u[:, ss].std()
            for ss in range(self.us.shape[1]):
                self.us[self.us[:, ss] < 0, ss] = self.train_us[self.train_us[:, ss] >= 0, ss].mean()
                self.train_us[self.train_us[:, ss] < 0, ss] = self.train_us[self.train_us[:, ss] >= 0, ss].mean()
                self.us[:, ss] = (self.us[:, ss] - self.train_us[:, ss].mean()) / self.train_us[:, ss].std()
    
    def __getitem__(self, index):
        x = load_img_path(self.image_path_list[index]) / 255.0
        if self.aug == 1:
            x = self._img_aug(x).reshape(self.args.in_channel,
                                         self.args.crop_size,
                                         self.args.crop_size,
                                         self.args.crop_size, )
        else:
            crop_x = int(x.shape[0] / 2 - self.args.crop_size / 2)
            crop_y = int(x.shape[1] / 2 - self.args.crop_size / 2)
            crop_z = int(x.shape[2] / 2 - self.args.crop_size / 2)
            x = x[crop_x: crop_x + self.args.crop_size, crop_y: crop_y + self.args.crop_size,
                crop_z: crop_z + self.args.crop_size]
            x = x.reshape(self.args.in_channel,
                          self.args.crop_size,
                          self.args.crop_size,
                          self.args.crop_size, )
        return torch.from_numpy(x.astype('float32')), \
               torch.from_numpy(np.array(self.u[index]).astype('float32')), \
               torch.from_numpy(np.array(self.us[index]).astype('float32')), \
               torch.from_numpy(np.array(self.y[index]).astype('int')), \
               torch.from_numpy(np.array(self.env[index]).astype('int'))
    
    def __len__(self):
        return len(self.image_path_list)
    
    def _img_aug(self, x):
        if self.args.shift > 0:
            shift_x = (np.random.choice(2) * 2 - 1) * np.random.choice(int(round(self.args.shift)))
            shift_y = (np.random.choice(2) * 2 - 1) * np.random.choice(int(round(self.args.shift)))
            shift_z = (np.random.choice(2) * 2 - 1) * np.random.choice(int(round(self.args.shift)))
        else:
            shift_x, shift_y, shift_z = 0, 0, 0
        
        crop_x = int(x.shape[0] / 2 - self.args.crop_size / 2 + shift_x)
        crop_y = int(x.shape[1] / 2 - self.args.crop_size / 2 + shift_y)
        crop_z = int(x.shape[2] / 2 - self.args.crop_size / 2 + shift_z)
        
        x_aug = x[crop_x: crop_x + self.args.crop_size, crop_y: crop_y + self.args.crop_size,
                crop_z: crop_z + self.args.crop_size]
        
        transpose_type = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]
        
        if self.args.transpose:
            trans_idx = np.random.choice(3)
            if trans_idx != 0:
                x_aug = x_aug.transpose(transpose_type[trans_idx])
        
        if self.args.flip:
            flip_x = np.random.choice(2) * 2 - 1
            flip_y = np.random.choice(2) * 2 - 1
            flip_z = np.random.choice(2) * 2 - 1
            x_aug = x_aug[::flip_x, ::flip_y, ::flip_z]
        
        return x_aug


def main():
    args = get_opt()
    args.aux_loss_multiplier_d = args.alpha
    args.aux_loss_multiplier_y = args.beta
    
    args = make_dirs(args)
    logger = get_logger(args)
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    if args.dataset == 'AD':
        train_loader = DataLoaderX(get_dataset(root=args.root, args=args, fold='train', aug=args.aug),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=2,
                                   pin_memory=True,
                                   drop_last=True)
        test_loader = DataLoaderX(get_dataset(root=args.root, args=args, fold='test', aug=0),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True, )
        val_loader = None

        args.d_dim = args.env_num
        args.x_dim = int(256 * 256 * 3)
        args.y_dim = args.num_classes
        args.zd_dim = int(1024)
        args.zx_dim = int(1024)
        args.zy_dim = int(1024)
        args.beta_d = 1
        args.beta_x = 1
        args.beta_y = 1
        from DIVA_model_AD import DIVA
        model = DIVA(args).cuda()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('model params: ', pytorch_total_params, '%0.4f M' % (pytorch_total_params / 1e6))
    elif 'mnist' in args.dataset:
        train_loader = DataLoader(get_dataset_2D_env(root=args.root, args=args, fold='train',
                                                     transform=transforms.Compose([
                                                         transforms.RandomHorizontalFlip(p=0.5),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        
                                                     ])),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True,
                                  drop_last=True)
        test_loader = DataLoaderX(get_dataset_2D_env(root=args.root, args=args, fold='test',
                                                     transform=transforms.Compose([
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        
                                                     ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True, )
        val_loader = None
        
        args.d_dim = args.env_num
        args.x_dim = int(784 * 3)
        args.y_dim = args.num_classes
        args.zd_dim = int(32)
        args.zx_dim = int(32)
        args.zy_dim = int(32)
        args.beta_d = 1
        args.beta_x = 1
        args.beta_y = 1
        from DIVA_model import DIVA
        model = DIVA(args).cuda()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('model params: ', pytorch_total_params, '%0.4f M' % (pytorch_total_params / 1e6))
        
    else:
        train_loader = DataLoaderX(get_dataset_2D(root=args.root, args=args, fold='train',
                                                  transform=transforms.Compose([
                                                      transforms.RandomResizedCrop((256, 256)),
                                                      transforms.RandomHorizontalFlip(p=0.5),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        
                                                  ])),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=True,
                                   drop_last=True)
        test_loader = DataLoaderX(get_dataset_2D(root=args.root, args=args, fold='test',
                                                 transform=transforms.Compose([
                                                     transforms.Resize((256, 256)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        
                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True)
        val_loader = DataLoaderX(get_dataset_2D(root=args.root, args=args, fold='val',
                                                transform=transforms.Compose([
                                                    transforms.Resize((256, 256)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        
                                                ])),
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True)

        args.d_dim = args.env_num
        args.x_dim = int(256 * 256 * 3)
        args.y_dim = args.num_classes
        args.zd_dim = int(1024)
        args.zx_dim = int(1024)
        args.zy_dim = int(1024)
        args.beta_d = 1
        args.beta_x = 1
        args.beta_y = 1
        from DIVA_model_NICO import DIVA
        model = DIVA(args).cuda()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('model params: ', pytorch_total_params, '%0.4f M' % (pytorch_total_params / 1e6))

    if args.inference_only and args.load_path != '':
        model.load_state_dict(torch.load(os.path.join(args.load_path, 'best_acc.pth.tar'))['state_dict'])
        test_loader_NICO = DataLoaderX(get_dataset_NICO_inter(args=args,
                                                              transform=transforms.Compose([
                                                                  transforms.Resize((256, 256)),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                                       (0.5, 0.5, 0.5)),
    
                                                              ])),
                                       batch_size=args.test_batch_size,
                                       shuffle=False,
                                       num_workers=1,
                                       pin_memory=True)
        pred_test, label_test, test_acc = evaluate_only(model, test_loader_NICO, args)
        save_pred_label_as_xlsx(args.model_save_dir, 'pred.xls', pred_test, label_test, test_loader_NICO, args)
        logger.info('model save path: %s' % args.model_save_dir)
        print('test_acc', test_acc)
        exit(123)
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    
    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay, args.lr_controler)
        _, _ = train(epoch, model, optimizer, train_loader, args)
        if val_loader is not None:
            pred_val, val_acc = evaluate(model, val_loader, args)
        else:
            pred_val = None
            val_acc = -1
        pred_test, test_acc = evaluate(model, test_loader, args)
        if test_acc >= best_acc:
            best_acc = copy.deepcopy(test_acc)
            best_acc_ep = copy.deepcopy(epoch)
            is_best = 1
        else:
            is_best = 0
        other_info = {
            'pred_val': pred_val,
            'pred_test': pred_test,
        }
        checkpoint(epoch, args.model_save_dir, model, is_best, other_info, logger)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d, val_acc: %0.4f'
                    % (epoch, test_acc, best_acc, best_acc_ep, val_acc))
    logger.info('model save path: %s' % args.model_save_dir)
    xlsx_name = '%s_DIVA_u_%d_fold_%d.xls' % \
                (os.path.basename(sys.argv[0][:-3]), args.is_use_u,
                 args.fold)
    save_results_as_xlsx('./results/', xlsx_name, best_acc, best_acc_ep, auc=None, args=args)
    logger.info('xls save path: %s' % xlsx_name)
    logger.info('*' * 50)
    logger.info('*' * 50)


if __name__ == '__main__':
    main()