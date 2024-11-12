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

def train(epoch, model, optimizer, dataloader, args):
    all_zs = np.zeros((len(dataloader.dataset), args.zs_dim))
    RECON_loss = AverageMeter()
    KLD_loss = AverageMeter()
    classify_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    model.train()
    for batch_idx, (x, u, us, target) in enumerate(dataloader):
        if args.cuda:
            x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda()

        # try:
        if args.model == 'VAE' or args.model == 'sVAE' or args.model == 'VAE_f' or args.model == 'sVAE_f':
            # print(u.size(), us.size())
            _, recon_x, mu, logvar, mu_prior, logvar_prior, zs = model(x, u, feature=1)
            pred_y = model.get_pred_y(x, u)
        else:
            pred_y = model(x, u)
        # except:
        #     continue
        if args.model == 'VAE' or args.model == 'sVAE' or args.model == 'VAE_f' or args.model == 'sVAE_f':
            recon_loss, kld_loss = VAE_loss(recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, args)
            cls_loss = F.nll_loss(torch.log(pred_y), target)
            loss = recon_loss + args.beta * kld_loss + args.alpha * cls_loss
            
            all_zs[batch_begin:batch_begin + x.size(0), :] = \
                zs.detach().view(x.size(0), args.zs_dim).cpu().numpy()
            batch_begin = batch_begin + x.size(0)
        else:
            loss = F.cross_entropy(pred_y, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' \
                or args.model == 'VAE_f' or args.model == 'sVAE_f':
            RECON_loss.update(recon_loss.item(), x.size(0))
            KLD_loss.update(kld_loss.item(), x.size(0))
            classify_loss.update(cls_loss.item(), x.size(0))
        all_loss.update(loss.item(), x.size(0))
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
        
        if batch_idx % 10 == 0:
            print(
                'epoch [{}/{}], batch: {}, rec_loss:{:.4f}, kld_loss:{:.4f} cls_loss:{:.4f}, overall_loss:{:.4f},acc:{:.4f}'
                .format(epoch,
                        args.epochs,
                        batch_idx,
                        RECON_loss.avg,
                        KLD_loss.avg * args.beta,
                        classify_loss.avg,
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
    for batch_idx, (x, u, us, target) in enumerate(dataloader):
        if args.cuda:
            x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda()
        
        if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' \
                or args.model == 'VAE_f' or args.model == 'sVAE_f':
            pred_y = model.get_pred_y(x, u)
        else:
            pred_y = model(x, u)
        pred[batch_begin:batch_begin + x.size(0), :] = pred_y.detach().cpu()
        batch_begin = batch_begin + x.size(0)
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
    return pred, accuracy.avg


class get_dataset(data.Dataset):
    def __init__(self,
                 root=None,
                 fold='train',
                 args=None,
                 aug=1,
                 select_env=0):
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
                workbook = xlrd.open_workbook(os.path.join(self.root, "Dataset/ADNI_dataset_APOE4.xlsx"))
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
            
            elif 'gene_5' in args.dataset_type:
                if args.fold == 1:
                    workbook = xlrd.open_workbook(os.path.join(self.root, "Dataset/ADNI_dataset_APOE4.xlsx"))
                    print('load data excel 1')
                elif args.fold == 2:
                    workbook = xlrd.open_workbook(os.path.join(self.root, "Dataset/ADNI_dataset_APOE4_2.xlsx"))
                    print('load data excel 2')
                elif args.fold == 3:
                    workbook = xlrd.open_workbook(os.path.join(self.root, "Dataset/ADNI_dataset_APOE4_3.xlsx"))
                    print('load data excel 3')
                if 'SEX' in args.dataset_type:
                    sheet = workbook.sheet_by_index(10)
                elif 'AV45' in args.dataset_type:
                    sheet = workbook.sheet_by_index(12)
                elif 'ABE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(13)
                elif 'TAU' in args.dataset_type:
                    sheet = workbook.sheet_by_index(14)
                    u_idx = 15
                elif 'APOE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(15)
                    print('workbook.sheet_by_index(15)')
                elif 'EDU' in args.dataset_type:
                    sheet = workbook.sheet_by_index(8)
                elif 'AGE2' in args.dataset_type:
                    sheet = workbook.sheet_by_index(11)
                    u_idx = 9
                elif 'AGE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(6)
                    u_idx = 9
                elif 're' in args.dataset_type:
                    sheet = workbook.sheet_by_index(4)
                else:
                    sheet = workbook.sheet_by_index(2)
                for rows in range(1, sheet.nrows):
                    if sheet.row_values(rows)[6] == fold_map['train']:
                        self.train_u.append(sheet.row_values(rows)[u_idx])
                        if 'm' in args.dataset_type:
                            self.train_us.append(sheet.row_values(rows)[12:21])
                        else:
                            self.train_us.append(sheet.row_values(rows)[12:17])
                    if sheet.row_values(rows)[6] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[u_idx])
                        if 'm' in args.dataset_type:
                            self.us.append(sheet.row_values(rows)[12:21])
                        else:
                            self.us.append(sheet.row_values(rows)[12:17])
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
        
        self.u = np.array(self.u).reshape(len(self.u), 1).astype('float32')
        self.us = np.array(self.us).astype('float32')
        self.train_u = np.array(self.train_u).reshape(len(self.train_u), 1).astype('float32')
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
        
        # print(self.env)
        # print(self.y)
        # print(self.u)
    
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
               torch.from_numpy(np.array(self.u[index]).reshape((1,)).astype('float32')), \
               torch.from_numpy(np.array(self.us[index]).astype('float32')), \
               torch.from_numpy(np.array(self.y[index]).astype('int'))
    
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
    args = make_dirs(args)
    logger = get_logger(args)
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    if args.dataset == 'AD':
        train_loader = DataLoaderX(get_dataset(root=args.root,args=args, fold='train', aug = args.aug),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=True,
                                   drop_last = True)
        test_loader = DataLoaderX(get_dataset(root=args.root,args=args, fold='test', aug = 0),
                                   batch_size=args.test_batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,)
        val_loader = None
    if args.model == 'VAE_f':
        model = Generative_model_u(in_channel=args.in_channel,
                             u_dim=args.u_dim,
                             us_dim=args.us_dim,
                             zs_dim=args.zs_dim,
                             num_classes=args.num_classes,
                             is_use_u=args.is_use_u,
                             ).cuda()
    elif args.model == 'sVAE':
        model = sVAE_u(in_channel=args.in_channel,
                             u_dim=args.u_dim,
                             us_dim=args.us_dim,
                             zs_dim=args.zs_dim,
                             num_classes=args.num_classes,
                             is_use_u=args.is_use_u,
                             ).cuda()
    elif args.model == 'MM_F':
        model = MM_F_u(in_channel=args.in_channel,
                       u_dim=args.u_dim,
                       us_dim=args.us_dim,
                       num_classes=args.num_classes,
                       is_use_u=args.is_use_u,
                       zs_dim=args.zs_dim,
                       ).cuda()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    
    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.lr_controler)
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
                    %(epoch, test_acc, best_acc, best_acc_ep, val_acc))
    logger.info('model save path: %s'%args.model_save_dir)
    xlsx_name = '%s_Data_%s_%s_model_c_lacim_%s_u_%d_fold_%d.xls' % \
                (os.path.basename(sys.argv[0][:-3]), args.dataset, args.dataset_type, args.model, args.is_use_u, args.fold)
    save_results_as_xlsx('./results/', xlsx_name, best_acc, best_acc_ep, auc=None, args=args)
    logger.info('xls save path: %s' % xlsx_name)
    logger.info('*' * 50)
    logger.info('*' * 50)
    
if __name__ =='__main__':
    main()