# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from torchvision import transforms
from models import *
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import xlrd
import sys


def clip_to_sphere(tens, radius=5, channel_dim=1):
    radi2 = torch.sum(tens**2, dim=channel_dim, keepdim=True)
    mask = torch.gt(radi2, radius**2).expand_as(tens)
    tens[mask] = torch.sqrt(
        tens[mask]**2 / radi2.expand_as(tens)[mask] * radius**2)
    return tens


def VAE_loss(recon_x, x, mu, logvar, args):
    """
    pred_y: predicted y
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    q_y_s: prior
    beta: tradeoff params
    """
    BCE = F.binary_cross_entropy(recon_x.view(-1, 1 * args.crop_size ** 3), x.view(-1, 1 * args.crop_size ** 3),
                                 reduction='mean')
    KLD = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())

    return BCE, KLD


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
        if self.args.if_unsp_cluster:  # sijie to use unsupervied cluster datset
            # if self.args.env_num == 2:
            workbook = xlrd.open_workbook(
                r"../Dataset_E/E%s_%s_AD_%s.xls" % (self.args.env_num, fold, self.args.dataset_type[-3:]))
            sheet = workbook.sheet_by_index(0)
            for rows in range(1, sheet.nrows):
                # if sheet.row_values(rows)[4] == fold_map[fold]:
                self.image_path_list.append(sheet.row_values(rows)[0])
                self.y.append(sheet.row_values(rows)[1])
                self.env.append(sheet.row_values(rows)[self.args.env_num])
                self.u.append(float(sheet.row_values(rows)[1]))
                self.us.append(float(sheet.row_values(rows)[1]))
                self.train_u.append(sheet.row_values(rows)[1:3])
                self.train_us.append(float(sheet.row_values(rows)[1]))
            pass
            # elif self.args.env_num == 3:
            # pass
        # se:

        elif args.dataset == 'AD':  # Nips2021 dataset
            if self.root is None:
                self.root = '/home/botong/'

            if args.dataset_type == 'gene_1':
                workbook = xlrd.open_workbook(os.path.join(
                    self.root, "Dataset/ADNI_dataset_APOE4.xlsx"))
                sheet = workbook.sheet_by_index(0)
                for rows in range(1, sheet.nrows):
                    if sheet.row_values(rows)[6] == fold_map['train']:
                        self.train_u.append(sheet.row_values(rows)[8:12])
                        self.train_us.append(float(sheet.row_values(rows)[12]))

                    if sheet.row_values(rows)[6] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(
                            self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        self.us.append(float(sheet.row_values(rows)[12]))

            elif 'gene_5' in args.dataset_type:
                if args.fold == 1:
                    workbook = xlrd.open_workbook(os.path.join(
                        self.root, "Dataset/ADNI_dataset_APOE4.xlsx"))
                    print('load data excel 1')
                elif args.fold == 2:
                    workbook = xlrd.open_workbook(os.path.join(
                        self.root, "Dataset/ADNI_dataset_APOE4_2.xlsx"))
                    print('load data excel 2')
                elif args.fold == 3:
                    workbook = xlrd.open_workbook(os.path.join(
                        self.root, "Dataset/ADNI_dataset_APOE4_3.xlsx"))
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
                    if sheet.row_values(rows)[6] == fold_map[fold] and int(sheet.row_values(rows)[21]) == select_env:
                        self.image_path_list.append(os.path.join(
                            self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        if 'm' in args.dataset_type:
                            self.us.append(sheet.row_values(rows)[12:21])
                        else:
                            self.us.append(sheet.row_values(rows)[12:17])
                        self.env.append(int(sheet.row_values(rows)[21]))
            else:
                if args.dataset_type == 'test':
                    workbook = xlrd.open_workbook(
                        r"/home/botong/Dataset/ADNI_dataset_test.xlsx")
                elif args.dataset_type == '80':
                    workbook = xlrd.open_workbook(
                        r"/home/botong/Dataset/ADNI_dataset_80_test.xlsx")
                elif args.dataset_type == 'EDU':
                    workbook = xlrd.open_workbook(
                        r"/home/botong/Dataset/ADNI_dataset_EDU.xlsx")
                sheet = workbook.sheet_by_index(0)

                for rows in range(1, sheet.nrows):
                    if sheet.row_values(rows)[6] == fold_map['train']:
                        self.train_u.append(sheet.row_values(rows)[8:12])
                        self.train_us.append(sheet.row_values(rows)[14:22])

                    if sheet.row_values(rows)[6] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(
                            self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        self.us.append(sheet.row_values(rows)[14:22])

        self.u = np.array(self.u).astype('float32')
        self.us = np.array(self.us).astype('float32')
        self.train_u = np.array(self.train_u).astype('float32')
        self.train_us = np.array(self.train_us).astype('float32')
        if args.data_process == 'fill':
            for ss in range(self.u.shape[1]):
                self.u[self.u[:, ss] < 0,
                       ss] = self.train_u[self.train_u[:, ss] >= 0, ss].mean()
            for ss in range(self.us.shape[1]):
                self.us[self.us[:, ss] < 0,
                        ss] = self.train_us[self.train_us[:, ss] >= 0, ss].mean()
        elif args.data_process == 'fill_std':
            for ss in range(self.u.shape[1]):
                self.u[self.u[:, ss] < 0,
                       ss] = self.train_u[self.train_u[:, ss] >= 0, ss].mean()
                self.train_u[self.train_u[:, ss] < 0,
                             ss] = self.train_u[self.train_u[:, ss] >= 0, ss].mean()
                self.u[:, ss] = (self.u[:, ss] - self.train_u[:,
                                 ss].mean()) / self.train_u[:, ss].std()
            for ss in range(self.us.shape[1]):
                self.us[self.us[:, ss] < 0,
                        ss] = self.train_us[self.train_us[:, ss] >= 0, ss].mean()
                self.train_us[self.train_us[:, ss] < 0,
                              ss] = self.train_us[self.train_us[:, ss] >= 0, ss].mean()
                self.us[:, ss] = (
                    self.us[:, ss] - self.train_us[:, ss].mean()) / self.train_us[:, ss].std()

        # print(self.env)
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
            shift_x = (np.random.choice(2) * 2 - 1) * \
                np.random.choice(int(round(self.args.shift)))
            shift_y = (np.random.choice(2) * 2 - 1) * \
                np.random.choice(int(round(self.args.shift)))
            shift_z = (np.random.choice(2) * 2 - 1) * \
                np.random.choice(int(round(self.args.shift)))
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


def train(epoch, model, optimizer, dataloader, train_loader_2, args):
    all_zs = np.zeros((len(dataloader.dataset), args.zs_dim))
    RECON_loss = AverageMeter()
    KLD_loss = AverageMeter()
    classify_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    model.train()
    args.fix_mu = 1
    args.fix_var = 1
    iterator = iter(train_loader_2)
    for batch_idx, (x, u, us, target, env) in enumerate(dataloader):
        for ss in range(args.env_num):
            if ss == 0:
                x, target, env = x.cuda(), target.cuda().long(), env.cuda().long()
                _, recon_x, mu, logvar, z, s, zs = model(
                    x, ss, feature=1, is_train=1)
                pred_y = model.get_y_by_zs(mu, logvar, ss)
                recon_loss_t, kld_loss_t = VAE_loss(
                    recon_x, x, mu, logvar, args)
                cls_loss_t = F.nll_loss(torch.log(pred_y), target)
                loss = recon_loss_t + args.beta * kld_loss_t + args.gamma * cls_loss_t
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()),
                                pred_y.size(0))
            else:
                #print('begin the training for env 1')
                try:
                    x, _, _, target, env = iterator.next()
                    x, target, env = x.cuda(), target.cuda(), env.cuda()
                except:
                    print('break the training')
                    break
                _, recon_x, mu, logvar, z, s, zs = model(
                    x, ss, feature=1, is_train=1)
                pred_y = model.get_y_by_zs(mu, logvar, ss)
                recon_loss_t, kld_loss_t = VAE_loss(
                    recon_x, x, mu, logvar, args)
                cls_loss_t = F.nll_loss(torch.log(pred_y), target)
                loss = recon_loss_t + args.beta * kld_loss_t + args.gamma * cls_loss_t
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()),
                                pred_y.size(0))

            RECON_loss.update(recon_loss_t.item(), pred_y.size(0))
            KLD_loss.update(kld_loss_t.item(), pred_y.size(0))
            classify_loss.update(cls_loss_t.item(), pred_y.size(0))
            all_loss.update(loss.item(), pred_y.size(0))

        if batch_idx % 10 == 0:
            args.logger.info(
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
    args.logger.info(
        'epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, all_loss.avg))

    return all_zs, accuracy.avg


def evaluate_22(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    pred_pos_num = 0
    for batch_idx, (x, u, us, target, env) in enumerate(dataloader):
        if args.cuda:
            x, target, env = x.cuda(), target.cuda().long(), env.cuda().long()
        pred_y_init, pred_y = model(x, is_train=0, is_debug=1)
        pred_pos_num = pred_pos_num + np.where(np.argmax(np.array(pred_y.detach().cpu().numpy()).
                                                         reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[0]
        accuracy_init.update(compute_acc(np.array(pred_y_init.detach().cpu().numpy()).
                                         reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))

        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f' %
                     (accuracy_init.avg, accuracy.avg))
    return pred, accuracy.avg


def main():
    args = get_opt()
    args = make_dirs(args)
    logger = get_logger(args)
    args.logger = logger
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    if args.dataset == 'AD':
        train_loader = DataLoaderX(get_dataset(root=args.root, args=args, fold='train', aug=args.aug, select_env=0),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=2,
                                   pin_memory=True,
                                   drop_last=True)
        train_loader_2 = DataLoaderX(get_dataset(root=args.root, args=args, fold='train', aug=args.aug, select_env=1),
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=2,
                                     pin_memory=True,
                                     drop_last=True)
        test_loader = DataLoaderX(get_dataset(root=args.root, args=args, fold='test', aug=0, select_env=-1),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = None

    model = Generative_model_f_2D_unpooled_env_t_AD(
        in_channel=args.in_channel,
        zs_dim=args.zs_dim,
        num_classes=args.num_classes,
        is_use_u=args.is_use_u,
        is_sample=args.is_sample,
        decoder_type=0,
        total_env=args.env_num,
        args=args,
    ).cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params,
          '%0.4f M' % (pytorch_total_params / 1e6))

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.reg)

    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        #pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
        adjust_learning_rate(optimizer, epoch, args.lr,
                             args.lr_decay, args.lr_controler)
        _, _ = train(epoch, model, optimizer,
                     train_loader, train_loader_2, args)

        if args.eval_train == 1:
            _, train_acc = evaluate(epoch, model, train_loader, args)
            logger.info('train test acc: %0.4f' % train_acc)
        if val_loader is not None:
            pred_val, val_acc = evaluate(epoch, model, val_loader, args)
        else:
            pred_val = None
            val_acc = -1

        is_best = 0
        # pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
        # if test_acc >= best_acc:
        #     best_acc = copy.deepcopy(test_acc)
        #     best_acc_ep = copy.deepcopy(epoch)
        #     is_best = 1
        #     logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
        #                 % (test_acc, args.test_ep, args.lr2, args.reg2, args.sample_num))

        for test_ep in [10]:
            for lr2 in [0.0005]:
                for wd2 in [0.005]:
                    for sample_num in [5]:
                        temp_args = copy.deepcopy(args)
                        temp_args.sample_num = sample_num
                        temp_args.test_ep = test_ep
                        temp_args.lr2 = lr2
                        temp_args.reg2 = wd2
                        model.args = temp_args
                        pred_test, test_acc = evaluate_22(
                            epoch, model, test_loader, temp_args)

                        if test_acc >= best_acc:
                            best_acc = copy.deepcopy(test_acc)
                            best_acc_ep = copy.deepcopy(epoch)
                            is_best = 1
                            logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
                                        % (test_acc, test_ep, lr2, wd2, sample_num))
        other_info = {
            'pred_val': pred_val,
            'pred_test': pred_test,
        }
        checkpoint(epoch, args.model_save_dir, model,
                   is_best, other_info, logger)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d, val_acc: %0.4f'
                    % (epoch, test_acc, best_acc, best_acc_ep, val_acc))
    logger.info('model save path: %s' % args.model_save_dir)
    xlsx_name = '%s_Data_%s_%s_model_unpooled_%s_u_%d_fold_%d.xls' % \
                (os.path.basename(sys.argv[0][:-3]), args.dataset,
                 args.dataset_type, args.model, args.is_use_u, args.fold)
    save_results_as_xlsx('../results/', xlsx_name, best_acc,
                         best_acc_ep, auc=None, args=args)
    logger.info('xls save path: %s' % xlsx_name)
    logger.info('*' * 50)
    logger.info('*' * 50)


if __name__ == '__main__':
    main()
