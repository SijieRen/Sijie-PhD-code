import os
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import copy
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from models import Generative_model_f_2D_unpooled_env_t_mnist_prior_share_NICO
import xlrd
from func_for_nico import compute_saliency_maps, make_fooling_image
import numpy as np


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


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, type='3d'):
        super(UnFlatten, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == '3d':
            return input.view(input.size(0), input.size(1), 1, 1, 1)
        else:
            return input.view(input.size(0), input.size(1), 1, 1)


class Generative_model_f_2D_unpooled_env_t_mnist_prior_share_decoder(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 decoder_type=0,
                 total_env=2,
                 args=None,
                 is_cuda=1
                 ):

        super(
            Generative_model_f_2D_unpooled_env_t_mnist_prior_share_decoder, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist_prior_share_decoder, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256
        self.z_dim = int(round(zs_dim * args.z_ratio))
        self.Enc_x = self.get_Enc_x_28()
        self.u_dim = total_env
        print('z_dim is ', self.z_dim)
        self.s_dim = int(self.zs_dim - self.z_dim)
        self.mean_z = []
        self.logvar_z = []
        self.mean_s = []
        self.logvar_s = []
        self.shared_s = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
        for env_idx in range(self.total_env):
            self.mean_z.append(
                nn.Sequential(
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )
            self.logvar_z.append(
                nn.Sequential(
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )
            self.mean_s.append(
                nn.Sequential(
                    self.shared_s,
                    nn.Linear(self.in_plane, self.s_dim)
                )
            )
            self.logvar_s.append(
                nn.Sequential(
                    self.shared_s,
                    nn.Linear(self.in_plane, self.s_dim)
                )
            )

        self.mean_z = nn.ModuleList(self.mean_z)
        self.logvar_z = nn.ModuleList(self.logvar_z)
        self.mean_s = nn.ModuleList(self.mean_s)
        self.logvar_s = nn.ModuleList(self.logvar_s)

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(32, self.zs_dim))
        self.logvar_zs_prior = nn.Sequential(
            nn.Linear(32, self.zs_dim))

        self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:
            # test only
            # init
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        if env_idx == 0:
                            mu_0 = mu.clone()
                        else:
                            mu_1 = mu.clone()

                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = zs[:, :self.z_dim]
                        s = zs[:, self.z_dim:]
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:
                            z_init, s_init = z, s
                            if self.args.mse_loss:
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x).view(-1, 3 *
                                                                       self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                                               3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x).view(-1,
                                                                   3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            z_init_save, s_init_save = z_init.clone(), s_init.clone()
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True

            optimizer = optim.Adam(
                params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)

            for i in range(self.args.test_ep):
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                if self.args.mse_loss:
                    BCE = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                     (raw_x).view(-1, 3 *
                                                  self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x * 0.5 + 0.5).view(-1,
                                                                          3 * self.args.image_size ** 2),
                                                 reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y, z_init_save, s_init_save, z, s, mu_0, mu_1
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = zs[:, self.z_dim:]
            return self.Dec_y(s)
        else:
            mu, logvar = self.encode(x, env)
            mu_prior, logvar_prior = self.encode_prior(x, env)
            zs = self.reparametrize(mu, logvar)
            z = zs[:, :self.z_dim]
            s = zs[:, self.z_dim:]
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(s)
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = zs[:, :self.z_dim]
        s = zs[:, self.z_dim:]
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return pred_y

    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def get_y(self, s):
        return self.Dec_y(s)

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = zs[:, self.z_dim:]
        return self.Dec_y(s)

    def encode(self, x, env_idx):
        return torch.cat([self.mean_z[env_idx](x), self.mean_s[env_idx](x)], dim=1), \
            torch.cat([self.logvar_z[env_idx](x),
                      self.logvar_s[env_idx](x)], dim=1)

    def encode_prior(self, x, env_idx):
        temp = env_idx * torch.ones(x.size()[0], 1)
        temp = temp.long().cuda()
        y_onehot = torch.FloatTensor(x.size()[0], self.args.env_num).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, temp, 1)
        # print(env_idx, y_onehot, 'onehot')
        u = self.Enc_u_prior(y_onehot)
        return self.mean_zs_prior(u), self.logvar_zs_prior(u)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(128, 128),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(64, 64),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(32, 32),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim - self.z_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 16),
            self.Fc_bn_ReLU(16, 32)
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def TConv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0,
                      bias=True, groups=1):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                               groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class get_dataset_2D_env(data.Dataset):
    def __init__(self,
                 root=None,
                 fold='train',
                 args=None,
                 transform=None):
        self.root = root
        self.args = args
        self.image_path_list = []
        self.u = []
        self.us = []
        self.train_u = []
        self.train_us = []
        self.y = []
        self.env = []
        self.transform = transform
        fold_map = {'train': 0, 'val': 1, 'test': 2}
        if args.dataset == 'mnist_2':
            if self.root is None:
                self.root = './data/colored_MNIST_0.02_env_2_0_c_2/%s/' % fold
            else:
                self.root = self.root + '%s/' % fold

            all_classes = os.listdir(self.root)
            for one_class in all_classes:
                for filename in os.listdir(os.path.join(self.root, one_class)):
                    self.u.append(float(filename[-10:-6]))
                    self.env.append(int(filename[-5:-4]))
                    self.image_path_list.append(
                        os.path.join(self.root, one_class, filename))
                    if int(one_class) <= 4:
                        self.y.append(0)
                    else:
                        self.y.append(1)

        print(self.root)

    def __getitem__(self, index):
        # print(self.image_path_list[index])
        with open(self.image_path_list[index], 'rb') as f:
            img_1 = Image.open(f)
            img_1 = Image.fromarray(np.asarray(
                img_1.convert('RGB')).astype('uint8'))
        if self.transform is not None:
            img_1 = self.transform(img_1)
        return img_1, \
            torch.from_numpy(np.array(self.y[index]).astype('int')), \
            torch.from_numpy(np.array(self.env[index]).astype('int')), \
            torch.from_numpy(np.array(self.u[index]).astype(
                'float32').reshape((1)))

    def __len__(self):
        return len(self.image_path_list)


class get_dataset_2D_env_selection_bias(data.Dataset):
    def __init__(self,
                 root=None,
                 fold='train',
                 args=None,
                 transform=None):
        self.root = root
        self.args = args
        self.image_path_list = []
        self.u = []
        self.us = []
        self.train_u = []
        self.train_us = []
        self.y = []
        self.env = []
        self.transform = transform
        fold_map = {'train': 0, 'val': 1, 'test': 2}
        if args.dataset == 'NICO':
            # todo xiugai unsp cluster results dataset(ADNI CMNIST)
            if self.args.if_unsp_cluster:  # sijie to use unsupervied cluster datset
                # if self.args.env_num == 2:
                workbook = xlrd.open_workbook(
                    r"../Dataset_E/E%s_%s_NICO.xls" % (self.args.env_num, fold))
                sheet = workbook.sheet_by_index(0)
                for rows in range(1, sheet.nrows):
                    # if sheet.row_values(rows)[4] == fold_map[fold]:
                    self.image_path_list.append(
                        os.path.join("..", sheet.row_values(rows)[0]))
                    self.y.append(sheet.row_values(rows)[1])
                    self.env.append(sheet.row_values(rows)[2])
                    self.u.append(sheet.row_values(rows)[2])
                pass

            else:  # Nips2021 dataset
                if self.root is None:
                    self.root = '/home/botong/Dataset/'

                workbook = xlrd.open_workbook(
                    r"%sNICO_dataset.xls" % self.root)
                if args.dataset_type == 'NICO_0':
                    sheet = workbook.sheet_by_index(0)
                elif args.dataset_type == 'NICO_1':
                    sheet = workbook.sheet_by_index(1)
                elif args.dataset_type == 'NICO_2':
                    sheet = workbook.sheet_by_index(2)
                elif args.dataset_type == 'NICO_3':
                    sheet = workbook.sheet_by_index(3)
                elif args.dataset_type == 'NICO_4':
                    sheet = workbook.sheet_by_index(4)
                elif args.dataset_type == 'NICO_5':
                    sheet = workbook.sheet_by_index(5)
                elif args.dataset_type == 'NICO_6':
                    sheet = workbook.sheet_by_index(6)
                elif args.dataset_type == 'NICO_7':
                    sheet = workbook.sheet_by_index(7)
                elif args.dataset_type == 'NICO_8':
                    sheet = workbook.sheet_by_index(8)
                elif args.dataset_type == 'NICO_9':
                    sheet = workbook.sheet_by_index(9)
                elif args.dataset_type == 'NICO_10':
                    sheet = workbook.sheet_by_index(10)
                elif args.dataset_type == 'NICO_11':
                    sheet = workbook.sheet_by_index(11)
                elif args.dataset_type == 'NICO_12':
                    sheet = workbook.sheet_by_index(12)
                else:
                    sheet = workbook.sheet_by_index(13)
                for rows in range(1, sheet.nrows):
                    if sheet.row_values(rows)[4] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(
                            self.root, sheet.row_values(rows)[0]))
                        self.y.append(sheet.row_values(rows)[1])
                        self.env.append(sheet.row_values(rows)[3])
                        self.u.append(sheet.row_values(rows)[3])

        elif args.dataset == 'mnist_2':
            if self.args.if_unsp_cluster:  # sijie to use unsupervied cluster datset
                # if self.args.env_num == 2:
                workbook = xlrd.open_workbook(
                    r"../Dataset_E/E%s_%s_CMNIST.xls" % (self.args.env_num, fold))
                sheet = workbook.sheet_by_index(0)
                for rows in range(1, sheet.nrows):
                    # if sheet.row_values(rows)[4] == fold_map[fold]:
                    self.image_path_list.append(
                        "../" + sheet.row_values(rows)[0])
                    self.y.append(sheet.row_values(rows)[1])
                    self.env.append(sheet.row_values(rows)[self.args.env_num])
                    self.u.append(float(sheet.row_values(rows)[0][-10:-6]))
                pass
                # elif self.args.env_num == 3:
                # pass
            else:  # Nips2021 dataset
                if self.root is None:
                    self.root = '../../data/colored_MNIST_0.02_env_2_0_c_2/%s/' % fold
                else:
                    self.root = self.root + '%s/' % fold

                all_classes = os.listdir(self.root)
                for one_class in all_classes:
                    for filename in os.listdir(os.path.join(self.root, one_class)):
                        self.u.append(float(filename[-10:-6]))
                        self.env.append(int(filename[-5:-4]))
                        self.image_path_list.append(
                            os.path.join(self.root, one_class, filename))
                        if int(one_class) <= 4:
                            self.y.append(0)
                        else:
                            self.y.append(1)
        # elif args.dataset == 'mnist_2_c5':
        #     if self.root is None:
        #         self.root = '../data/colored_MNIST_0.02_env_2_c_5/%s/' % fold
        #     else:
        #         self.root = self.root + '%s/' % fold
        #     all_classes = os.listdir(self.root)
        #     for one_class in all_classes:
        #         for filename in os.listdir(os.path.join(self.root, one_class)):
        #             self.u.append(float(filename[-10:-6]))
        #             self.env.append(int(filename[-5:-4]))
        #             self.image_path_list.append(
        #                 os.path.join(self.root, one_class, filename))
        #             if int(one_class) <= 1:
        #                 self.y.append(0)
        #             elif int(one_class) <= 3:
        #                 self.y.append(1)
        #             elif int(one_class) <= 5:
        #                 self.y.append(2)
        #             elif int(one_class) <= 7:
        #                 self.y.append(3)
        #             else:
        #                 self.y.append(4)
        # elif args.dataset == 'mnist_5_c5':
        #     if self.root is None:
        #         self.root = '../data/colored_MNIST_0.02_env_2_c_5/%s/' % fold
        #     else:
        #         self.root = self.root + '%s/' % fold
        #     all_classes = os.listdir(self.root)
        #     for one_class in all_classes:
        #         for filename in os.listdir(os.path.join(self.root, one_class)):
        #             self.u.append(float(filename[-10:-6]))
        #             self.env.append(int(filename[-5:-4]))
        #             self.image_path_list.append(
        #                 os.path.join(self.root, one_class, filename))
        #             if int(one_class) <= 1:
        #                 self.y.append(0)
        #             elif int(one_class) <= 3:
        #                 self.y.append(1)
        #             elif int(one_class) <= 5:
        #                 self.y.append(2)
        #             elif int(one_class) <= 7:
        #                 self.y.append(3)
        #             else:
        #                 self.y.append(4)
        print(self.root)

    def __getitem__(self, index):
        # print(self.image_path_list[index])
        with open(self.image_path_list[index], 'rb') as f:
            img_1 = Image.open(f)
            if '225. cat_mm8_2-min.png' in self.image_path_list[index]:
                img_1 = np.asarray(img_1.convert('RGBA'))[:, :, :3]
                img_1 = Image.fromarray(img_1.astype('uint8'))
            else:
                img_1 = Image.fromarray(np.asarray(
                    img_1.convert('RGB')).astype('uint8'))
        if self.transform is not None:
            img_1 = self.transform(img_1)
        return img_1, \
            torch.from_numpy(np.array(self.y[index]).astype('int')), \
            torch.from_numpy(np.array(self.env[index]).astype('int')), \
            torch.from_numpy(np.array(self.u[index]).astype(
                'float32').reshape((1)))

    def __len__(self):
        return len(self.image_path_list)


def evaluate(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    z_init = np.zeros((dataloader.dataset.__len__(), 16))
    s_init = np.zeros((dataloader.dataset.__len__(), 16))
    z = np.zeros((dataloader.dataset.__len__(), 16))
    s = np.zeros((dataloader.dataset.__len__(), 16))
    zs_0 = np.zeros((dataloader.dataset.__len__(), 32))
    zs_1 = np.zeros((dataloader.dataset.__len__(), 32))
    label = np.zeros((dataloader.dataset.__len__(), ))
    x_img = np.zeros((dataloader.dataset.__len__(), 3, 256, 256))

    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    pred_pos_num = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if x.size(0) == 1:
            if args.cuda:
                x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
            # pred_y_init, pred_y, z_init_t, s_init_t, z_t, s_t, mu_0, mu_1 = model(
            #     x, is_train=0, is_debug=1)
            pred_y_init, pred_y = model(x, is_train=0, is_debug=1)
            # z_init[batch_begin:batch_begin +
            #        x.size(0), :] = z_init_t.detach().cpu().numpy()
            # s_init[batch_begin:batch_begin +
            #        x.size(0), :] = s_init_t.detach().cpu().numpy()
            # z[batch_begin:batch_begin+x.size(0), :] = z_t.detach().cpu().numpy()
            # s[batch_begin:batch_begin+x.size(0), :] = s_t.detach().cpu().numpy()
            # zs_0[batch_begin:batch_begin +
            #      x.size(0), :] = mu_0.detach().cpu().numpy()
            # zs_1[batch_begin:batch_begin +
            #      x.size(0), :] = mu_1.detach().cpu().numpy()
            pred[batch_begin:batch_begin +
                 x.size(0), :] = pred_y.detach().cpu().numpy()
            label[batch_begin:batch_begin +
                  x.size(0), ] = target.detach().cpu().numpy()
            x_img[batch_begin:batch_begin +
                  x.size(0), :, :, :] = x.detach().cpu().numpy()

            saliency = compute_saliency_maps(x, target, model)
            X_fooling = make_fooling_image(x, target, model)
            s_max, s_min = saliency.max(), saliency.min()
            saliency_rescaled = (saliency-s_min)/(s_max - s_min)

            x_show = x.squeeze(0).detach().permute(
                1, 2, 0).cpu().numpy()*0.5 + 0.5
            X_fooling_difference = X_fooling - x
            print("difference: ", X_fooling_difference.max(),
                  X_fooling_difference.min())


            # saliency_rescaled[saliency_rescaled > 0.4] = 0.5
            # saliency_rescaled = -(saliency_rescaled-1)


            # saliency_rescaled[saliency_rescaled > 0.3] = 0.5
            # print("saliency_rescaled: ", saliency_rescaled.max(),
            #       saliency_rescaled.min())
            # for ii in range(3):
            # img_list = []
            # img_list.append(x_show)
            # img_list.append((20*X_fooling_difference).squeeze(0).detach().permute(
            #     1, 2, 0).cpu().numpy()*0.5 + 0.5)
            # img_list.append(saliency_rescaled.squeeze(
            #     0).detach().cpu().numpy())
            # # for ii in range(3):
            # # print(img_list[0].shape)
            # # print(img_list[1].shape)
            # # print(img_list[2].shape)
            # print("batch: ", batch_begin)
            # plt.subplot(1, 3, 0 + 1)
            # plt.imshow(img_list[0])
            # plt.axis('off')
            # plt.subplot(1, 3, 1 + 1)
            # plt.imshow(img_list[1])
            # plt.axis('off')
            # plt.subplot(1, 3, 2 + 1)
            # plt.imshow(img_list[2], cmap=plt.cm.gray)
            # plt.axis('off')
            if not os.path.exists('./96_1_confounding_bias_saved_img_grdient_NICO_test/'):
                os.makedirs('./96_1_confounding_bias_saved_img_grdient_NICO_test/')
            # plt.savefig('./93_2_confounding_bias_saved_img_grdient_NICO_train/%03d.png' %
            #             (batch_begin))
            img1=Image.fromarray(x_show.astype(np.uint8))
            img1.save('./96_1_confounding_bias_saved_img_grdient_NICO_test/%03d_1.png' % (batch_begin))
            img2 = (20*X_fooling_difference).squeeze(0).detach().permute(
                1, 2, 0).cpu().numpy()*0.5 + 0.5
            img2 = Image.fromarray(img2.astype(np.uint8))
            img2.save('./96_1_confounding_bias_saved_img_grdient_NICO_test/%03d_2.png' % (batch_begin))
            img3=saliency_rescaled.squeeze(0).detach().cpu().numpy()
            img3 = Image.fromarray(img3.astype(np.uint8))
            img3.save('./96_1_confounding_bias_saved_img_grdient_NICO_test/%03d_3.png' % (batch_begin))

            pred_pos_num = pred_pos_num + np.where(np.argmax(np.array(pred_y.detach().cpu().numpy()).
                                                             reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[0]
            accuracy_init.update(compute_acc(np.array(pred_y_init.detach().cpu().numpy()).
                                             reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
            accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                        reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        else:
            pass
        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    print('init_acc: %0.4f, after acc: %0.4f' %
          (accuracy_init.avg, accuracy.avg))
    return pred, label  # , accuracy.avg, z_init, s_init, z, s, x_img, zs_0, zs_1


def compute_acc(pred, target):
    return (np.sum(np.argmax(pred, axis=1) == target).astype('int')) / pred.shape[0]


class params(object):
    def __init__(self):
        self.in_channel = 3
        self.zs_dim = 32
        self.num_classes = 2
        self.env_num = 8
        self.z_ratio = 0.5
        self.lr2 = 0.003
        self.reg2 = 0.008
        self.test_ep = 42
        self.dataset = 'NICO'
        self.data_process = 'fill_std'
        self.is_reparameterize = 0
        # nips 2021
        # self.eval_path = './results/mnist_2/save_model_VAE_f_share_decoder_test_32_4704.0_0.1_lr_0.0100_80_0.5000_wd_0.0005_2021-03-28_18-25-11'
        # TPAMI 2023
        # env=3
        # self.eval_path = '/home/ruxin2/PPA-classification-prediction/Causality/LaCIM-followUpWorks/results/mnist_2/save_model_VAE_f_share_test_32_4704.0_0.1_lr_0.0100_80_0.5000_wd_0.0005_2022-12-02_23-01-00'
        # env=2
        # 54 /home/ruxin2/PPA-classification-prediction/Causality/results/NICO/save_model_VAE_f_share_NICO_10_2_10.0_0.1_lr_0.0100_60_0.5000_wd_0.0005_2023-05-20_20-37-19
        # 5.20网上继续试
        # self.eval_path = '/home/ruxin2/PPA-classification-prediction/Causality/results/NICO/save_model_VAE_f_share_NICO_10_2_10.0_0.1_lr_0.0100_60_0.5000_wd_0.0005_2023-05-20_20-37-19'
        # env 8
        #../results/NICO/save_model_VAE_f_share_NICO_10_2_10.0_0.1_lr_0.0100_60_0.5000_wd_0.0005_2023-09-03_18-18-32/best_acc.pth.tar
        self.eval_path = '/home/ruxin2/PPA-classification-prediction/Causality/results/NICO/save_model_VAE_f_share_NICO_10_2_10.0_0.1_lr_0.0100_60_0.5000_wd_0.0005_2023-09-03_18-18-32'

        self.root = '/home/ruxin2/PPA-classification-prediction/Causality/Dataset/'
        self.model = 'VAE_f_share'  # 'shared'#
        self.u_dim = 1
        self.us_dim = 1
        self.is_use_u = 1
        self.test_batch_size = 1
        self.cuda = 1
        self.sample_num = 10
        self.image_size = 256
        self.eval_optim = 'sgd'
        self.use_best = 1
        self.mse_loss = 1
        self.is_sample = 1
        self.if_unsp_cluster = 0
        self.dataset_type = "NICO_10"


args = params()
# sijie TPAMI 2023 xiugai test data loader
test_loader = DataLoaderX(get_dataset_2D_env_selection_bias(root=args.root, args=args, fold='test',
                                                            transform=transforms.Compose([
                                                                transforms.Resize(
                                                                    (256, 256)),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(
                                                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                            ])),
                          batch_size=args.test_batch_size,
                          shuffle=False,
                          num_workers=1,
                          pin_memory=True)


model = Generative_model_f_2D_unpooled_env_t_mnist_prior_share_NICO(in_channel=args.in_channel,
                                                                    zs_dim=args.zs_dim,
                                                                    num_classes=args.num_classes,
                                                                    decoder_type=1,
                                                                    total_env=args.env_num,
                                                                    args=args,
                                                                    ).cuda()
# model = Generative_model_f_2D_unpooled_env_t_mnist_prior_share_decoder(in_channel=args.in_channel,
#                                                                        zs_dim=args.zs_dim,
#                                                                        num_classes=args.num_classes,
#                                                                        decoder_type=1,
#                                                                        total_env=args.env_num,
#                                                                        args=args
#                                                                        ).cuda()


check = torch.load('%s/checkpoints.pth.tar' % args.eval_path,
                   map_location=torch.device('cpu'))  # best_acc
model.load_state_dict(check['state_dict'], strict=True)
model = model.cuda()
# pred, label, _, z_init, s_init, z, s, x_img, mu_0, mu_1 = evaluate(
#     0, model, test_loader, args)
pred, label = evaluate(0, model, test_loader, args)


# score = model.get_pred_y()

# if not os.path.exists('./selection_bias_saved_img_fix_z/'):
#     os.makedirs('./selection_bias_saved_img_fix_z/')
# plt.imsave('./selection_bias_saved_img_fix_z/fix_z_dim%03d_%05d.png' %
#            (dim, i), plot_img)
