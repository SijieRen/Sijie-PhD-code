import os
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import copy
import argparse
import time
import logging
import xlrd
import xlwt
from xlutils.copy import copy as x_copy
import imageio
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from PIL import Image


def save_pred_label_as_xlsx(root, xlsx_name, pred, label, dataloader, args=None):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('Sheet1')
    worksheet.write(0, 0, 'label')
    worksheet.write(0, 1, 'pred')
    worksheet.write(0, 2, 'path')
    for i in range(label.shape[0]):
        worksheet.write(i + 1, 0, label[i])
        worksheet.write(i + 1, 1, int(np.argmax(pred[i, :])))
        worksheet.write(i + 1, 2, dataloader.dataset.image_path_list[i])

    workbook.save(os.path.join(root, xlsx_name))


def save_results_as_xlsx(root, xlsx_name, acc, acc_ep, auc=None, args=None):
    if not os.path.exists(os.path.join(root, xlsx_name)):
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet('Sheet1')
        worksheet.write(0, 0, 'exp_index')
        worksheet.write(0, 1, 'lr')
        worksheet.write(0, 2, 'reg')
        worksheet.write(0, 3, 'epochs')
        worksheet.write(0, 4, 'model')
        worksheet.write(0, 5, 'alpha')
        worksheet.write(0, 6, 'beta')
        worksheet.write(0, 7, 'gamma')
        worksheet.write(0, 8, 'dataset')
        worksheet.write(0, 9, 'dataset_type')
        worksheet.write(0, 10, 'lr_decay')
        worksheet.write(0, 11, 'lr_controler')
        worksheet.write(0, 12, 'is_use_u')
        worksheet.write(0, 13, 'acc')
        worksheet.write(0, 14, 'acc_ep')
        worksheet.write(0, 15, 'auc')
        worksheet.write(0, 16, 'seed')
        worksheet.write(0, 17, 'fold')
        worksheet.write(0, 18, 'root')
        worksheet.write(0, 19, 'data_root')
        worksheet.write(0, 20, 'sample_num')
        worksheet.write(0, 21, 'lr2')
        worksheet.write(0, 22, 'reg2')
        idx = 1
        worksheet.write(idx, 0, 1)
        worksheet.write(idx, 1, args.lr)
        worksheet.write(idx, 2, args.reg)
        worksheet.write(idx, 3, args.epochs)
        worksheet.write(idx, 4, args.model)
        worksheet.write(idx, 5, args.alpha)
        worksheet.write(idx, 6, args.beta)
        worksheet.write(idx, 7, args.gamma)
        worksheet.write(idx, 8, args.dataset)
        worksheet.write(idx, 9, args.dataset_type)
        worksheet.write(idx, 10, args.lr_decay)
        worksheet.write(idx, 11, args.lr_controler)
        worksheet.write(idx, 12, args.is_use_u)
        worksheet.write(idx, 13, acc)
        worksheet.write(idx, 14, acc_ep)
        worksheet.write(idx, 15, args.z_ratio)
        worksheet.write(idx, 16, args.seed)
        worksheet.write(idx, 17, args.fold)
        worksheet.write(idx, 18, args.model_save_dir)
        worksheet.write(idx, 19, args.root)
        worksheet.write(idx, 20, args.sample_num)
        worksheet.write(idx, 21, args.lr2)
        worksheet.write(idx, 22, args.reg2)

        workbook.save(os.path.join(root, xlsx_name))
    else:
        rb = xlrd.open_workbook(os.path.join(root, xlsx_name))
        wb = x_copy(rb)
        worksheet = wb.get_sheet(0)
        idx = len(worksheet.get_rows())

        worksheet.write(idx, 0, idx)
        worksheet.write(idx, 1, args.lr)
        worksheet.write(idx, 2, args.reg)
        worksheet.write(idx, 3, args.epochs)
        worksheet.write(idx, 4, args.model)
        worksheet.write(idx, 5, args.alpha)
        worksheet.write(idx, 6, args.beta)
        worksheet.write(idx, 7, args.gamma)
        worksheet.write(idx, 8, args.dataset)
        worksheet.write(idx, 9, args.dataset_type)
        worksheet.write(idx, 10, args.lr_decay)
        worksheet.write(idx, 11, args.lr_controler)
        worksheet.write(idx, 12, args.is_use_u)
        worksheet.write(idx, 13, acc)
        worksheet.write(idx, 14, acc_ep)
        worksheet.write(idx, 15, args.z_ratio)
        worksheet.write(idx, 16, args.seed)
        worksheet.write(idx, 17, args.fold)
        worksheet.write(idx, 18, args.model_save_dir)
        worksheet.write(idx, 19, args.root)
        worksheet.write(idx, 20, args.sample_num)
        worksheet.write(idx, 21, args.lr2)
        worksheet.write(idx, 22, args.reg2)

        wb.save(os.path.join(root, xlsx_name))


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def load_img_path(filepath):
    img = imageio.imread(filepath)
    img_shape = img.shape[1]
    img_3d = img.reshape((img_shape, img_shape, img_shape))
    img_3d = img_3d.transpose((1, 2, 0))
    return img_3d


def mean_nll(logits, y):
    return F.cross_entropy(logits, y)


def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


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
        self.aug = aug
        fold_map = {'train': 0, 'val': 1, 'test': 2}
        Label_map = {'AD': 2, 'MCI': 1, 'NC': 0}
        if args.dataset == 'AD':
            if self.root is None:
                self.root = '/home/botong/'

            if args.dataset_type == 'gene_1':
                workbook = xlrd.open_workbook(
                    r"/home/botong/Dataset/ADNI_dataset_APOE4.xlsx")
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

            elif 'gene_3' in args.dataset_type:
                if args.fold == 1:
                    workbook = xlrd.open_workbook(
                        r"/home/botong/Dataset/ADNI_dataset_APOE4.xlsx")
                    print('load data excel 1')
                elif args.fold == 2:
                    workbook = xlrd.open_workbook(
                        r"/home/botong/Dataset/ADNI_dataset_APOE4_2.xlsx")
                    print('load data excel 2')
                elif args.fold == 3:
                    workbook = xlrd.open_workbook(
                        r"/home/botong/Dataset/ADNI_dataset_APOE4_3.xlsx")
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
                        self.image_path_list.append(os.path.join(
                            self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        if 'm' in args.dataset_type:
                            self.us.append(sheet.row_values(rows)[12:17])
                        else:
                            self.us.append(sheet.row_values(rows)[12:15])

            elif 'gene_5' in args.dataset_type:
                if args.fold == 1:
                    workbook = xlrd.open_workbook(
                        r"%sDataset/ADNI_dataset_APOE4.xlsx" % self.root)
                    print('load data excel 1')
                elif args.fold == 2:
                    workbook = xlrd.open_workbook(
                        r"%sDataset/ADNI_dataset_APOE4_2.xlsx" % self.root)
                    print('load data excel 2')
                elif args.fold == 3:
                    workbook = xlrd.open_workbook(
                        r"%sDataset/ADNI_dataset_APOE4_3.xlsx" % self.root)
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
                        self.image_path_list.append(os.path.join(
                            self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        if 'm' in args.dataset_type:
                            self.us.append(sheet.row_values(rows)[12:21])
                        else:
                            self.us.append(sheet.row_values(rows)[12:17])
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

    def __getitem__(self, index):
        x = load_img_path(self.image_path_list[index]) / 255.0
        if self.aug == 1:
            x = self._img_aug(x).reshape(self.args.in_channel,
                                         self.args.crop_size,
                                         self.args.crop_size,
                                         self.args.crop_size,)
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


class get_dataset_2D(data.Dataset):
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
            if self.root is None:
                self.root = '/home/botong/Dataset/'

            workbook = xlrd.open_workbook(r"%sNICO_dataset.xls" % self.root)
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
                    self.u.append(sheet.row_values(rows)[5:5+4])

        elif args.dataset == 'mnist_2':
            if self.root is None:
                self.root = '../data/colored_MNIST_0.02_env_2_0_c_2/%s/' % fold
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
                'float32').reshape((len(self.u[index]))))

    def __len__(self):
        return len(self.image_path_list)


class get_dataset_NICO_inter(data.Dataset):
    def __init__(self,
                 root=None,
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

        if self.root is None:
            self.root = '/home/botong/Dataset/'

        for files in os.listdir(os.path.join(self.root, './NICO/replace/')):
            if files[-4:] == '.png' or files[-4:] == '.jpg':
                self.image_path_list.append(os.path.join(
                    self.root, './NICO/replace/', files))
                if 'cat' in files:
                    self.y.append(0)
                else:
                    self.y.append(1)
                self.u.append([0.5, 0.5, 0.5, 0.5])
                self.env.append(-1)
        print(self.root, len(self.y))

    def __getitem__(self, index):
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
                'float32').reshape((len(self.u[index]))))

    def __len__(self):
        return len(self.image_path_list)


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
        if args.dataset == 'NICO':
            if self.root is None:
                self.root = '/home/botong/Dataset/'

            workbook = xlrd.open_workbook(r"%sNICO_dataset.xls" % self.root)
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
            if self.root is None:
                self.root = '../data/colored_MNIST_0.02_env_2_0_c_2/%s/' % fold
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
        elif args.dataset == 'mnist_2_c5':
            if self.root is None:
                self.root = '../data/colored_MNIST_0.02_env_2_c_5/%s/' % fold
            else:
                self.root = self.root + '%s/' % fold
            all_classes = os.listdir(self.root)
            for one_class in all_classes:
                for filename in os.listdir(os.path.join(self.root, one_class)):
                    self.u.append(float(filename[-10:-6]))
                    self.env.append(int(filename[-5:-4]))
                    self.image_path_list.append(
                        os.path.join(self.root, one_class, filename))
                    if int(one_class) <= 1:
                        self.y.append(0)
                    elif int(one_class) <= 3:
                        self.y.append(1)
                    elif int(one_class) <= 5:
                        self.y.append(2)
                    elif int(one_class) <= 7:
                        self.y.append(3)
                    else:
                        self.y.append(4)
        elif args.dataset == 'mnist_5_c5':
            if self.root is None:
                self.root = '../data/colored_MNIST_0.02_env_2_c_5/%s/' % fold
            else:
                self.root = self.root + '%s/' % fold
            all_classes = os.listdir(self.root)
            for one_class in all_classes:
                for filename in os.listdir(os.path.join(self.root, one_class)):
                    self.u.append(float(filename[-10:-6]))
                    self.env.append(int(filename[-5:-4]))
                    self.image_path_list.append(
                        os.path.join(self.root, one_class, filename))
                    if int(one_class) <= 1:
                        self.y.append(0)
                    elif int(one_class) <= 3:
                        self.y.append(1)
                    elif int(one_class) <= 5:
                        self.y.append(2)
                    elif int(one_class) <= 7:
                        self.y.append(3)
                    else:
                        self.y.append(4)
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


def get_opt():
    parser = argparse.ArgumentParser(description='PyTorch')
    # Model parameters
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',  # test_batch_size������
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,  # GPU������Ĭ��ΪFalse
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model', type=str, default='VAE')
    parser.add_argument('--root', type=str, default='/home/botong/Dataset/')
    parser.add_argument('--solve', type=str, default='none')
    parser.add_argument('--eval_path', type=str, default='')
    parser.add_argument('--load_path', type=str, default='')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--reg', type=float, default=0.0005)
    parser.add_argument('--reg2', type=float, default=0.00005)
    parser.add_argument('--reg3', type=float, default=0.00005)
    parser.add_argument('--lr2', type=float, default=0.001)
    parser.add_argument('--lr3', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_controler', type=int, default=80)
    parser.add_argument('--sample_type', type=int, default=0)
    parser.add_argument('--is_cuda', type=int, default=0)
    parser.add_argument('--smaller_net', type=int, default=0)
    parser.add_argument('--is_bn', type=int, default=1)
    parser.add_argument('--rex', type=int, default=0)
    parser.add_argument('--inference_only', type=int, default=0)

    # Dataset
    parser.add_argument('--dataset', type=str, default='AD')
    parser.add_argument('--dataset_type', type=str, default='test')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--data_process', type=str, default='none')
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--aug', type=int, default=1)
    parser.add_argument('--crop_size', type=int, default=48)
    parser.add_argument('--transpose', type=int, default=1)
    parser.add_argument('--flip', type=int, default=1)
    parser.add_argument('--shift', type=int, default=5)
    parser.add_argument('--sample_num', type=int, default=1)
    parser.add_argument('--one_hot', type=int, default=0)
    parser.add_argument('--decoder_type', type=int, default=0)
    parser.add_argument('--worker', type=int, default=2)
    parser.add_argument('--mse_loss', type=int, default=0)

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--zs_dim', type=int, default=256)
    parser.add_argument('--u_dim', type=int, default=4)
    parser.add_argument('--us_dim', type=int, default=8)
    parser.add_argument('--is_use_u', type=int, default=1)
    parser.add_argument('--fix_mu', type=int, default=0)
    parser.add_argument('--fix_var', type=int, default=1)
    parser.add_argument('--KLD_type', type=int, default=1)
    parser.add_argument('--zs_time', type=int, default=1)
    parser.add_argument('--train_u_only', type=int, default=0)
    parser.add_argument('--env_num', type=int, default=8)
    parser.add_argument('--epoch_in', type=int, default=10)
    parser.add_argument('--is_sample', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--test_ep', type=int, default=50)
    parser.add_argument('--train_alpha', type=int, default=0)
    parser.add_argument('--decay_test_lr', type=int, default=0)
    parser.add_argument('--eval_optim', type=str, default='adam')
    parser.add_argument('--use_best', type=int, default=1)
    parser.add_argument('--more_shared', type=int, default=0)
    parser.add_argument('--more_layer', type=int, default=0)
    parser.add_argument('--eval_train', type=int, default=0)
    parser.add_argument('--is_IRM', type=int, default=0)
    parser.add_argument('--alpha_epoch', type=int, default=10)

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--alpha2', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--gamma2', type=float, default=1.0)
    parser.add_argument('--lambd', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--dp', type=float, default=0.25)
    parser.add_argument('--z_ratio', type=float, default=0.5)

    parser.add_argument('--hidden', type=int, nargs='+',
                        default=[512, 1024, 512], help='')

    # whether use unsupervised cluster results
    parser.add_argument('--if_unsp_cluster', type=int, default=0)
    parser.add_argument('--is_reparameterize', type=int, default=0,
                        help="if reparameterize in model")

    # for HRM
    parser.add_argument('--hard_sum', type=int, default=20)
    parser.add_argument('--overall_threshold', type=float, default=0.2)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def make_dirs(args):
    current_time = time.strftime(
        '%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    args.model_save_dir = '../results/%s/save_model_%s_%s_%d_%0.1f_%0.1f_lr_%0.4f_%d_%0.4f_wd_%0.4f_%s/'\
                          % (args.dataset, args.model, args.dataset_type, args.beta, args.alpha,
                             args.lambd, args.lr, args.lr_controler, args.lr_decay, args.reg, current_time)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    return args


def get_logger(opt):
    logger = logging.getLogger('AL')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler('{}training.log'.format(opt.model_save_dir))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def compute_acc(pred, target):
    return (np.sum(np.argmax(pred, axis=1) == target).astype('int')) / pred.shape[0]


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


def VAE_loss(recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, args):
    """
    pred_y: predicted y
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    q_y_s: prior
    beta: tradeoff params
    """
    eps = 1e-5
    if args.dataset == 'AD':
        BCE = F.binary_cross_entropy(
            recon_x.view(-1, 48 ** 3), x.view(-1, 48 ** 3), reduction='mean')
    elif 'mnist' in args.dataset:
        x = x * 0.5 + 0.5
        BCE = F.binary_cross_entropy(
            recon_x.view(-1, 3 * 28 ** 2), x.view(-1, 3 * 28 ** 2), reduction='mean')
    else:
        x = x * 0.5 + 0.5
        BCE = F.binary_cross_entropy(
            recon_x.view(-1, 3 * 256 ** 2), x.view(-1, 3 * 256 ** 2), reduction='mean')
    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = KLD_element.mul_(-0.5).mean()
    if args.fix_mu == 1:
        mu_prior = torch.zeros(mu_prior.size()).cuda()
    if args.fix_var == 1:
        logvar_prior = torch.ones(logvar_prior.size()).cuda()
    if args.KLD_type == 1:
        KLD_element = torch.log(logvar_prior.exp() ** 0.5 / logvar.exp() ** 0.5) + \
            0.5 * ((mu - mu_prior).pow(2) + logvar.exp()) / \
            logvar_prior.exp() - 0.5
        KLD = KLD_element.mul_(1).mean()
    else:
        log_p_zs = torch.log(eps + (1 / ((torch.exp(logvar_prior) ** 0.5) * np.sqrt(2 * np.pi))) *
                             torch.exp(-0.5 * ((zs - mu_prior) / (torch.exp(logvar_prior) ** 0.5)) ** 2))
        log_q_zs = torch.log(eps + (1 / ((torch.exp(logvar) ** 0.5) * np.sqrt(2 * np.pi))) *
                             torch.exp(-0.5 * ((zs - mu) / (torch.exp(logvar) ** 0.5)) ** 2))
        KLD = (log_q_zs - log_p_zs).sum() / x.size(0)

    return BCE, KLD


def VAE_old_loss(pred_y, recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, q_y_s, y, args):
    """
    pred_y: predicted y
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    q_y_s: prior
    beta: tradeoff params
    """
    eps = 1e-5
    BCE_all = F.binary_cross_entropy(
        recon_x.view(-1, 48 ** 3), x.view(-1, 48 ** 3), reduction='none').mean(dim=1)
    if args.fix_mu == 1:
        mu_prior = torch.zeros(mu_prior.size()).cuda()
    if args.fix_var == 1:
        logvar_prior = torch.ones(logvar_prior.size()).cuda()

    log_p_zs = torch.log(eps + (1 / ((torch.exp(logvar_prior) ** 0.5) * np.sqrt(2 * np.pi))) *
                         torch.exp(-0.5 * ((zs - mu_prior) / (torch.exp(logvar_prior) ** 0.5)) ** 2)).mean(dim=1)
    log_q_zs = torch.log(eps + (1 / ((torch.exp(logvar) ** 0.5) * np.sqrt(2 * np.pi))) *
                         torch.exp(-0.5 * ((zs - mu) / (torch.exp(logvar) ** 0.5)) ** 2)).mean(dim=1)
    for ss in range(x.size(0)):
        if ss == 0:
            BCE = (q_y_s[ss][y[ss]] / pred_y[ss][y[ss]]) * BCE_all[ss]
            KLD = (q_y_s[ss][y[ss]] / pred_y[ss][y[ss]]) * \
                (log_q_zs[ss] - log_p_zs[ss]).mean()
        else:
            BCE = torch.add(
                BCE, ((q_y_s[ss][y[ss]] / pred_y[ss][y[ss]]) * BCE_all[ss]))
            KLD = torch.add(
                KLD, (q_y_s[ss][y[ss]] / pred_y[ss][y[ss]]) * (log_q_zs[ss] - log_p_zs[ss]).mean())

    return BCE, KLD


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
            if 'mnist' in args.dataset or 'NICO' in args.dataset:
                x, u, env, target = x.cuda(), target.cuda(), us.cuda(), u.cuda()
            else:
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda()

        # try:
        if args.model == 'VAE' or args.model == 'sVAE' or args.model == 'VAE_f' or args.model == 'sVAE_f':
            #print(u.size(), us.size())
            # print(u, us)
            _, recon_x, mu, logvar, mu_prior, logvar_prior, zs = model(
                x, u, us, feature=1)
            pred_y = model.get_pred_y(x, u, us)
        elif args.model == 'VAE_old':
            q_y_s, _, _, _, _, _ = model(x, u, us, feature=0)
            pred_y = model.get_pred_y(x, u, us)
            _, recon_x, mu, logvar, mu_prior, logvar_prior, zs = model(
                x, u, us, feature=1)
        else:
            pred_y = model(x, u, us)
        # except:
        #     continue
        if args.model == 'VAE' or args.model == 'sVAE' or args.model == 'VAE_f' or args.model == 'sVAE_f':
            if args.solve == 'IRM':
                recon_loss, kld_loss = VAE_loss(
                    recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, args)
                #cls_loss = F.nll_loss(torch.log(pred_y), target)
                cls_loss = torch.FloatTensor([0.0]).cuda()
                for ss in range(args.env_num):
                    if torch.sum(env == ss) == 0:
                        continue
                    cls_loss = torch.add(cls_loss,
                                         torch.sum(env == ss) * (F.cross_entropy(pred_y[env == ss, :], target[env == ss]) +
                                                                 args.alpha * penalty(pred_y[env == ss, :],
                                                                                      target[env == ss])))
                    #print('IRM loss:', F.cross_entropy(pred_y[env == ss, :], target[env == ss]), args.alpha * penalty(pred_y[env == ss, :],target[env == ss]))
                cls_loss = cls_loss / pred_y.size(0)
                #print(recon_loss, args.beta * kld_loss, args.gamma * cls_loss, args.beta, args.gamma)
                loss = recon_loss + args.beta * kld_loss + args.gamma * cls_loss
            else:
                recon_loss, kld_loss = VAE_loss(
                    recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, args)
                cls_loss = F.nll_loss(torch.log(pred_y), target)
                loss = recon_loss + args.beta * kld_loss + args.alpha * cls_loss

            all_zs[batch_begin:batch_begin + x.size(0), :] = \
                zs.detach().view(x.size(0), args.zs_dim).cpu().numpy()
            batch_begin = batch_begin + x.size(0)
        elif args.model == 'VAE_old':
            recon_loss, kld_loss = VAE_old_loss(pred_y, recon_x, x, mu, logvar, mu_prior,
                                                logvar_prior, zs, q_y_s, target, args)
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
        if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE'  \
                or args.model == 'VAE_f' or args.model == 'sVAE_f':
            RECON_loss.update(recon_loss.item(), x.size(0))
            KLD_loss.update(kld_loss.item(), x.size(0))
            classify_loss.update(cls_loss.item(), x.size(0))
        all_loss.update(loss.item(), x.size(0))
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(),
                        target.detach().cpu().numpy()), x.size(0))

        if batch_idx % 10 == 0:
            args.logger.info('epoch [{}/{}], batch: {}, rec_loss:{:.4f}, kld_loss:{:.4f} cls_loss:{:.4f}, overall_loss:{:.4f},acc:{:.4f}'
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


def evaluate(model, dataloader, args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    for batch_idx, (x, u, us, target) in enumerate(dataloader):
        if args.cuda:
            if 'mnist' in args.dataset or 'NICO' in args.dataset:
                x, u, us, target = x.cuda(), target.cuda(), us.cuda(), u.cuda()
            else:
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda()

        if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE'  \
                or args.model == 'VAE_f' or args.model == 'sVAE_f':
            pred_y = model.get_pred_y(x, u, us)
        else:
            pred_y = model(x, u, us)
        pred[batch_begin:batch_begin+x.size(0), :] = pred_y.detach().cpu()
        batch_begin = batch_begin + x.size(0)
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(),
                        target.detach().cpu().numpy()), x.size(0))
    return pred, accuracy.avg


def evaluate_only(model, dataloader, args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    label = np.zeros((dataloader.dataset.__len__(), ))
    batch_begin = 0
    for batch_idx, (x, u, us, target) in enumerate(dataloader):
        if args.cuda:
            if 'mnist' in args.dataset or 'NICO' in args.dataset:
                x, u, us, target = x.cuda(), target.cuda(), us.cuda(), u.cuda()
            else:
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda()

        if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' \
                or args.model == 'VAE_f' or args.model == 'sVAE_f':
            pred_y = model.get_pred_y(x, u, us)
        else:
            pred_y = model(x, u, us)
        pred[batch_begin:batch_begin +
             x.size(0), :] = pred_y.detach().cpu().numpy()
        label[batch_begin:batch_begin +
              x.size(0)] = target.detach().cpu().numpy()
        batch_begin = batch_begin + x.size(0)
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(),
                        target.detach().cpu().numpy()), x.size(0))
    return pred, label, accuracy.avg


def checkpoint(epoch, save_folder, save_model, is_best=0, other_info=None, logger=None):
    if is_best:
        model_out_tar_path = os.path.join(save_folder, "best_acc.pth.tar")
    else:
        model_out_tar_path = os.path.join(save_folder, "checkpoints.pth.tar")

    torch.save({
        'state_dict': save_model.state_dict(),
        'epoch': epoch,
        'other_info': other_info
    }, model_out_tar_path)
    if logger is not None:
        logger.info("Checkpoint saved to {}".format(model_out_tar_path))
    else:
        print("Checkpoint saved to {}".format(model_out_tar_path))


def adjust_learning_rate(optimizer, epoch, lr, lr_decay, lr_controler):
    for param_group in optimizer.param_groups:
        new_lr = lr * lr_decay ** (epoch // lr_controler)
        param_group['lr'] = lr * lr_decay ** (epoch // lr_controler)
    # if epoch % 500 == 0:
    #     print('current lr is ', new_lr)
