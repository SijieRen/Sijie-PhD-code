from ast import Str
import os
from typing import Dict
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


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def load_img_path(filepath):
    img = imageio.imread(filepath)
    img_shape = img.shape[1]
    img_3d = img.reshape((img_shape, img_shape, img_shape))
    img_3d = img_3d.transpose((1, 2, 0))
    return img_3d


"""
AD

Returns:
    x, u, us, y, env
"""


class get_dataset(data.Dataset):
    def __init__(self,
                 root=None,
                 fold='train',
                 transform=None,
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
        self.transform = transform
        fold_map = {'train': 0, 'val': 1, 'test': 2}
        Label_map = {'AD': 2, 'MCI': 1, 'NC': 0}
        if args.dataset == 'AD':
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
                if args.fold == 1:  # sijie use this dataset
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
                elif 'TAU' in args.dataset_type:
                    sheet = workbook.sheet_by_index(14)  # sijie zhuankan
                elif 'AGE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(6)  # sijie zhuankan

                for rows in range(1, sheet.nrows):
                    if fold == "traintest":
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
                        if sheet.row_values(rows)[6] == fold_map[fold]:
                            self.image_path_list.append(os.path.join(
                                self.root, sheet.row_values(rows)[7]))
                            self.y.append(Label_map[sheet.row_values(rows)[2]])
                            self.u.append(sheet.row_values(rows)[8:12])
                            if 'm' in args.dataset_type:
                                self.us.append(sheet.row_values(rows)[12:21])
                            else:
                                self.us.append(sheet.row_values(rows)[12:17])
                            self.env.append(int(sheet.row_values(rows)[21]))
                print(len(self.image_path_list))
        self.u = np.array(self.u).astype('float32')
        self.us = np.array(self.us).astype('float32')
        self.train_u = np.array(self.train_u).astype('float32')
        self.train_us = np.array(self.train_us).astype('float32')

    def __getitem__(self, index):
        x = load_img_path(self.image_path_list[index]) / 255.0
        # print(x)
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

        if self.transform is not None:
            x = self.transform(x)
            # torch.from_numpy(x.astype('float32')), \
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


"""
CMNIST / NICO

Returns:
    x, y, env, U
"""


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
            elif args.dataset_type == 'NICO_10':
                sheet = workbook.sheet_by_index(10)
            for rows in range(1, sheet.nrows):
                if fold == "traintest":
                    self.image_path_list.append(os.path.join(
                        self.root, sheet.row_values(rows)[0]))
                    self.y.append(sheet.row_values(rows)[1])
                    self.env.append(sheet.row_values(rows)[3])
                    self.u.append(sheet.row_values(rows)[5:5+4])

                else:
                    if sheet.row_values(rows)[4] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(
                            self.root, sheet.row_values(rows)[0]))
                        self.y.append(sheet.row_values(rows)[1])
                        self.env.append(sheet.row_values(rows)[3])
                        self.u.append(sheet.row_values(rows)[5:5+4])

        elif args.dataset == 'mnist_2':
            if fold == "traintest":
                fold_list = ["train", "test"]
                for fold__ in fold_list:
                    if self.root is None:
                        root__ = '../data/colored_MNIST_0.02_env_2_0_c_2/%s/' % fold__
                    else:
                        root__ = self.root + '%s/' % fold__

                    all_classes = os.listdir(root__)
                    for one_class in all_classes:
                        for filename in os.listdir(os.path.join(root__, one_class)):
                            self.u.append(float(filename[-10:-6]))
                            self.env.append(int(filename[-5:-4]))
                            self.image_path_list.append(
                                os.path.join(root__, one_class, filename))
                            if int(one_class) <= 4:
                                self.y.append(0)
                            else:
                                self.y.append(1)
                    print(self.root)

            else:
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
            # torch.from_numpy(np.array(self.u[index]).astype(
        #     'float32').reshape((len(self.u[index]))))

    def __len__(self):
        return len(self.image_path_list)


def save_results_as_xlsx(root: str, xlsx_name: str, res_dict: Dict, args):
    """_summary_

    Args:
        root (str): _description_
        xlsx_name (str): _description_
        res_dict (Dict): _description_
                {'total_loss': total_loss, 'rec_loss': recon_loss,
                'gauss_loss': gauss_loss, 'cat_loss': cat_loss,
                'predict_label': predicted_labels.tolist(), 'image_path': image_path_list,
                'true_label':true_labels.tolist()}
        args (_type_): _description_
    """
    if not os.path.exists(os.path.join(root, xlsx_name)):
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet('Sheet1')
        col_idx = args.num_classes
        worksheet.write(0, 0, 'image_path')
        worksheet.write(0, 1, 'label')
        worksheet.write(0, col_idx, 'cat_num={}'.format(args.num_classes))
        row_idx = 1
        # col_idx = 1

        print("write in {}th column".format(col_idx))
        idx = 1
        for ii in range(len(res_dict['predict_label'])):
            worksheet.write(row_idx, 0, res_dict['image_path'][ii])
            worksheet.write(row_idx, 1, res_dict['true_label'][ii])
            worksheet.write(row_idx, col_idx, res_dict['predict_label'][ii])
            row_idx += 1

        workbook.save(os.path.join(root, xlsx_name))
    else:
        rb = xlrd.open_workbook(os.path.join(root, xlsx_name))
        # col_idx = rb.sheets()[0].ncols
        col_idx = args.num_classes

        wb = x_copy(rb)
        worksheet = wb.get_sheet(0)
        row_idx = 1
        # col_idx = len(worksheet.get_cols())
        # nrows = worksheet.nrows

        # print("write in {}th row".format(nrows))
        print("write in {}th column".format(col_idx))
        worksheet.write(0, col_idx, 'cat_num={}'.format(args.num_classes))
        for ii in range(len(res_dict['predict_label'])):
            # worksheet.write(row_idx, 0, res_dict['image_path'][ii])
            worksheet.write(row_idx, col_idx, res_dict['predict_label'][ii])
            row_idx += 1

        wb.save(os.path.join(root, xlsx_name))
