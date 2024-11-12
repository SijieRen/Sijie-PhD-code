# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from torchvision import transforms
from models import *
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys

# sijie nips zhuankan on use ADNI


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
        if self.args.if_unsp_cluster:  # sijie to use unsupervied cluster datset
            # if self.args.env_num == 2:
            print("load unsupervised cluster ADNI dataset")
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
            # if args.dataset == 'AD':
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
                elif 'AV45' in args.dataset_type:
                    sheet = workbook.sheet_by_index(12)
                elif 'ABE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(13)
                elif 'TAU' in args.dataset_type:
                    sheet = workbook.sheet_by_index(14)  # sijie zhuankan
                elif 'APOE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(15)
                    print('workbook.sheet_by_index(15)')
                elif 'EDU' in args.dataset_type:
                    sheet = workbook.sheet_by_index(8)
                elif 'AGE2' in args.dataset_type:
                    sheet = workbook.sheet_by_index(11)
                elif 'AGE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(6)  # sijie zhuankan
                elif 're' in args.dataset_type:
                    sheet = workbook.sheet_by_index(4)
                else:
                    sheet = workbook.sheet_by_index(2)
                for rows in range(1, sheet.nrows):
                    # if sheet.row_values(rows)[6] == fold_map['train']:
                    #     self.train_u.append(sheet.row_values(rows)[8:12])
                    #     if 'm' in args.dataset_type:
                    #         self.train_us.append(sheet.row_values(rows)[12:21])
                    #     else:
                    #         self.train_us.append(sheet.row_values(rows)[12:17])
                    # and int(sheet.row_values(rows)[21]) == select_env:
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
            # torch.from_numpy(np.array(self.env[index]).astype('int'))

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

# sijie nips zhuankan on use CMNIST & NICO


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
            # todo xiugai unsp cluster results dataset(ADNI CMNIST)
            if self.args.if_unsp_cluster:  # sijie to use unsupervied cluster datset
                # if self.args.env_num == 2:
                workbook = xlrd.open_workbook(
                    r"../Dataset_E/E%s_%s_NICO.xls" % (self.args.env_num, fold))
                sheet = workbook.sheet_by_index(0)
                for rows in range(1, sheet.nrows):
                    # if sheet.row_values(rows)[4] == fold_map[fold]:
                    self.image_path_list.append(sheet.row_values(rows)[0])
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
                    self.image_path_list.append(sheet.row_values(rows)[0])
                    self.y.append(sheet.row_values(rows)[1])
                    self.env.append(sheet.row_values(rows)[self.args.env_num])
                    self.u.append(float(sheet.row_values(rows)[0][-10:-6]))
                pass
                # elif self.args.env_num == 3:
                # pass
            else:  # Nips2021 dataset
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
            torch.from_numpy(np.array(self.u[index]).astype(
                'float32').reshape((1))),\
            torch.from_numpy(np.array(self.env[index]).astype('int')), \
            torch.from_numpy(np.array(self.y[index]).astype('int'))

    def __len__(self):
        return len(self.image_path_list)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class MM_F_2D_NICO(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim=256,
                 ):
        super(MM_F_2D_NICO, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 128),
            self.Conv_bn_ReLU(128, 256, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )

        self.fc = nn.Sequential(
            self.Fc_bn_ReLU(1024, 2048),
            self.Fc_bn_ReLU(2048, self.zs_dim),
            nn.Linear(self.zs_dim, num_classes))
        self.domain_fc = nn.Sequential(
            self.Fc_bn_ReLU(1024, 2048),
            self.Fc_bn_ReLU(2048, self.zs_dim),
            nn.Linear(self.zs_dim, num_classes))

    def forward(self, x, alpha):
        feature = self.get_feature(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.fc(feature)
        domain_output = self.domain_fc(reverse_feature)
        return class_output, domain_output

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class MM_F_2D_L(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim=256,
                 ):
        super(MM_F_2D_L, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

        self.fc = nn.Sequential(
            self.Fc_bn_ReLU(256, self.zs_dim),
            nn.Linear(self.zs_dim, num_classes))
        self.domain_fc = nn.Sequential(
            self.Fc_bn_ReLU(256, self.zs_dim),
            nn.Linear(self.zs_dim, num_classes))

    def forward(self, x, alpha):
        feature = self.get_feature(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.fc(feature)
        domain_output = self.domain_fc(reverse_feature)
        return class_output, domain_output

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class MM_F_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim=256,
                 ):
        super(MM_F_2D, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )
        self.fc = nn.Sequential(
            self.Fc_bn_ReLU(1024, self.zs_dim),
            nn.Linear(self.zs_dim, num_classes))
        self.domain_fc = nn.Sequential(
            self.Fc_bn_ReLU(1024, self.zs_dim),
            nn.Linear(self.zs_dim, num_classes))

    def forward(self, x, alpha):
        feature = self.get_feature(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.fc(feature)
        domain_output = self.domain_fc(reverse_feature)
        return class_output, domain_output

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class MM_F(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 ):

        super(MM_F, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool3d(1),
            Flatten()
        )
        self.fc = nn.Sequential(
            self.Fc_bn_ReLU(1024, 1024),
            nn.Linear(1024, num_classes))
        self.domain_fc = nn.Sequential(
            self.Fc_bn_ReLU(1024, 1024),
            nn.Linear(1024, num_classes))

    def forward(self, x, alpha):
        feature = self.get_feature(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.fc(feature)
        domain_output = self.domain_fc(reverse_feature)
        return class_output, domain_output

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


def train(epoch,
          model,
          optimizer,
          dataloader,
          test_loader,
          args):
    len_dataloader = len(dataloader)
    test_loader_iter = iter(test_loader)
    all_zs = np.zeros((len(dataloader.dataset), args.zs_dim))
    RECON_loss = AverageMeter()
    KLD_loss = AverageMeter()
    classify_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    i = 0
    model.train()
    for batch_idx, (x, u, us, target) in enumerate(dataloader):
        model.zero_grad()
        batch_size = x.size(0)

        p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        if args.cuda:

            # print(u,us,target)
            if 'mnist' in args.dataset or 'NICO' in args.dataset:
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda().long()
            else:
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda().long()

        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long().cuda()
        if batch_size == 1:
            break
        class_output, domain_output = model(x, alpha=alpha)
        # print(target, domain_label, class_output.size(), domain_output.size())
        err_s_label = F.cross_entropy(class_output, target)
        err_s_domain = F.cross_entropy(domain_output, domain_label)

        try:
            t_img, _, _, _ = test_loader_iter.next()
        except:
            break
        t_img = t_img.cuda()
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long().cuda()
        if t_img.size(0) != batch_size:
            break
        _, domain_output = model(t_img, alpha=alpha)
        err_t_domain = F.cross_entropy(domain_output, domain_label)
        loss = err_t_domain + err_s_domain + err_s_label

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss.update(loss.item(), x.size(0))
        accuracy.update(compute_acc(class_output.detach().cpu(
        ).numpy(), target.detach().cpu().numpy()), x.size(0))

        if batch_idx % 10 == 0:
            print(
                'epoch [{}/{}], batch: {}, rec_loss:{:.4f}, kld_loss:{:.4f} cls_loss:{:.4f}, overall_loss:{:.4f},acc:{:.4f}'
                .format(epoch,
                        args.epochs,
                        batch_idx,
                        RECON_loss.avg,
                        KLD_loss.avg * args.beta,
                        classify_loss.avg * args.alpha,
                        all_loss.avg,
                        accuracy.avg * 100))

        i = i + 1
    if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE':
        all_zs = all_zs[:batch_begin]
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch,
          args.epochs, all_loss.avg))

    return all_zs, accuracy.avg


def evaluate(model, dataloader, args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    for batch_idx, (x, u, us, target) in enumerate(dataloader):
        if args.cuda:
            if 'mnist' in args.dataset or 'NICO' in args.dataset:
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda().long()
            else:
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda().long()

        pred_y, _ = model(x, alpha=0)
        pred[batch_begin:batch_begin + x.size(0), :] = pred_y.detach().cpu()
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
    for batch_idx, (x, target, u, us) in enumerate(dataloader):
        if args.cuda:
            x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda()

        pred_y, _ = model(x, alpha=0)
        pred[batch_begin:batch_begin +
             x.size(0), :] = pred_y.detach().cpu().numpy()
        label[batch_begin:batch_begin +
              x.size(0), ] = target.detach().cpu().numpy()

        batch_begin = batch_begin + x.size(0)
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(),
                        target.detach().cpu().numpy()), x.size(0))
    return pred, label, accuracy.avg


def main():
    args = get_opt()
    args = make_dirs(args)
    logger = get_logger(args)
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    if args.dataset == 'AD':
        train_loader = DataLoader(get_dataset(args=args, fold='train', aug=args.aug),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)
        test_loader = DataLoader(get_dataset(args=args, fold='test', aug=0),
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 pin_memory=True)
        val_loader = None
    elif 'mnist' in args.dataset:
        train_loader = DataLoader(get_dataset_2D_env(root=args.root, args=args, fold='train',
                                                     transform=transforms.Compose([
                                                         transforms.RandomHorizontalFlip(
                                                             p=0.5),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                     ])),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True,
                                  drop_last=True)
        test_loader = DataLoader(get_dataset_2D_env(root=args.root, args=args, fold='test',
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                    ])),
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 pin_memory=True)
        val_loader = None
    else:
        train_loader = DataLoader(get_dataset_2D_env(args=args, fold='train',
                                                     transform=transforms.Compose([
                                                         transforms.RandomResizedCrop(
                                                             (256, 256)),
                                                         transforms.RandomHorizontalFlip(
                                                             p=0.5),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                     ])),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)
        test_loader = DataLoader(get_dataset_2D_env(args=args, fold='test',
                                                    transform=transforms.Compose([
                                                        transforms.Resize(
                                                            (256, 256)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                    ])),
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 pin_memory=True)
        val_loader = None
        # val_loader = DataLoader(get_dataset_2D_env(args=args, fold='val',
        #                                         transform=transforms.Compose([
        #                                             transforms.Resize((256, 256)),
        #                                             transforms.ToTensor(),
        #                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #
        #                                         ])),
        #                          batch_size=args.test_batch_size,
        #                          shuffle=False,
        #                          num_workers=1,
        #                          pin_memory=True)
    if args.dataset == 'AD':
        model = MM_F(in_channel=args.in_channel,
                     u_dim=args.u_dim,
                     us_dim=args.us_dim,
                     num_classes=args.num_classes,
                     is_use_u=args.is_use_u,
                     ).cuda()
    elif 'NICO' in args.dataset:
        model = MM_F_2D_NICO(in_channel=args.in_channel,
                             u_dim=args.u_dim,
                             us_dim=args.us_dim,
                             num_classes=args.num_classes,
                             is_use_u=args.is_use_u,
                             zs_dim=args.zs_dim
                             ).cuda()
    else:
        model = MM_F_2D_L(in_channel=args.in_channel,
                          u_dim=args.u_dim,
                          us_dim=args.us_dim,
                          num_classes=args.num_classes,
                          is_use_u=args.is_use_u,
                          ).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params,
          '%0.4f M' % (pytorch_total_params / 1e6))

    # if args.inference_only and args.load_path != '':
    #     model.load_state_dict(torch.load(os.path.join(
    #         args.load_path, 'best_acc.pth.tar'))['state_dict'])
    #     test_loader_NICO = DataLoaderX(get_dataset_NICO_inter(args=args,
    #                                                           transform=transforms.Compose([
    #                                                               transforms.Resize(
    #                                                                   (256, 256)),
    #                                                               transforms.ToTensor(),
    #                                                               transforms.Normalize((0.5, 0.5, 0.5),
    #                                                                                    (0.5, 0.5, 0.5)),

    #                                                           ])),
    #                                    batch_size=args.test_batch_size,
    #                                    shuffle=False,
    #                                    num_workers=1,
    #                                    pin_memory=True)
    #     pred_test, label_test, test_acc = evaluate_only(
    #         model, test_loader_NICO, args)
    #     save_pred_label_as_xlsx(
    #         args.model_save_dir, 'pred.xls', pred_test, label_test, test_loader_NICO, args)
    #     logger.info('model save path: %s' % args.model_save_dir)
    #     print('test_acc', test_acc)
    #     exit(123)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.reg)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params,
          '%0.4f M' % (pytorch_total_params / 1e6))

    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr,
                             args.lr_decay, args.lr_controler)
        _, _ = train(epoch, model, optimizer, train_loader, test_loader, args)
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
        checkpoint(epoch, args.model_save_dir, model,
                   is_best, other_info, logger)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d, val_acc: %0.4f'
                    % (epoch, test_acc, best_acc, best_acc_ep, val_acc))
    logger.info('model save path: %s' % args.model_save_dir)
    xlsx_name = '%s_DANN_u_%d_fold_%d_env_%d.xls' % \
                (os.path.basename(sys.argv[0][:-3]), args.is_use_u,
                 args.fold, args.env_num)
    save_results_as_xlsx('./results/', xlsx_name, best_acc,
                         best_acc_ep, auc=None, args=args)
    logger.info('*' * 50)
    logger.info('*' * 50)


if __name__ == '__main__':
    main()
