# coding=utf-8
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import random
import xlrd
from torchvision import transforms
import imageio


def load_img_path(filepath):
    # img = imageio.imread(filepath)
    # img_shape = img.shape[1]
    # img_3d = img.reshape((img_shape, img_shape, img_shape))

    img_3d = np.load(filepath)
    img_3d = img_3d.transpose((1, 2, 0))
    return img_3d

class Dataloader_3D(Dataset):
    def __init__(self, data_index=0, fold='train', transform="None", args=None):
        self.data_index = data_index
        # self.root = data_root
        self.fold = fold
        self.transform = transform
        self.args = args
        # All Data
        workbook = xlrd.open_workbook(self.data_index)
        
        worksheet = workbook.sheet_by_name(self.fold)
        self.id = []
        self.samples = []
        self.labels = []
        self.A_1 = []
        self.A_2 = []
        self.Machine = []
        self.Hospital = []
        for i in range(2, worksheet.nrows):
            # self.samples.append(os.path.join("../../../..", worksheet.row_values(i)[0]))#pat of the samples
            self.samples.append(worksheet.row_values(i)[0])#pat of the samples
            # print(worksheet.row_values(i),"[2]", worksheet.row_values(i)[2])
            self.Machine.append(int(worksheet.row_values(i)[2]))#
            self.Hospital.append(int(worksheet.row_values(i)[3]))#
            self.labels.append(int(worksheet.row_values(i)[4]))# 0/1
            self.A_1.append(worksheet.row_values(i)[9:11])#load gender and age
            self.A_2.append(worksheet.row_values(i)[11:25])#load the bi-predict in A2
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        samples = load_img_path(self.samples[idx]) / 255.0
        # print(samples)
        # TODO to print the value load from numpy, if devided by 255.0
        from scipy.interpolate import RegularGridInterpolator
        original_data = samples
        # 创建一个插值函数
        interp_func = RegularGridInterpolator((np.arange(original_data.shape[0]), 
                                               np.arange(original_data.shape[1]), 
                                               np.arange(original_data.shape[2])), original_data)

        # 创建新的坐标点
        new_coords = np.array(np.meshgrid(np.linspace(0, original_data.shape[0]-1, self.args.crop_size), 
                                          np.linspace(0, original_data.shape[1]-1, self.args.crop_size), 
                                          np.linspace(0, original_data.shape[2]-1, self.args.crop_size))).T.reshape(-1, 3)
        # 使用插值函数获取插值结果
        interpolated_data = interp_func(new_coords)
        # 将插值结果重新排列为32x32x32的数组
        interpolated_data = interpolated_data.reshape(self.args.crop_size, 
                                                      self.args.crop_size, 
                                                      self.args.crop_size)

        samples = interpolated_data
        if self.transform:
            x = self._img_aug(samples).reshape(self.args.in_channel,
                                         self.args.crop_size,
                                         self.args.crop_size,
                                         self.args.crop_size, )
        else:
            crop_x = int(samples.shape[0] / 2 - self.args.crop_size / 2)
            crop_y = int(samples.shape[1] / 2 - self.args.crop_size / 2)
            crop_z = int(samples.shape[2] / 2 - self.args.crop_size / 2)
            x = samples[crop_x: crop_x + self.args.crop_size, crop_y: crop_y + self.args.crop_size,
                  crop_z: crop_z + self.args.crop_size]
            x = x.reshape(self.args.in_channel,
                          self.args.crop_size,
                          self.args.crop_size,
                          self.args.crop_size, )

        # samples = self.samples[idx]
        # samples = Image.open(samples)
        # if self.transform is not None:
        #     samples = self.transform(samples)
        labels = self.labels[idx]
        A_1 = self.A_1[idx]
        A_2 = self.A_2[idx]
        Machine = self.Machine[idx]
        Hospital = self.Hospital[idx]


        return torch.from_numpy(x.astype("float32")),\
               torch.from_numpy(np.array(labels).astype("int")),\
               torch.from_numpy(np.array(A_1).astype("float32")),\
               torch.from_numpy(np.array(A_2).astype("int")),\
               torch.from_numpy(np.array(Machine).astype("int")),\
               torch.from_numpy(np.array(Hospital).astype("int"))

    def __len__(self):
        return len(self.samples)
    
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
    

def read_nii_file(nii_file_path):
    sitk_data = sitk.ReadImage(nii_file_path)
    return sitk.GetArrayFromImage(sitk_data), sitk_data 

# class Dataloader_3D(Dataset):
#     def __init__(self, data_index=0, data_root="", fold='train', transform="None"):
#         self.data_index = data_index
#         self.root = data_root
#         self.mode = fold
#         self.transform = transform
#         # All Data
#         workbook = xlrd.open_workbook(self.data_index)
#         if self.mode == 'train':
#             worksheet = workbook.sheet_by_name('train')
#         else:
#             worksheet = workbook.sheet_by_name('test')
#         self.id = []
#         self.samples = []
#         self.labels = []
#         self.A_1 = []
#         self.A_2 = []
#         self.Machine = []
#         self.Hospital = []
#         for i in range(2, worksheet.nrows):
#             self.samples.append(os.path.join("..", worksheet.row_values(i)[0]))#pat of the samples
#             self.Machine.append(int(worksheet.row_values(i)[2]))#
#             self.Hospital.append(int(worksheet.row_values(i)[3]))#
#             self.labels.append(int(worksheet.row_values(i)[8]))# 0/1
#             self.A_1.append(int(worksheet.row_values(i)[12:14]))#load gender and age
#             self.A_2.append(int(worksheet.row_values(i)[16:25]))#load the bi-predict in A2
    
#     def get_labels(self):
#         return self.labels

#     def __getitem__(self, idx):
#         samples = self.samples[idx]
#         samples = Image.open(samples)
#         if self.transform is not None:
#             samples = self.transform(samples)
#         labels = self.labels[idx]
#         A_1 = self.A_1[idx]
#         A_2 = self.A_2[idx]
#         Machine = self.Machine[idx]
#         Hospital = self.Hospital[idx]


#         return samples,\
#                torch.from_numpy(np.array(labels).astype("int")),\
#                torch.from_numpy(np.array(A_1).astype("int")),\
#                torch.from_numpy(np.array(A_2).astype("int")),\
#                torch.from_numpy(np.array(Machine).astype("int")),\
#                torch.from_numpy(np.array(Hospital).astype("int"))

#     def __len__(self):
#         return len(self.samples)