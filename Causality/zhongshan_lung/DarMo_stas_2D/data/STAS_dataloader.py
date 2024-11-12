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

class Dataloader_2D(Dataset):
    def __init__(self, data_index=0, fold='train', transform="None"):
        self.data_index = data_index
        # self.root = data_root
        self.fold = fold
        self.transform = transform
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
            self.samples.append(os.path.join("../../../..", worksheet.row_values(i)[0]))#pat of the samples
            self.Machine.append(int(worksheet.row_values(i)[2]))#
            self.Hospital.append(int(worksheet.row_values(i)[3]))#
            self.labels.append(int(worksheet.row_values(i)[4]))# 0/1
            self.A_1.append(worksheet.row_values(i)[9:11])#load gender and age
            self.A_2.append(worksheet.row_values(i)[11:25])#load the bi-predict in A2
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        samples = self.samples[idx]
        samples = Image.open(samples)
        if self.transform is not None:
            samples = self.transform(samples)
        labels = self.labels[idx]
        A_1 = self.A_1[idx]
        A_2 = self.A_2[idx]
        Machine = self.Machine[idx]
        Hospital = self.Hospital[idx]


        return samples,\
               torch.from_numpy(np.array(labels).astype("int")),\
               torch.from_numpy(np.array(A_1).astype("float32")),\
               torch.from_numpy(np.array(A_2).astype("int")),\
               torch.from_numpy(np.array(Machine).astype("int")),\
               torch.from_numpy(np.array(Hospital).astype("int"))

    def __len__(self):
        return len(self.samples)
    

def read_nii_file(nii_file_path):
    sitk_data = sitk.ReadImage(nii_file_path)
    return sitk.GetArrayFromImage(sitk_data), sitk_data 

class Dataloader_3D(Dataset):
    def __init__(self, data_index=0, data_root="", fold='train', transform="None"):
        self.data_index = data_index
        self.root = data_root
        self.mode = fold
        self.transform = transform
        # All Data
        workbook = xlrd.open_workbook(self.data_index)
        if self.mode == 'train':
            worksheet = workbook.sheet_by_name('train')
        else:
            worksheet = workbook.sheet_by_name('test')
        self.id = []
        self.samples = []
        self.labels = []
        self.A_1 = []
        self.A_2 = []
        self.Machine = []
        self.Hospital = []
        for i in range(2, worksheet.nrows):
            self.samples.append(os.path.join("..", worksheet.row_values(i)[0]))#pat of the samples
            self.Machine.append(int(worksheet.row_values(i)[2]))#
            self.Hospital.append(int(worksheet.row_values(i)[3]))#
            self.labels.append(int(worksheet.row_values(i)[8]))# 0/1
            self.A_1.append(int(worksheet.row_values(i)[12:14]))#load gender and age
            self.A_2.append(int(worksheet.row_values(i)[16:25]))#load the bi-predict in A2
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        samples = self.samples[idx]
        samples = Image.open(samples)
        if self.transform is not None:
            samples = self.transform(samples)
        labels = self.labels[idx]
        A_1 = self.A_1[idx]
        A_2 = self.A_2[idx]
        Machine = self.Machine[idx]
        Hospital = self.Hospital[idx]


        return samples,\
               torch.from_numpy(np.array(labels).astype("int")),\
               torch.from_numpy(np.array(A_1).astype("int")),\
               torch.from_numpy(np.array(A_2).astype("int")),\
               torch.from_numpy(np.array(Machine).astype("int")),\
               torch.from_numpy(np.array(Hospital).astype("int"))

    def __len__(self):
        return len(self.samples)