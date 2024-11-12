import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from torchvision import transforms
from PIL import Image
import xlrd
import numpy as np
import os


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class ppa_dataloader_order1_Extractor(data.Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 fold='train',
                 eye=0,
                 center=0,
                 feature_list=[8, 9],
                 ):
        super(ppa_dataloader_order1_Extractor, self).__init__()
        self.root = root
        self.transform = transform
        self.eye = eye
        self.center = center
        self.image_path_all_1 = []
        self.image_path_all_2 = []
        self.ppa_t1 = []
        self.target_ppa = []
        self.feature_list = feature_list
        self.feature_all_1 = []
        self.feature_all_2 = []
        self.base_grade_num = []
        self.base_grade_num_2 = []
        self.feature_mask_1 = np.zeros(74, ).astype('bool')
        self.feature_mask_2 = np.zeros(74, ).astype('bool')
        for ids in feature_list:
            self.feature_mask_1[ids] = True
            self.feature_mask_2[ids + 32] = True

        print('load std data')
        workbook1 = xlrd.open_workbook(
            "../ppa-classi-dataset-onuse/ppa_2021-12-06-order1-std-8-withAllLabel.xls")
        sheet1 = workbook1.sheet_by_index(0)

        for rows in range(1, sheet1.nrows):
            if sheet1.row_values(rows)[5] == fold:
                if sheet1.row_values(rows)[3] in self.eye:
                    if str(sheet1.row_values(rows)[4]) in self.center:
                        self.image_path_all_1.append(os.path.join(
                            self.root, sheet1.row_values(rows)[1]))
                        self.image_path_all_2.append(os.path.join(
                            self.root, sheet1.row_values(rows)[2]))
                        self.ppa_t1.append(
                            sheet1.row_values(rows)[72])  # ppa_t1
                        self.target_ppa.append(
                            sheet1.row_values(rows)[73])  # ppa_target
                        self.feature_all_1.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))
                        self.feature_all_2.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_2].astype('float32'))
                        self.base_grade_num.append(
                            int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                        self.base_grade_num_2.append(
                            int(sheet1.row_values(rows)[2].split('/')[0][-1]))

    def __getitem__(self, index):

        img_path_1, img_path_2 = self.image_path_all_1[index], self.image_path_all_2[index]
        target_ppa = self.target_ppa[index]
        ppa_t1 = self.ppa_t1[index]

        img_1 = Image.open(img_path_1)
        img_2 = Image.open(img_path_2)
        base_target = [-1, -1]
        if (target_ppa == 0 or target_ppa == 1) and self.base_grade_num[index] == 1:
            base_target[0] = 0

        if (target_ppa == 0 or target_ppa == 1) and self.base_grade_num_2[index] == 6:
            base_target[1] = target_ppa

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, \
            img_2, \
            torch.from_numpy(np.array(ppa_t1).astype('int')), \
            torch.from_numpy(np.array(target_ppa).astype('int')), \
            torch.from_numpy(np.array(self.feature_all_1[index]).astype('float32')), \
            torch.from_numpy(np.array(self.feature_all_2[index]).astype('float32')), \
            torch.from_numpy(np.array(self.base_grade_num[index]).astype('int')), \
            torch.from_numpy(
                np.array(self.base_grade_num_2[index]).astype('int'))

    def __len__(self):
        return len(self.image_path_all_1)


class ppa_dataloader_order1(data.Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 fold='train',
                 eye=0,
                 center=0,
                 filename='std',
                 feature_list=[8, 9],
                 ):
        super(ppa_dataloader_order1, self).__init__()
        self.root = root
        self.transform = transform
        self.eye = eye
        self.center = center
        self.image_path_all_1 = []
        self.image_path_all_2 = []
        self.ppa_t1 = []
        self.target_ppa = []
        self.feature_list = feature_list
        self.feature_all_1 = []
        self.feature_all_2 = []
        self.base_grade_num = []
        self.base_grade_num_2 = []
        self.feature_mask_1 = np.zeros(74, ).astype('bool')
        self.feature_mask_2 = np.zeros(74, ).astype('bool')
        for ids in feature_list:
            self.feature_mask_1[ids] = True
            self.feature_mask_2[ids + 32] = True

        if filename == 'std_all':
            print('load data std all pair')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2021-12-06-order1-std-8-withAllLabel.xls")

        elif filename == "rulebase":
            print('load data std all pair')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2021-12-06-order1-std-8-withAllLabel-rulebasedValTest1.xls")

        elif filename == "baoliu1to6":
            print('load data baoliu1to6 pair')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2022-0222-order1-std-8-baoliu1to6AllLabel.xls")

        # print('load std data')
        # workbook1 = xlrd.open_workbook(
        #     "../ppa-classi-dataset-onuse/ppa_2021-12-06-order1-std-8-withAllLabel-rulebasedValTest1.xls")
        sheet1 = workbook1.sheet_by_index(0)

        for rows in range(1, sheet1.nrows):
            if sheet1.row_values(rows)[5] == fold:
                if sheet1.row_values(rows)[3] in self.eye:
                    if str(sheet1.row_values(rows)[4]) in self.center:
                        self.image_path_all_1.append(os.path.join(
                            self.root, sheet1.row_values(rows)[1]))
                        self.image_path_all_2.append(os.path.join(
                            self.root, sheet1.row_values(rows)[2]))
                        self.ppa_t1.append(
                            sheet1.row_values(rows)[72])  # ppa_t1
                        self.target_ppa.append(
                            sheet1.row_values(rows)[73])  # ppa_target
                        self.feature_all_1.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))
                        self.feature_all_2.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_2].astype('float32'))
                        self.base_grade_num.append(
                            int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                        self.base_grade_num_2.append(
                            int(sheet1.row_values(rows)[2].split('/')[0][-1]))

    def __getitem__(self, index):

        img_path_1, img_path_2 = self.image_path_all_1[index], self.image_path_all_2[index]
        target_ppa = self.target_ppa[index]
        ppa_t1 = self.ppa_t1[index]

        img_1 = Image.open(img_path_1)
        img_2 = Image.open(img_path_2)
        base_target = [-1, -1]
        if (target_ppa == 0 or target_ppa == 1) and self.base_grade_num[index] == 1:
            base_target[0] = 0

        if (target_ppa == 0 or target_ppa == 1) and self.base_grade_num_2[index] == 6:
            base_target[1] = target_ppa

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, \
            img_2, \
            torch.from_numpy(np.array(ppa_t1).astype('int')), \
            torch.from_numpy(np.array(target_ppa).astype('int')), \
            torch.from_numpy(np.array(self.feature_all_1[index]).astype('float32')), \
            torch.from_numpy(np.array(self.feature_all_2[index]).astype('float32')), \
            torch.from_numpy(np.array(self.base_grade_num[index]).astype('int')), \
            torch.from_numpy(
                np.array(self.base_grade_num_2[index]).astype('int'))

    def __len__(self):
        return len(self.image_path_all_1)


class ppa_dataloader_order2(data.Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 fold='train',
                 eye=0,
                 center=0,
                 label=0,
                 filename='std',
                 feature_list=[7, 8, 9],
                 order=2):
        super(ppa_dataloader_order2, self).__init__()
        self.root = root  # excel path
        self.transform = transform
        self.eye = eye
        self.center = center
        self.label = label
        self.image_path_all_1 = []
        self.image_path_all_2 = []
        self.image_path_all_3 = []
        self.ppa_t1 = []
        self.ppa_t2 = []
        self.target_ppa = []
        self.feature_list = feature_list
        self.feature_all_1 = []
        self.feature_all_2 = []
        self.feature_all_3 = []

        self.base_grade_num = []
        self.base_grade_num_2 = []
        self.base_grade_num_3 = []

        self.feature_mask_1 = np.zeros(107, ).astype('bool')
        self.feature_mask_2 = np.zeros(107, ).astype('bool')
        self.feature_mask_3 = np.zeros(107, ).astype('bool')
        for ids in feature_list:
            self.feature_mask_1[ids] = True
            self.feature_mask_2[ids + 32] = True
            self.feature_mask_3[ids + 64] = True

        if filename == 'std_all':
            print('load data std all pair')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2021-12-06-order2-std-8-withAllLabel.xls")

        elif filename == "rulebase":
            print('load data std all pair')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2021-12-06-order2-std-8-withAllLabel-rulebasedValTest1.xls")

        elif filename == "baoliu1to6":
            print('load data baoliu1to6 pair')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2022-0222-order2-std-8-baoliu1to6AllLabel.xls")
        elif filename == "baoliu1to6_15pairs":
            print('load data baoliu1to6 pair-15pairs')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2022-0222-order2-std-8-baoliu1to6AllLabel-2.xls")
        # print('load data std all pair')
        # workbook1 = xlrd.open_workbook(
        #     r"../ppa-classi-dataset-onuse/ppa_2021-12-06-order2-std-8-withAllLabel-rulebasedValTest1.xls")
        sheet1 = workbook1.sheet_by_index(0)

        for rows in range(1, sheet1.nrows):
            if sheet1.row_values(rows)[6] == fold:
                if sheet1.row_values(rows)[4] in self.eye:
                    if str(sheet1.row_values(rows)[5]) in self.center:
                        self.image_path_all_1.append(os.path.join(
                            self.root, sheet1.row_values(rows)[1]))
                        self.image_path_all_2.append(os.path.join(
                            self.root, sheet1.row_values(rows)[2]))
                        self.image_path_all_3.append(os.path.join(
                            self.root, sheet1.row_values(rows)[3]))
                        self.ppa_t1.append(sheet1.row_values(rows)[104])
                        self.ppa_t2.append(sheet1.row_values(rows)[105])
                        self.target_ppa.append(sheet1.row_values(rows)[106])

                        self.feature_all_1.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))
                        self.feature_all_2.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_2].astype('float32'))
                        self.feature_all_3.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_3].astype('float32'))

                        self.base_grade_num.append(
                            int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                        self.base_grade_num_2.append(
                            int(sheet1.row_values(rows)[2].split('/')[0][-1]))
                        self.base_grade_num_3.append(
                            int(sheet1.row_values(rows)[3].split('/')[0][-1]))

    def __getitem__(self, index):

        img_path_1, img_path_2 = self.image_path_all_1[index], self.image_path_all_2[index]
        img_path_3 = self.image_path_all_3[index]
        ppa_t1 = self.ppa_t1[index]
        ppa_t2 = self.ppa_t2[index]
        target_ppa = self.target_ppa[index]

        img_1 = Image.open(img_path_1)
        img_2 = Image.open(img_path_2)
        img_3 = Image.open(img_path_3)

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)

        return img_1, \
            img_2, \
            img_3, \
            torch.from_numpy(np.array(ppa_t1).astype('int')), \
            torch.from_numpy(np.array(ppa_t2).astype('int')), \
            torch.from_numpy(np.array(target_ppa).astype('int')), \
            torch.from_numpy(np.array(self.feature_all_1[index]).astype('float32')), \
            torch.from_numpy(np.array(self.feature_all_2[index]).astype('float32')), \
            torch.from_numpy(np.array(self.feature_all_3[index]).astype('float32')), \
            torch.from_numpy(np.array(self.base_grade_num[index]).astype('int')), \
            torch.from_numpy(np.array(self.base_grade_num_2[index]).astype('int')), \
            torch.from_numpy(
                np.array(self.base_grade_num_3[index]).astype('int'))

    def __len__(self):
        return len(self.image_path_all_1)


class ppa_dataloader_order3(data.Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 fold='train',
                 eye=0,
                 center=0,
                 label=0,
                 filename='std',
                 feature_list=[7, 8, 9],
                 order=2):
        super(ppa_dataloader_order3, self).__init__()
        self.root = root  # excel path
        self.transform = transform
        self.eye = eye
        self.center = center
        self.label = label
        self.image_path_all_1 = []
        self.image_path_all_2 = []
        self.image_path_all_3 = []
        self.image_path_all_4 = []
        self.ppa_t1 = []
        self.ppa_t2 = []
        self.ppa_t3 = []
        self.target_ppa = []
        self.feature_list = feature_list
        self.feature_all_1 = []
        self.feature_all_2 = []
        self.feature_all_3 = []
        self.feature_all_4 = []

        self.base_grade_num = []
        self.base_grade_num_2 = []
        self.base_grade_num_3 = []
        self.base_grade_num_4 = []

        self.feature_mask_1 = np.zeros(141, ).astype('bool')
        self.feature_mask_2 = np.zeros(141, ).astype('bool')
        self.feature_mask_3 = np.zeros(141, ).astype('bool')
        self.feature_mask_4 = np.zeros(141, ).astype('bool')
        for ids in feature_list:
            self.feature_mask_1[ids + 1] = True
            self.feature_mask_2[ids + 1 + 32] = True
            self.feature_mask_3[ids + 1 + 64] = True
            self.feature_mask_4[ids + 1 + 96] = True

        if filename == 'std_all':
            print('load data std all pair')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2021-12-06-order3-std-8-withAllLabel.xls")

        elif filename == "rulebase":
            print('load data std all pair')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2021-12-06-order3-std-8-withAllLabel-rulebasedValTest1.xls")

        elif filename == "baoliu1to6":
            print('load data baoliu1to6 pair')
            workbook1 = xlrd.open_workbook(
                r"../ppa-classi-dataset-onuse/ppa_2022-0222-order3-std-8-baoliu1to6AllLabel.xls")

        sheet1 = workbook1.sheet_by_index(0)

        for rows in range(1, sheet1.nrows):
            if sheet1.row_values(rows)[7] == fold:
                if sheet1.row_values(rows)[5] in self.eye:
                    if str(sheet1.row_values(rows)[6]) in self.center:
                        self.image_path_all_1.append(os.path.join(
                            self.root, sheet1.row_values(rows)[1]))
                        self.image_path_all_2.append(os.path.join(
                            self.root, sheet1.row_values(rows)[2]))
                        self.image_path_all_3.append(os.path.join(
                            self.root, sheet1.row_values(rows)[3]))
                        self.image_path_all_4.append(os.path.join(
                            self.root, sheet1.row_values(rows)[4]))
                        self.ppa_t1.append(sheet1.row_values(rows)[137])
                        self.ppa_t2.append(sheet1.row_values(rows)[138])
                        self.ppa_t3.append(sheet1.row_values(rows)[139])
                        self.target_ppa.append(sheet1.row_values(rows)[140])

                        self.feature_all_1.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_1].astype('float32'))
                        self.feature_all_2.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_2].astype('float32'))
                        self.feature_all_3.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_3].astype('float32'))
                        self.feature_all_4.append(
                            np.array(sheet1.row_values(rows))[self.feature_mask_4].astype('float32'))

                        self.base_grade_num.append(
                            int(sheet1.row_values(rows)[1].split('/')[0][-1]))
                        self.base_grade_num_2.append(
                            int(sheet1.row_values(rows)[2].split('/')[0][-1]))
                        self.base_grade_num_3.append(
                            int(sheet1.row_values(rows)[3].split('/')[0][-1]))
                        self.base_grade_num_4.append(
                            int(sheet1.row_values(rows)[4].split('/')[0][-1]))

    def __getitem__(self, index):

        img_path_1, img_path_2 = self.image_path_all_1[index], self.image_path_all_2[index]
        img_path_3, img_path_4 = self.image_path_all_3[index], self.image_path_all_4[index]
        ppa_t1 = self.ppa_t1[index]
        ppa_t2 = self.ppa_t2[index]
        ppa_t3 = self.ppa_t3[index]
        target_ppa = self.target_ppa[index]

        img_1 = Image.open(img_path_1)
        img_2 = Image.open(img_path_2)
        img_3 = Image.open(img_path_3)
        img_4 = Image.open(img_path_4)

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)
            img_4 = self.transform(img_4)

        return img_1, \
            img_2, \
            img_3, \
            img_4, \
            torch.from_numpy(np.array(ppa_t1).astype('int')), \
            torch.from_numpy(np.array(ppa_t2).astype('int')), \
            torch.from_numpy(np.array(ppa_t3).astype('int')), \
            torch.from_numpy(np.array(target_ppa).astype('int')), \
            torch.from_numpy(np.array(self.feature_all_1[index]).astype('float32')), \
            torch.from_numpy(np.array(self.feature_all_2[index]).astype('float32')), \
            torch.from_numpy(np.array(self.feature_all_3[index]).astype('float32')), \
            torch.from_numpy(np.array(self.feature_all_4[index]).astype('float32')), \
            torch.from_numpy(np.array(self.base_grade_num[index]).astype('int')), \
            torch.from_numpy(np.array(self.base_grade_num_2[index]).astype('int')), \
            torch.from_numpy(np.array(self.base_grade_num_3[index]).astype('int')), \
            torch.from_numpy(
                np.array(self.base_grade_num_4[index]).astype('int'))

    def __len__(self):
        return len(self.image_path_all_1)


def get_all_dataloader_order1_Extractor(args):
    test_loader_list = []
    val_loader_list = []
    kwargs = {'num_workers': args.works, 'pin_memory': True}
    train_loader = DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='train', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.RandomRotation(30),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='test', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='val', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='test2', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='val2', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='test3', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='val3', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='test4', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='val4', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='test5', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order1_Extractor(args.data_root, fold='val5', eye=args.eye, center=args.center, feature_list=args.feature_list,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    return train_loader, val_loader_list, test_loader_list


def get_all_dataloader_order1(args):
    test_loader_list = []
    val_loader_list = []
    kwargs = {'num_workers': args.works, 'pin_memory': True}
    train_loader = DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='train', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.RandomRotation(30),
                                  transforms.ToTensor(),
                                  #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='test', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='val', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='test2', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #   transforms.Normalize(
                                  #       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='val2', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #   transforms.Normalize(
                                  #       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='test3', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #   transforms.Normalize(
                                  #       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='val3', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #   transforms.Normalize(
                                  #       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='test4', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #   transforms.Normalize(
                                  #       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='val4', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #   transforms.Normalize(
                                  #       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='test5', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #   transforms.Normalize(
                                  #       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order1(args.data_root, fold='val5', eye=args.eye, center=args.center,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  #   transforms.Normalize(
                                  #       (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    return train_loader, val_loader_list, test_loader_list


def get_all_dataloader_order2(args):
    test_loader_list = []
    val_loader_list = []
    kwargs = {'num_workers': args.works, 'pin_memory': True}
    train_loader = DataLoaderX(
        ppa_dataloader_order2(args.data_root, fold='train', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.RandomRotation(30),
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order2(args.data_root, fold='test', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order2(args.data_root, fold='val', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order2(args.data_root, fold='test2', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order2(args.data_root, fold='val2', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order2(args.data_root, fold='test3', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order2(args.data_root, fold='val3', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order2(args.data_root, fold='test4', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order2(args.data_root, fold='val4', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    return train_loader, val_loader_list, test_loader_list


def get_all_dataloader_order3(args):
    test_loader_list = []
    val_loader_list = []
    kwargs = {'num_workers': args.works, 'pin_memory': True}
    train_loader = DataLoaderX(
        ppa_dataloader_order3(args.data_root, fold='train', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.RandomRotation(30),
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order3(args.data_root, fold='test', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order3(args.data_root, fold='val', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order3(args.data_root, fold='test2', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order3(args.data_root, fold='val2', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    test_loader_list.append(DataLoaderX(
        ppa_dataloader_order3(args.data_root, fold='test3', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    val_loader_list.append(DataLoaderX(
        ppa_dataloader_order3(args.data_root, fold='val3', eye=args.eye, center=args.center, label=args.label,
                              filename=args.filename, feature_list=args.feature_list,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))

    # test_loader_list.append(DataLoaderX(
    #     ppa_dataloader_order3(args.data_root, fold='test4', eye=args.eye, center=args.center, label=args.label,
    #                    filename=args.filename, feature_list=args.feature_list,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True))
    # val_loader_list.append(DataLoaderX(
    #     ppa_dataloader_order3(args.data_root, fold='val4', eye=args.eye, center=args.center, label=args.label,
    #                    filename=args.filename, feature_list=args.feature_list,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True))
    return train_loader, val_loader_list, test_loader_list
