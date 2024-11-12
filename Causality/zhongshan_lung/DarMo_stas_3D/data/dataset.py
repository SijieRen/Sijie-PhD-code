import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy
from PIL import Image
import torchvision as tv
import copy
import json
import torch.utils.data.distributed



def z_make_dataset(img_dir, json_dir):
    file_test = open(json_dir, "r")
    json_data = file_test.read()
    dict_data = json.loads(json_data)
    file_test.close()
    images = []
    id_dict = {}
    # class_to_idx = dict_data['class_to_idx']
    class_to_idx = dict_data['class_to_idx']
    img_info_list = dict_data['img_data']

    for img_info in img_info_list:

        img_nm = img_info[0]
        newname = img_nm.split('/data/')[1]
        if newname.split('/')[0] not in id_dict.keys():
            id_dict[newname.split('/')[0]] = 1
        else:
            id_dict[newname.split('/')[0]] = id_dict[newname.split('/')[0]] + 1

    for img_info in img_info_list:
        if 'ddsm' in img_info[0]:
            bingli_0, bingli_1, shape_circle, shape_oval, shape_irre, margin_clear, margin_blur, margin_shade, fenye_no, fenye_is, maoci_no, maoci_is = img_info
            img_nm = img_info[0]
            benmag_cls = img_info[1]
            list_cls = [bingli_0, bingli_1, shape_circle, shape_oval, shape_irre, margin_clear, margin_blur, margin_shade, fenye_no, fenye_is, maoci_no, maoci_is]
        else:
            bingli_0, bingli_1, shape_circle, shape_oval, shape_irre, density_high, density_low, density_equal, density_fat, \
            margin_clear, margin_blur, margin_shade, fenye_no, fenye_is, maoci_no, maoci_is = img_info
            img_nm = img_info[0]
            benmag_cls = img_info[1]
            list_cls = [bingli_0, bingli_1, shape_circle, shape_oval, shape_irre, margin_clear, margin_blur,
                        margin_shade, fenye_no, fenye_is, maoci_no, maoci_is]

        id_d = list(id_dict.keys()).index(newname.split('/')[0])
        item = (default_loader(img_nm), class_to_idx[str(benmag_cls)], list_cls, str(id_d))
        images.append(item)

    return class_to_idx, images


def default_loader(path):
    return Image.open(path).convert('RGB')


class CalImageFolder(torch.utils.data.Dataset):
    def __init__(self, img_path, json_path, transform=None, target_transform=None, loader=default_loader):
        class_to_idx, imgs = z_make_dataset(img_path, json_path)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + img_path + "\n"
                                                                                 "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.root = img_path
        self.imgs = imgs
        # self.classes = class_to_idx.keys()
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target, gcn_target, id_tar = self.imgs[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, gcn_target, id_tar

    def __len__(self):
        return len(self.imgs)