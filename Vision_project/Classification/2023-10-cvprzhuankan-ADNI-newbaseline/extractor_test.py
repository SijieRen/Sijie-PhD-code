# coding=utf-8
import argparse
import os
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import ExtractorData
from model import *
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score



def extractor_test(opt):
    # checkpoints
    model_1_state = torch.load(opt.checkpoints)['extractor_state_dict']
    model_2_state = torch.load(opt.checkpoints)['generator_state_dict']


    # Model
    model_1_extractor = RN18_extrator().cuda()
    model_2_generator = RN18_generator().cuda()
    model_1_extractor.eval()
    model_2_generator.eval()
    model_1_extractor.load_state_dict(model_1_state)
    model_2_generator.load_state_dict(model_2_state)

    # Dataloader
    dataset_extractor_train = ExtractorData(data_index=opt.data_index, data_root=opt.data_root, mode='train')
    dataset_extractor_test = ExtractorData(data_index=opt.data_index, data_root=opt.data_root, mode='test')
    dataloader_extractor_train = DataLoader(dataset_extractor_train, batch_size=opt.batchsize, shuffle=True, num_workers=8)
    dataloader_extractor_test = DataLoader(dataset_extractor_test, batch_size=opt.batchsize, shuffle=True, num_workers=8)

    feature_all = []
    pred_all = []
    label_all = []
    for i, (data, label) in tqdm(enumerate(dataloader_extractor_train)):
        data, label = data.cuda().float(), label.cuda().float()
        data = F.interpolate(data, scale_factor=opt.data_scale)

        feature = model_1_extractor(data)
        print(feature.shape)
        feature_all.extend(feature.detach().cpu().numpy())

        pred = model_2_generator(feature)
        pred = pred.sigmoid().detach().cpu().numpy()
        pred_all.extend(pred)

        label = label.cpu().numpy()
        label_all.extend(label)

    for i, (data, label) in tqdm(enumerate(dataloader_extractor_test)):
        data, label = data.cuda().float(), label.cuda().float()
        data = F.interpolate(data, scale_factor=opt.data_scale)

        feature = model_1_extractor(data)
        feature_all.extend(feature.detach().cpu().numpy())

        pred = model_2_generator(feature)
        pred = pred.sigmoid().detach().cpu().numpy()
        pred_all.extend(pred)

        label = label.cpu().numpy()
        label_all.extend(label)

    name = 1
    for feature in feature_all:

        np.save(os.path.join('./result', str(name)+'.npy'), feature)
        name += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Optimizer config
    parser.add_argument('--epoch', type=int,
                        default=1e3, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='SGD momentum rate')
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    # Checkpoint config
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/model-best.pth')

    # Datasets config
    parser.add_argument('--data_scale', type=float,
                        default=1, help='downsample or upsample for data')
    parser.add_argument('--data_index', type=str,
                        default='./data/dataset_extractor.xls', help='path to train dataset')
    parser.add_argument('--data_root', type=str,
                        default='/home/thaumiel/Code/datasets/TIP2023/ImageAll_V1_part1/npy_processed_data/',
                        help='path to box train dataset')

    opt = parser.parse_args()
    extractor_test(opt)