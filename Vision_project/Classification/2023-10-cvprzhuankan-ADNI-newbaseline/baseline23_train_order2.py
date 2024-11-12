# coding=utf-8
import argparse
import os
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import ExtractorData, Order1Data
from model import *
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score


# clip_gradient(optimizer, opt.clip)

def train(opt):
    # Datasets
    if opt.data_order == 1:
        dataset_train = Order1Data(data_index='./data/dataset_order%s.xls' % '1',
                                   data_root=opt.data_root, mode='train')
        dataset_test = Order1Data(data_index='./data/dataset_order%s.xls' % '1',
                                  data_root=opt.data_root, mode='test')
    elif opt.data_order == 2:
        dataset_train = ExtractorData(data_index='./data/dataset_order%s.xls' % '2',
                                      data_root=opt.data_root, mode='train')
        dataset_test = ExtractorData(data_index='./data/dataset_order%s.xls' % '2',
                                     data_root=opt.data_root, mode='test')
    else:
        raise TypeError('Invalid order!')

    dataloader_train = DataLoader(dataset_train, batch_size=opt.batchsize, shuffle=True,
                                  num_workers=8)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batchsize, shuffle=True,
                                 num_workers=8)

    # Model
    if opt.baseline == 'RN18':
        model_baseline = RN18().cuda()
    elif opt.baseline == 'ARL':
        model_baseline = RN18_ARL().cuda()
    elif opt.baseline == 'FFC':
        model_baseline = RN18_FFC_3d().cuda()
    elif opt.baseline == 'GPDBN':
        model_baseline = RN18_GPDBN().cuda()
    # elif opt.baseline == 'MM':
    #     model_baseline = RN18()
    else:
        raise TypeError('This baseline is not supported!')

    # optimizer
    optimizer_baseline = torch.optim.SGD([{'params': model_baseline.parameters(), 'lr': opt.lr,
                                           'weight_decay': opt.decay_rate, 'momentum': opt.momentum}])

    # train
    if not os.path.exists(opt.savepath):
        os.makedirs(opt.savepath)
    global_step = 0
    best_auc = 0

    for epoch in range(int(opt.epoch)):  # sijie waited to modified
        model_baseline.train()

        total_step = len(dataloader_train)
        for step, (data_3D, label) in enumerate(dataloader_train):
            data_3D, label = data_3D.cuda().float(), label.cuda().float()
            data_3D = F.interpolate(data_3D, scale_factor=opt.data_scale)

            pred = model_baseline(data_3D)

            loss = F.cross_entropy(pred, label)

            optimizer_baseline.zero_grad()
            loss.backward()
            optimizer_baseline.step()
            global_step += 1
            if step % 10 == 0 or step == total_step-1:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f ' %
                      (datetime.datetime.now(), global_step, epoch + 1, opt.epoch,
                       optimizer_baseline.param_groups[0]['lr'], loss.item()))

        model_baseline.eval()
        pred_all = []
        label_all = []
        for i, (data, label) in tqdm(enumerate(dataloader_test)):
            data, label = data.cuda().float(), label.cuda().float()
            data = F.interpolate(data, scale_factor=opt.data_scale)
            if opt.baseline == 'FFC':
                pred = model_baseline(data_3D, delta_t)
            elif opt.baseline == 'GPDB':
                pred = model_baseline(data_3D, delta_t)
            else:
                pred = model_baseline(data)
            pred = pred.sigmoid().detach().cpu().numpy()
            pred_all.extend(pred)

            label = label.cpu().numpy()
            label_all.extend(label)
        auc = roc_auc_score(label_all, pred_all, multi_class='ovr')
        if auc > best_auc:
            states = {
                'metric': auc,
                'baseline_state_dict': model_baseline.state_dict(),
            }
            torch.save(states, os.path.join(opt.savepath,
                                            'model-%s-order%s-best.pth' % (opt.baseline, str(opt.data_order))))
            best_auc = auc

        print("auc:{:.4f} | best_auc:{:.4f}".format(auc, best_auc))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model config
    parser.add_argument('--baseline', type=str,
                        default='RN18', help='baseline option')

    # Optimizer config
    parser.add_argument('--epoch', type=int,
                        default=5e3, help='epoch number')
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

    # Checkpoints config
    parser.add_argument('--savepath', type=str,
                        default='./checkpoints/baseline/')

    # Datasets config
    parser.add_argument('--data_order', type=int,
                        default=1, help='data order')
    parser.add_argument('--data_scale', type=float,
                        default=0.5, help='downsample or upsample for data')
    parser.add_argument('--data_root', type=str,
                        default='/home/thaumiel/Code/datasets/TIP2023/ImageAll_V1_part1/npy_processed_data/', help='path to box train dataset')

    opt = parser.parse_args()
    train(opt)
