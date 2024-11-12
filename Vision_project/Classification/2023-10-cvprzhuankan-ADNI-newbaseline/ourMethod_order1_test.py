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
from torch.utils.tensorboard import SummaryWriter


# clip_gradient(optimizer, opt.clip)

def train(opt):
    # Datasets
    dataset_train = Order1Data(data_index='./data/dataset_order%s.xls' % '1',
                               data_root=opt.data_root, mode='train')
    dataset_test = Order1Data(data_index='./data/dataset_order%s.xls' % '1',
                              data_root=opt.data_root, mode='test')

    dataloader_train = DataLoader(dataset_train, batch_size=opt.batchsize, shuffle=True,
                                  num_workers=8)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batchsize, shuffle=False,
                                 num_workers=8)

    # Model
    ## extractor
    extractor = RN18_extrator().cuda()
    extractor.eval()
    extractor.load_state_dict(torch.load(opt.extractor_path)['extractor_state_dict'])
    ## generator
    G_net = Generator_LSTM_1(in_channel=64).cuda()
    G_net.eval()
    G_net.load_state_dict(torch.load(opt.model_path)['G_Net_dict'])
    ## classifer
    C_net = RN18_Classifer().cuda()
    C_net.eval()
    C_net.load_state_dict(torch.load(opt.model_path)['C_Net_dict'])

    C_Res_net = RN18_Res_Classifer().cuda()
    C_Res_net.eval()
    C_Res_net.load_state_dict(torch.load(opt.model_path)['C_Res_Net_dict'])

    # Test
    pred_all = []
    pred_gen_all = []
    pred_res_all = []
    pred_average_all = []
    label_all = []
    for step, (data_3D_t, data_3D_T, label, grad1, grad2) in enumerate(dataloader_test):
        data_size = 128
        batch_size = data_3D_t.size(0)
        data_3D_t, data_3D_T, label, grad1, grad2 = data_3D_t.cuda().float(), \
                                                    data_3D_t.cuda().float(), \
                                                    label.cuda().float(), \
                                                    grad1.cuda().float(), \
                                                    grad2.cuda().float()

        data_3D_t = F.interpolate(data_3D_t, size=(data_size, data_size, data_size), mode='trilinear')

        # inference
        featuremap_t = extractor(data_3D_t)
        ## generate feature
        z = torch.randn(batch_size, 100, 1, 1, 1).cuda()
        generated_feature = G_net(featuremap_t, z, grad1, grad2)
        ## res
        res_feature = generated_feature - featuremap_t
        ## pred
        pred_t = torch.softmax(C_net(featuremap_t), dim=1)
        pred_res = torch.softmax(C_Res_net(res_feature), dim=1)
        pred_gen = torch.softmax(C_net(generated_feature), dim=1)
        pred1 = pred_t[::, 1] + pred_t[::, 0] * pred_res[::, 1]
        pred2 = pred_t[::, 2] + (1 - pred_t[::, 2]) * pred_res[::, 2]
        pred = torch.cat([(1 - pred1 - pred2).unsqueeze(1), pred1.unsqueeze(1), pred2.unsqueeze(1)], dim=1)
        pred_average = (pred_gen + pred) / 2

        pred_res = torch.softmax(pred_res, dim=1).detach().cpu().numpy()
        pred_res_all.extend(pred_res)

        pred = torch.softmax(pred, dim=1).detach().cpu().numpy()
        pred_all.extend(pred)

        pred_gen = torch.softmax(pred_gen, dim=1).detach().cpu().numpy()
        pred_gen_all.extend(pred_gen)

        pred_average = torch.softmax(pred_average, dim=1).detach().cpu().numpy()
        pred_average_all.extend(pred_average)

        label = label.cpu().numpy()
        label_all.extend(label)

    auc = roc_auc_score(label_all, pred_all, multi_class='ovr')
    auc_res = roc_auc_score(label_all, pred_res_all, multi_class='ovr')
    auc_gen = roc_auc_score(label_all, pred_gen_all, multi_class='ovr')
    auc_average = roc_auc_score(label_all, pred_average_all, multi_class='ovr')

    print(np.argmax(pred_all, axis=1))
    acc = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_all, axis=1))
    acc_res = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_res_all, axis=1))
    acc_gen = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_gen_all, axis=1))
    acc_average = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_average_all, axis=1))
    print("auc:{:.4f} | acc:{:.4f}".format(auc, acc))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', type=int,
                    default=8, help='test batch size')
    
    # Checkpoints config
    parser.add_argument('--model_path', type=str, default='./checkpoints/baseline/model-ourMethod-order1-0.9312.pth')
    parser.add_argument('--extractor_path', type=str,
                        default='./checkpoints/extractor/model-best-scale=0.5.pth')

    # Datasets config
    parser.add_argument('--data_root', type=str,
                        default='/root/autodl-tmp/ImageAll_V1_part1/npy_processed_data/',
                        help='path to box train dataset')

    opt = parser.parse_args()
    train(opt)


if __name__ == '__main__':
    main()

    # a = torch.randn([1])
    # print(a.shape)
