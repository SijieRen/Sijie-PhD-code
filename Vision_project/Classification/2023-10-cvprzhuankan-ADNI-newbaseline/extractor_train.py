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
from torchsampler import ImbalancedDatasetSampler


# clip_gradient(optimizer, opt.clip)

def train(opt):
    # Datasets
    dataset_extractor_train = ExtractorData(data_index=opt.data_index, data_root=opt.data_root, mode='train')
    dataset_extractor_test = ExtractorData(data_index=opt.data_index, data_root=opt.data_root, mode='test')
    dataloader_extractor_train = DataLoader(dataset_extractor_train, sampler=ImbalancedDatasetSampler(dataset_extractor_train), 
                                            batch_size=opt.batchsize, shuffle=False, num_workers=8)
    dataloader_extractor_test = DataLoader(dataset_extractor_test, batch_size=opt.batchsize, shuffle=False, num_workers=8)

    # Model
    model_1_extractor = RN18_extrator().cuda()
    model_2_generator = RN18_generator().cuda()

    # optimizer
    total_step = len(dataloader_extractor_train)
    optimizer_extractor = torch.optim.SGD([{'params': model_1_extractor.parameters(), 'lr': opt.lr,
                                'weight_decay': opt.decay_rate, 'momentum': opt.momentum}])
    scheduler_extractor = torch.optim.lr_scheduler.StepLR(optimizer_extractor,step_size=total_step*opt.decay_epoch,gamma=0.5)
    
    optimizer_generator = torch.optim.SGD([{'params': model_1_extractor.parameters(), 'lr': opt.lr,
                                'weight_decay': opt.decay_rate, 'momentum': opt.momentum}])
    scheduler_generator = torch.optim.lr_scheduler.StepLR(optimizer_generator,step_size=total_step*opt.decay_epoch,gamma=0.5)

    # train
    if not os.path.exists(opt.savepath):
        os.makedirs(opt.savepath)
    global_step = 0
    best_auc = 0
    best_acc = 0

    for epoch in range(int(opt.epoch)):
        model_1_extractor.train()
        model_2_generator.train()

        for step, (data_3D, label) in enumerate(dataloader_extractor_train):
            data_3D, label = data_3D.cuda().float(), label.cuda().float()
            data_3D = F.interpolate(data_3D, scale_factor=opt.data_scale)

            feature = model_1_extractor(data_3D)
            pred = model_2_generator(feature)

            loss = F.cross_entropy(pred, label)

            optimizer_extractor.zero_grad()
            optimizer_generator.zero_grad()
            loss.backward()
            optimizer_extractor.step()
            optimizer_generator.step()
            scheduler_extractor.step()
            scheduler_generator.step()
            
            global_step += 1
            if step % 10 == 0 or step == total_step-1:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f ' %
                      (datetime.datetime.now(), global_step, epoch + 1, opt.epoch, optimizer_extractor.param_groups[0]['lr'], loss))


        model_1_extractor.eval()
        model_2_generator.eval()
        pred_all = []
        label_all = []
        for i, (data, label) in tqdm(enumerate(dataloader_extractor_test)):
            data, label = data.cuda().float(), label.cuda().float()
            data = F.interpolate(data, scale_factor=opt.data_scale)
            feature = model_1_extractor(data)
            pred = model_2_generator(feature)
            pred = pred.sigmoid().detach().cpu().numpy()
            pred_all.extend(pred)

            label = label.cpu().numpy()
            label_all.extend(label)
            
        auc = roc_auc_score(label_all, pred_all, multi_class='ovr')
        acc = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_all, axis=1))
        print(np.argmax(label_all, axis=1))
        print(np.argmax(pred_all, axis=1))
        
        if auc > best_auc and epoch >= 10:
            states = {
                'metric_auc': auc,
                'metric_acc': acc,
                'extractor_state_dict': model_1_extractor.state_dict(),
                'generator_state_dict': model_2_generator.state_dict(),
            }
            torch.save(states, os.path.join(opt.savepath, 'model-best.pth'))
            best_auc = auc
        best_acc = acc if best_acc < acc else best_acc

        print("auc:{:.4f} | best_auc:{:.4f} | acc:{:.4f} | best_acc:{:.4f}".format(auc, best_auc, acc, best_acc))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Optimizer config
    parser.add_argument('--epoch', type=int,
                        default=5e3, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='SGD momentum rate')
    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of model weight')
    parser.add_argument('--decay_epoch', type=int,
                        default=500, help='every n epochs decay learning rate')

    # Checkpoints config
    parser.add_argument('--savepath', type=str, default='./checkpoints/extractor/')

    # Datasets config
    parser.add_argument('--data_scale', type=float,
                        default=1, help='downsample or upsample for data')
    parser.add_argument('--data_index', type=str,
                        default='./data/dataset_extractor.xls', help='path to train dataset')
    parser.add_argument('--data_root', type=str,
                        default='/root/autodl-tmp/ImageAll_V1_part1/npy_processed_data/', help='path to box train dataset')


    opt = parser.parse_args()
    train(opt)