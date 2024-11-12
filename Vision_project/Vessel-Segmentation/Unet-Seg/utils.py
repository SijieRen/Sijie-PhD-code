import torch
import torch.nn as nn
from collections import defaultdict
# -*- coding: UTF-8 -*-
import torch
import numpy as np
import os
import xlrd
from PIL import Image
import copy
import os
import pickle
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
import sklearn.metrics
import shutil
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
import datetime
import torch
from torchvision import models
import cv2
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils
import torch.autograd as autograd
import argparse


def get_opts():
    parser = argparse.ArgumentParser(description='PyTorch PrimarySchool Fundus photography RA-Regression')
    # Params
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=24, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='Mixed',
                        help='choose the optimizer function')
    
    ### Unet
    parser.add_argument('--U_net_type', type=str, default='normal')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--lr_decay', type=float, default=0.2, help='')
    parser.add_argument('--lr_controler', type=int, default=60, )
    parser.add_argument('--wd', type=float, default=0.00001, help='')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--betas1', type=float, default=0.9,
                        help='parameter for SGD')
    parser.add_argument('--betas2', type=float, default=0.999,
                        help='parameter for SGD')
    
    parser.add_argument('--sequence', type=str, default='12-08/12-08-1',
                        help='save the check point of the model and optim')
    parser.add_argument('--bi_linear', type=int, default='1')

    parser.add_argument('--plot_ep', type=int, default='1')
    parser.add_argument('--threshold', type=float, default='200')
    parser.add_argument('--clip', type=int, default='0')

    # Dataset
    parser.add_argument('--data_root', type=str, default='../data/JPG256SIZE-95')
    parser.add_argument('--filename', type=str, default='std')
    parser.add_argument('--feature_list', type=int, nargs='+', default=[8, 9], help='')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--eye', type=str, default='R',
                        help='choose the right or left eye')
    parser.add_argument('--center', type=str, default='Maculae',
                        help='choose the center type pf the picture,input(Maculae)(Disc)(Maculae&Disc) ')

    # Model
    parser.add_argument('--load_dir', type=str, default='/home')
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--works', type=int, default=2)

    # Visulization
    parser.add_argument('--is_plot', type=int, default=0)
    parser.add_argument('--plot_sigmoid', type=int, default=0)
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--label', type=str, default='RA',
                        help='choose the label we want to do research')

    # Train set
    
    parser.add_argument('--wd2', type=float, default=0.0001, help='')
    parser.add_argument('--dp', type=float, default=0.0)
    parser.add_argument('--is_ESPCN', type=int, default=0,
                        help='whether load model ESPCN')
    parser.add_argument('--dw_midch', type=int, default=1024,
                        help='choose the mid-channel type of ESPCN')
    parser.add_argument('--RNN_hidden', type=int, nargs='+', default=[256], help='')
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--dp_n', type=float, default=0.5)
    parser.add_argument('--wcl', type=float, default=0.01,
                        help='control the clamp parameter of Dnet, weight_cliping_limit')
    parser.add_argument('--isplus', type=int, default=0)
    parser.add_argument('--final_tanh', type=int, default=0)
    parser.add_argument('--train_ori_data', type=int, default=0)
    parser.add_argument('--pred_future', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1,
                        help='parameter of loss image')

    parser.add_argument('--gamma', type=float, default=1,
                        help='parameter of real loss RA prediction')
    parser.add_argument('--delta1', type=float, default=0.001,
                        help='parameter of real loss RA prediction')
    parser.add_argument('--delta', type=float, default=0.001,
                        help='parameter of real loss RA prediction')
    
    parser.add_argument('--D_epoch', type=int, default=4)

    parser.add_argument('--noise_size', type=int, default=516,
                        help='control the size of the input of Gnet')
    parser.add_argument('--save_checkpoint', type=int, default=1,
                        help='save the check point of the model and optim')


    args = parser.parse_args()
    args.final_tanh = 0
    time_stamp = datetime.datetime.now()
    save_dir = '../results/Unet-Seg_%s/%s_%s_%f_%f_size_%s_ep_%d_%d_%s_%s/' % (
        args.sequence,
        time_stamp.strftime(
            '%Y-%m-%d-%H-%M-%S'),
        args.optimizer,
        args.lr,
        args.lr_controler,
        args.image_size,
        args.epochs,
        args.wcl,
        args.eye,
        args.center)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.logger = get_logger(args)

    return args

def binarize_image(image, threshold):
    """Binarize an image."""
    image = image.convert('L')  # convert image to monochrome
    image = np.array(image)
    #print('threshold, before binarize',threshold, np.max(image), image)
    image = binarize_array(image, threshold)
    
    return image



def binarize_array(numpy_array, threshold=200):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


def save_image(args, image_sequence, image, test_loader, epoch):
    unloader2 = transforms.Compose([
        transforms.ToPILImage(),
    
    ])
    
    print('Ploting the %d-th image'%(image_sequence))
    time_stamp = datetime.datetime.now()
    data = image
    image_path_sequence = image_sequence
    save_dir_2 = '../Unet-Seg-Generate-image/%s/%s_%s_%f_%f___ep_%d/' % (
        args.sequence,
        time_stamp.strftime(
            '%Y-%m-%d'), ##-%H-%M-%S
        args.optimizer,
        args.lr,
        args.lr_controler,
        epoch,
    )
    if not os.path.exists(save_dir_2):
        os.makedirs(save_dir_2)
    for i in range(data.size(0)):
        image = data[i, :, :, :].cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader2(image)
        image = np.array(image)
        image = binarize_array(image, args.threshold)
        #print('after binirize', image)
        image = Image.fromarray(image)
        image_sequence_ = image_path_sequence - data.size(0) + i
        image.save(os.path.join(save_dir_2, 'generate_' + os.path.basename(
            test_loader.dataset.image_path[image_sequence_])))
    

def load_pytorch_model(model, path):
    from collections import OrderedDict
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)  ###parameter strict: do not load the part dont match
    return model


def get_logger(opt, fold=0):
    import logging
    logger = logging.getLogger('AL')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler('{}training_{}.log'.format(opt.save_dir, fold))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def draw_features(width, height, feature, savepath, image_sequence, train_loader, a, batchsize, dpi):
    str = a
    batch_size = batchsize
    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)

    num = image_sequence
    x = feature
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255
        img = img.astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        img = img[:, :, ::-1]
        plt.imshow(img)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savename = os.path.join(savepath, '%s' % (str) + os.path.basename(
        train_loader.dataset.image_path_all_1[num]))
    fig.savefig(savename, dpi=dpi)
    fig.clf()
    plt.close()


def init_metric(args):
    args.best_test_loss = 2
    args.best_test_loss_epoch = 0

    args.best_val_loss = 2
    args.best_val_loss_epoch = 0
    
    return args


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', base_dir='./'):
    torch.save(state, os.path.join(base_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(base_dir, filename), os.path.join(base_dir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.lr_decay ** (epoch // args.lr_controler))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
    
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        
        #probs = F.sigmoid(logits)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()



def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



