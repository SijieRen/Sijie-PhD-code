import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from utils import *
from torchvision import transforms
from models import *
import torch.utils.data as data
import xlrd
from torch.utils.data import DataLoader
import torch.nn.functional as F

class MM_F_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 ):
        
        super(MM_F_2D, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        
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
        self.get_feature_u = nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512)
        )
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x, feature=0):
        x = self.get_feature(x)
        return self.fc(x)
    
    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer
    
    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class get_dataset_2D(data.Dataset):
    def __init__(self,
                 root=None,
                 fold='train',
                 args=None,
                 transform=None,
                 env_idx = 0):
        self.root = root
        self.args = args
        self.image_path_list = []
        self.u = []
        self.us = []
        self.train_u = []
        self.train_us = []
        self.y = []
        self.transform = transform
        self.env_idx = env_idx
        fold_map = {'train': 0, 'val': 1, 'test': 2}
        if args.dataset == 'NICO':
            if self.root is None:
                self.root = '/home/botong/Dataset/'
            
            workbook = xlrd.open_workbook(r"/home/botong/Dataset/NICO_dataset.xls")
            if '8' in args.dataset_type:
                sheet = workbook.sheet_by_index(0)
            elif '2' in args.dataset_type:
                sheet = workbook.sheet_by_index(2)
            else:
                sheet = workbook.sheet_by_index(1)
            for rows in range(1, sheet.nrows):
                if sheet.row_values(rows)[4] == fold_map[fold]:
                    if fold == 'train':
                        if sheet.row_values(rows)[3] == env_idx:
                            self.image_path_list.append(os.path.join(self.root, sheet.row_values(rows)[0]))
                            self.y.append(sheet.row_values(rows)[1])
                    else:
                        self.image_path_list.append(os.path.join(self.root, sheet.row_values(rows)[0]))
                        self.y.append(sheet.row_values(rows)[1])
        
    def __getitem__(self, index):
        with open(self.image_path_list[index], 'rb') as f:
            img_1 = Image.open(f)
            if '225. cat_mm8_2-min.png' in self.image_path_list[index]:
                img_1 = np.asarray(img_1.convert('RGBA'))[:,:,:3]
                img_1 = Image.fromarray(img_1.astype('uint8'))
            else:
                img_1 = Image.fromarray(np.asarray(img_1.convert('RGB')).astype('uint8'))
        if self.transform is not None:
            img_1 = self.transform(img_1)
        return img_1, \
               torch.from_numpy(np.array(self.y[index]).astype('int'))
    
    def __len__(self):
        return len(self.image_path_list)

def get_train_loader_list(args):
    data_loader_list = []
    if '8' in args.dataset_type:
        args.env_num = 8
    elif '2' in args.dataset_type:
        args.env_num = 2
    else:
        args.env_num = 10
    for ss in range(args.env_num):
        data_loader_list.append(DataLoader(get_dataset_2D(args=args, fold='train', env_idx=ss,
                                              transform=transforms.Compose([
                                                  transforms.RandomResizedCrop(256),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
                                              ])),
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=0,
                               pin_memory=True))
    return args, data_loader_list
    
def mean_nll(logits, y):
    return F.cross_entropy(logits, y)

def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def train(epoch,
          model,
          optimizer,
          dataloader_list,
          args):
    RECON_loss = AverageMeter()
    KLD_loss = AverageMeter()
    classify_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    model.train()
    for dataloader in dataloader_list:
        print(dataloader.dataset.env_idx)
        for batch_idx, (x, target) in enumerate(dataloader):
            if args.cuda:
                x, target = x.cuda(), target.cuda()
            
            pred_y = model(x)
            loss = mean_nll(pred_y, target) + args.alpha * penalty(pred_y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_loss.update(loss.item(), x.size(0))
            accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
            
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
        
    return accuracy.avg


def evaluate(model, dataloader, args):
    model.eval()
    accuracy = AverageMeter()
    for batch_idx, (x, target) in enumerate(dataloader):
        if args.cuda:
            x, target = x.cuda(), target.cuda()
        
        pred_y = model(x)
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
    return accuracy.avg

def main():
    args = get_opt()
    args = make_dirs(args)
    logger = get_logger(args)
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    args, train_loader_list = get_train_loader_list(args)
    test_loader = DataLoaderX(get_dataset_2D(args=args, fold='test',
                                             transform=transforms.Compose([
                                                 transforms.CenterCrop(256),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
                                             ])),
                              batch_size=args.test_batch_size,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True)
    model = MM_F_2D(in_channel=args.in_channel,
                    u_dim=args.u_dim,
                    us_dim=args.us_dim,
                    num_classes=args.num_classes,
                    is_use_u=args.is_use_u,
                    ).cuda()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    
    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.lr_controler)
        _ = train(epoch, model, optimizer, train_loader_list, args)
        # val_acc = evaluate(model, val_loader, args)
        test_acc = evaluate(model, test_loader, args)
        if test_acc > best_acc:
            best_acc = copy.deepcopy(test_acc)
            best_acc_ep = copy.deepcopy(epoch)
            is_best = 1
        else:
            is_best = 0
        checkpoint(epoch, args.model_save_dir, model, is_best)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d'
                    % (epoch, test_acc, best_acc, best_acc_ep))
    logger.info('model save path: %s' % args.model_save_dir)
    logger.info('*' * 50)
    logger.info('*' * 50)


if __name__ =='__main__':
    main()

