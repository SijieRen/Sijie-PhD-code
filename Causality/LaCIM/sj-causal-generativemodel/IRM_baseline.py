import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from utils import *
from utils import get_dataset_2D_env as get_dataset_2D
from torchvision import transforms
from models import *
import torch.utils.data as data
import xlrd
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys

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


class MM_F_ff_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim = 256,
                 ):
        super(MM_F_ff_2D, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.relu = nn.ReLU()
        
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
        self.fc_1 = nn.Linear(1024, self.zs_dim)
        self.fc = nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes), )
    
    def forward(self, x, feature=0):
        x = self.get_feature(x)
        fea = self.relu(self.fc_1(x))
        if feature:
            return self.fc(fea), fea
        else:
            return self.fc(fea)
    
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
          dataloader,
          args):
    RECON_loss = AverageMeter()
    KLD_loss = AverageMeter()
    IRM_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    model.train()
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda(), env.cuda(), u.cuda()
        
        pred_y = model(x)
        loss = torch.FloatTensor([0.0]).cuda()
        irm_loss = torch.FloatTensor([0.0]).cuda()
        l2_reg = torch.tensor(0.).cuda()
        for param in model.parameters():
            l2_reg += torch.norm(param)
        
        loss = torch.add(loss, args.reg *l2_reg)
        
        cls_loss_all = []
        for ss in range(args.env_num):
            if torch.sum(env == ss) == 0:
                continue
            loss = torch.add(loss, torch.sum(env == ss) * (F.cross_entropy(pred_y[env == ss, :], target[env == ss])))
            if args.rex:
                cls_loss_all.append(F.cross_entropy(pred_y[env==ss, :], target[env == ss]))
            else:
                irm_loss = torch.add(irm_loss, penalty(pred_y[env == ss, :], target[env == ss]))

        if args.rex:
            for s_i in range(args.env_num):
                for s_j in range(s_i + 1, args.env_num):
                    irm_loss = torch.add(irm_loss, (cls_loss_all[s_i] - cls_loss_all[s_j]) ** 2)
        else:
            irm_loss = irm_loss / pred_y.size(0)
        if epoch <= args.alpha_epoch:
            weight = 1
        else:
            weight = args.alpha
        loss = weight * irm_loss  + loss / pred_y.size(0)
        if weight > 1:
            loss = loss / weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss.update(loss.item(), x.size(0))
        IRM_loss.update(irm_loss.item(), x.size(0))
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
        
        if batch_idx % 10 == 0:
            args.logger.info(
                'epoch [{}/{}], batch: {}, rec_loss:{:.4f}, kld_loss:{:.4f} irm_loss:{:.4f}, overall_loss:{:.4f},acc:{:.4f}'
                .format(epoch,
                        args.epochs,
                        batch_idx,
                        RECON_loss.avg,
                        KLD_loss.avg * args.beta,
                        IRM_loss.avg,
                        all_loss.avg,
                        accuracy.avg * 100))
        
    return accuracy.avg


def evaluate(model, dataloader, args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target = x.cuda(), target.cuda()
        
        pred_y = model(x)
        pred[batch_begin:batch_begin + x.size(0), :] = pred_y.detach().cpu()
        batch_begin = batch_begin + x.size(0)
        
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
    return pred, accuracy.avg

def main():
    args = get_opt()
    args = make_dirs(args)
    logger = get_logger(args)
    args.logger = logger
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    if args.dataset == 'NICO':
        train_loader = DataLoaderX(get_dataset_2D(args=args, fold='train',
                                                          transform=transforms.Compose([
                                                              transforms.RandomResizedCrop((256, 256)),
                                                              transforms.RandomHorizontalFlip(p=0.5),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                          ])),
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)
        test_loader = DataLoaderX(get_dataset_2D(args=args, fold='test',
                                                 transform=transforms.Compose([
                                                     transforms.Resize((256, 256)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True)

        val_loader = DataLoaderX(get_dataset_2D(args=args, fold='val',
                                                 transform=transforms.Compose([
                                                     transforms.Resize((256, 256)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True)
    elif 'mnist' in args.dataset:
        train_loader = DataLoader(get_dataset_2D(root = args.root, args=args, fold='train',
                                                 transform=transforms.Compose([
                                                     transforms.RandomHorizontalFlip(p=0.5),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.worker,
                                  pin_memory=True)
        test_loader = DataLoaderX(get_dataset_2D(root = args.root, args=args, fold='test',
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=args.worker,
                                  pin_memory=True)
    
    if 'NICO' in args.dataset:
        model = MM_F_ff_2D_NICO(in_channel=args.in_channel,
                                u_dim=args.u_dim,
                                us_dim=args.us_dim,
                                num_classes=args.num_classes,
                                is_use_u=args.is_use_u,
                                zs_dim=args.zs_dim,
                                ).cuda()
    else:
        if args.model == 'MM_F_f':
            if args.smaller_net:
                model = MM_F_ff_2D_mnist(in_channel=args.in_channel,
                                   u_dim=args.u_dim,
                                   us_dim=args.us_dim,
                                   num_classes=args.num_classes,
                                   is_use_u=args.is_use_u,
                                   zs_dim=args.zs_dim,
                                   ).cuda()
            else:
                model = MM_F_ff_2D(in_channel=args.in_channel,
                                u_dim=args.u_dim,
                                us_dim=args.us_dim,
                                num_classes=args.num_classes,
                                is_use_u=args.is_use_u,
                                   zs_dim=args.zs_dim,
                                ).cuda()
        
        elif args.model == 'MM_F_f_L':
            model = MM_F_ff_2D_mnist_L(in_channel=args.in_channel,
                                     u_dim=args.u_dim,
                                     us_dim=args.us_dim,
                                     num_classes=args.num_classes,
                                     is_use_u=args.is_use_u,
                                     zs_dim=args.zs_dim,
                                     ).cuda()
        
        else:
            model = MM_F_2D(in_channel=args.in_channel,
                        u_dim=args.u_dim,
                        us_dim=args.us_dim,
                        num_classes=args.num_classes,
                        is_use_u=args.is_use_u,
                        ).cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params, '%0.4f M' % (pytorch_total_params / 1e6))

    if args.inference_only and args.load_path != '':
        model.load_state_dict(torch.load(os.path.join(args.load_path, 'best_acc.pth.tar'))['state_dict'])
        test_loader_NICO = DataLoaderX(get_dataset_NICO_inter(args=args,
                                                              transform=transforms.Compose([
                                                                  transforms.Resize((256, 256)),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                                       (0.5, 0.5, 0.5)),
    
                                                              ])),
                                       batch_size=args.test_batch_size,
                                       shuffle=False,
                                       num_workers=1,
                                       pin_memory=True)
        pred_test, label_test, test_acc = evaluate_only(model, test_loader_NICO, args)
        save_pred_label_as_xlsx(args.model_save_dir, 'pred.xls', pred_test, label_test, test_loader_NICO, args)
        logger.info('model save path: %s' % args.model_save_dir)
        print('test_acc', test_acc)
        exit(123)
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay, args.lr_controler)
        _ = train(epoch, model, optimizer, train_loader, args)
        pred_test, test_acc = evaluate(model, test_loader, args)
        if args.dataset == 'NICO':
            pred_val, val_acc = evaluate(model, val_loader, args)
        else:
            pred_val = copy.deepcopy(pred_test)
            val_acc = copy.deepcopy(test_acc)

        if test_acc > best_acc:
            best_acc = copy.deepcopy(test_acc)
            best_acc_ep = copy.deepcopy(epoch)
            is_best = 1
        else:
            is_best = 0
        other_info = {
            'pred_val': pred_val,
            'pred_test': pred_test,
        }
        checkpoint(epoch, args.model_save_dir, model, is_best, other_info, logger)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d, val_acc: %0.4f'
                    % (epoch, test_acc, best_acc, best_acc_ep, val_acc))
    logger.info('model save path: %s' % args.model_save_dir)
    xlsx_name = '%s_IRM_u_%d_fold_%d.xls' % \
                (os.path.basename(sys.argv[0][:-3]), args.is_use_u,
                 args.fold)
    save_results_as_xlsx('./results/', xlsx_name, best_acc, best_acc_ep, auc=None, args=args)
    logger.info('model save path: %s' % args.model_save_dir)
    logger.info('*' * 50)
    logger.info('*' * 50)


if __name__ =='__main__':
    main()

