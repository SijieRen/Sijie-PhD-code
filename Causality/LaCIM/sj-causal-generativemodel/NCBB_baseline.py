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
from torch.autograd import Variable

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def projection_simplex_pivot(v, z=1, random_state=None):
    rs = np.random.RandomState(random_state)
    n_features = len(v)
    U = np.arange(n_features)
    s = 0
    rho = 0
    while len(U) > 0:
        G = []
        L = []
        k = U[rs.randint(0, len(U))]
        ds = v[k]
        for j in U:
            if v[j] >= v[k]:
                if j != k:
                    ds += v[j]
                    G.append(j)
            elif v[j] < v[k]:
                L.append(j)
        drho = len(G) + 1
        if s + ds - (rho + drho) * v[k] < z:
            s += ds
            rho += drho
            U = L
        else:
            U = G
    theta = (s - z) / float(rho)
    return np.maximum(v - theta, 0)


def projection_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000):
    lower = 0
    upper = np.max(v)
    current = np.inf

    for it in range(max_iter):
        if np.abs(current) / z < tau and current < 0:
            break

        theta = (upper + lower) / 2.0
        w = np.maximum(v - theta, 0)
        current = np.sum(w) - z
        if current <= 0:
            upper = theta
        else:
            lower = theta
    return w

def lossb(cfeaturec, weight, cfs):
    cfeatureb = (cfeaturec.sign() + 1).sign()
    mfeatureb = 1 - cfeatureb
    loss = Variable(torch.FloatTensor([0]).cuda())
    for p in range(cfs):
        if p == 0:
            cfeaturer = cfeaturec[:, 1: cfs]
        elif p == cfs - 1:
            cfeaturer = cfeaturec[:, 0: cfs - 1]
        else:
            cfeaturer = torch.cat((cfeaturec[:, 0: p], cfeaturec[:, p + 1: cfs]), 1)
        
        if cfeatureb[:, p: p + 1].t().mm(weight).view(1).data[0] != 0 or \
                mfeatureb[:, p: p + 1].t().mm(weight).view(1).data[0] != 0:
            if cfeatureb[:, p: p + 1].t().mm(weight).view(1).data[0] == 0:
                loss += (cfeaturer.t().mm(mfeatureb[:, p: p + 1] * weight) / mfeatureb[:, p: p + 1].t().mm(weight)).pow(
                    2).sum(0).view(1)
            elif mfeatureb[:, p: p + 1].t().mm(weight).view(1).data[0] == 0:
                loss += (cfeaturer.t().mm(cfeatureb[:, p: p + 1] * weight) / cfeatureb[:, p: p + 1].t().mm(weight)).pow(
                    2).sum(0).view(1)
            else:
                loss += (cfeaturer.t().mm(cfeatureb[:, p: p + 1] * weight) / cfeatureb[:, p: p + 1].t().mm(weight) -
                         cfeaturer.t().mm(mfeatureb[:, p: p + 1] * weight) / mfeatureb[:, p: p + 1].t().mm(weight)).pow(
                    2).sum(0).view(1)
    
    return loss

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
        if feature:
            return self.fc(x), x
        else:
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


class MM_F_2D_NICO(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 ):
        
        super(MM_F_2D_NICO, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        
        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 128),
            self.Conv_bn_ReLU(128, 256, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )
        self.fc = nn.Sequential(
                self.Fc_bn_ReLU(1024, 1024),
                self.Fc_bn_ReLU(1024, 1024),
                self.Fc_bn_ReLU(1024, 2048),
                nn.Linear(2048, self.num_classes), )
    
    def forward(self, x, feature=0):
        x = self.get_feature(x)
        if feature:
            return self.fc(x), x
        else:
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

class get_dataset_2D(data.Dataset):
    def __init__(self,
                 root=None,
                 fold='train',
                 args=None,
                 transform=None):
        self.root = root
        self.args = args
        self.image_path_list = []
        self.u = []
        self.us = []
        self.train_u = []
        self.train_us = []
        self.y = []
        self.env = []
        self.transform = transform
        fold_map = {'train': 0, 'val': 1, 'test': 2}
        if args.dataset == 'NICO':
            if self.root is None:
                self.root = '/home/botong/Dataset/'
            
            workbook = xlrd.open_workbook(r"/home/botong/Dataset/NICO_dataset.xls")
            if args.dataset_type == 'NICO_0':
                sheet = workbook.sheet_by_index(0)
            elif args.dataset_type == 'NICO_1':
                sheet = workbook.sheet_by_index(1)
            elif args.dataset_type == 'NICO_2':
                sheet = workbook.sheet_by_index(2)
            elif args.dataset_type == 'NICO_3':
                sheet = workbook.sheet_by_index(3)
            elif args.dataset_type == 'NICO_4':
                sheet = workbook.sheet_by_index(4)
            elif args.dataset_type == 'NICO_5':
                sheet = workbook.sheet_by_index(5)
            elif args.dataset_type == 'NICO_6':
                sheet = workbook.sheet_by_index(6)
            elif args.dataset_type == 'NICO_7':
                sheet = workbook.sheet_by_index(7)
            elif args.dataset_type == 'NICO_8':
                sheet = workbook.sheet_by_index(8)
            elif args.dataset_type == 'NICO_9':
                sheet = workbook.sheet_by_index(9)
            else:
                sheet = workbook.sheet_by_index(10)
            for rows in range(1, sheet.nrows):
                if sheet.row_values(rows)[4] == fold_map[fold]:
                    self.image_path_list.append(os.path.join(self.root, sheet.row_values(rows)[0]))
                    self.y.append(sheet.row_values(rows)[1])
                    self.env.append(sheet.row_values(rows)[3])
                    
        
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
               torch.from_numpy(np.array(self.y[index]).astype('int')), \
               torch.from_numpy(np.array(self.env[index]).astype('int'))
    
    def __len__(self):
        return len(self.image_path_list)
    
def train(epoch,
          model,
          optimizer,
          dataloader,
          args):
    RECON_loss = AverageMeter()
    MSE_loss = AverageMeter()
    classify_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0

    for batch_idx, (x, target, env) in enumerate(dataloader):
        if args.cuda:
            x, target, env = x.cuda(), target.cuda(), env.cuda()
        
        model.eval()
        sample_weight = torch.ones(x.size(0), 1).cuda() / x.size(0)
        sample_weight.requires_grad = True
        optim_in = optim.SGD(params=[sample_weight], lr=0.1, weight_decay=0.0001)
        try:
            for inner_iter in range(args.epoch_in):
                pred_y, fea = model(x, feature=1)
                loss = lossb(fea, sample_weight, fea.size(1))
                loss.backward()
                optim_in.step()
                sample_weight = torch.from_numpy(projection_simplex_sort(sample_weight.detach().cpu().numpy()[:, 0]).reshape(x.size(0), 1)).cuda()
                sample_weight.requires_grad = True
                #print(sample_weight.sum())
        except:
            pass

        sample_weight.requires_grad = False
        model.train()
        pred_y, fea = model(x, feature=1)
        loss = torch.FloatTensor([0.0]).cuda()
        
        for idx in range(x.size(0)):
            loss = torch.add(loss, sample_weight[idx] * F.cross_entropy(pred_y[idx, :].unsqueeze(0), target[idx].unsqueeze(0)) +
                             args.alpha * F.mse_loss(fea[idx,:], fea[idx,:].sign()) / x.size(0))
            MSE_loss.update(args.alpha * F.mse_loss(fea[idx,:], fea[idx,:].sign()).item() / x.size(0), 1)
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
                        MSE_loss.avg,
                        classify_loss.avg * args.alpha,
                        all_loss.avg,
                        accuracy.avg * 100))
        
    return accuracy.avg


def evaluate(model, dataloader, args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    for batch_idx, (x, target, env) in enumerate(dataloader):
        if args.cuda:
            x, target = x.cuda(), target.cuda()
        
        pred_y = model(x)
        pred[batch_begin:batch_begin + x.size(0), :] = pred_y.detach().cpu()
        batch_begin = batch_begin + x.size(0)
        
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
    return pred, accuracy.avg


def evaluate_only(model, dataloader, args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    label = np.zeros((dataloader.dataset.__len__(),))
    batch_begin = 0
    for batch_idx, (x, target, _,_) in enumerate(dataloader):
        if args.cuda:
            x, target = x.cuda(), target.cuda()
        
        pred_y = model(x)
        pred[batch_begin:batch_begin + x.size(0), :] = pred_y.detach().cpu().numpy()
        label[batch_begin:batch_begin + x.size(0), ] = target.detach().cpu().numpy()
        batch_begin = batch_begin + x.size(0)
        
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
    return pred, label, accuracy.avg

def main():
    args = get_opt()
    args = make_dirs(args)
    logger = get_logger(args)
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    train_loader = DataLoader(get_dataset_2D(args=args, fold='train',
                                                      transform=transforms.Compose([
                                                          transforms.RandomResizedCrop((256, 256)),
                                                          transforms.RandomHorizontalFlip(p=0.5),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                      ])),
                                       batch_size=args.batch_size,
                                       drop_last=True,
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
                              num_workers=1,
                              pin_memory=True)

    val_loader = DataLoaderX(get_dataset_2D(args=args, fold='val',
                                             transform=transforms.Compose([
                                                 transforms.Resize((256, 256)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                             ])),
                              batch_size=args.test_batch_size,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True)
    
    if 'NICO' in args.dataset:
        model = MM_F_2D_NICO(in_channel=args.in_channel,
                        u_dim=args.u_dim,
                        us_dim=args.us_dim,
                        num_classes=args.num_classes,
                        is_use_u=args.is_use_u
                        ).cuda()
    else:
        if args.model == 'MM_F_f':
            model = MM_F_ff_2D(in_channel=args.in_channel,
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
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    
    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay, args.lr_controler)
        _ = train(epoch, model, optimizer, train_loader, args)
        pred_val, val_acc = evaluate(model, val_loader, args)
        pred_test, test_acc = evaluate(model, test_loader, args)
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
    import sys
    xlsx_name = '%s_Data_%s_%s_model_%s_u_%d_fold_%d_env_%d.xls' % \
                (os.path.basename(sys.argv[0][:-3]), args.dataset, args.dataset_type, args.model, args.is_use_u,
                 args.fold, args.env_num)
    save_results_as_xlsx('./results/', xlsx_name, best_acc, best_acc_ep, auc=None, args=args)
    logger.info('*' * 50)
    logger.info('*' * 50)


if __name__ =='__main__':
    main()

