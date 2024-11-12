# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from torchvision import transforms
from models import *
import sys
import torch.nn.functional as F
import torch.nn as nn


class Generative_model_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 ):

        super(Generative_model_2D, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()

        if self.is_use_u == 1 and self.is_use_us == 1:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(2048, 128),
                nn.Linear(128, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(2048, 128),
                nn.Linear(128, self.zs_dim))
        elif self.is_use_u == 1 and self.is_use_us == 0:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1536, 128),
                nn.Linear(128, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1536, 128),
                nn.Linear(128, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 128),
                nn.Linear(128, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 128),
                nn.Linear(128, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(512, 128),
            nn.Linear(128, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(512, 128),
            nn.Linear(128, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)  # using Z+S to reconstruct X~
        pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):]
                            )  # using S to predict Y~
        if feature == 1:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        if self.is_sample:
            z = self.reparametrize(mu, logvar)
            pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        else:
            pred_y = self.Dec_y(mu[:, int(self.zs_dim / 2):])
        return pred_y

    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            # us = self.Enc_us(us)
            concat = torch.cat([x, u], dim=1)
        else:
            # print('not use u and us')
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        concat = u
        return self.mean_zs_prior(concat), self.sigma_zs_prior(concat)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim / 2), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x(self):
        return nn.Sequential(
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
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def TConv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0,
                      bias=True, groups=1):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                               groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class get_dataset_2D_env(data.Dataset):
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

            workbook = xlrd.open_workbook(r"%sNICO_dataset.xls" % self.root)
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
            elif args.dataset_type == 'NICO_10':
                sheet = workbook.sheet_by_index(10)
            elif args.dataset_type == 'NICO_11':
                sheet = workbook.sheet_by_index(11)
            elif args.dataset_type == 'NICO_12':
                sheet = workbook.sheet_by_index(12)
            else:
                sheet = workbook.sheet_by_index(13)
            for rows in range(1, sheet.nrows):
                if sheet.row_values(rows)[4] == fold_map[fold]:
                    self.image_path_list.append(os.path.join(
                        self.root, sheet.row_values(rows)[0]))
                    self.y.append(sheet.row_values(rows)[1])
                    self.env.append(sheet.row_values(rows)[3])
                    self.u.append(sheet.row_values(rows)[3])

        elif args.dataset == 'mnist_2':
            if self.root is None:
                self.root = './data/colored_MNIST_0.02_env_2_0_c_2/%s/' % fold
            else:
                self.root = self.root + '%s/' % fold

            all_classes = os.listdir(self.root)
            for one_class in all_classes:
                for filename in os.listdir(os.path.join(self.root, one_class)):
                    self.u.append(float(filename[-10:-6]))
                    self.env.append(int(filename[-5:-4]))
                    self.image_path_list.append(
                        os.path.join(self.root, one_class, filename))
                    if int(one_class) <= 4:
                        self.y.append(0)
                    else:
                        self.y.append(1)
        elif args.dataset == 'mnist_2_c5':
            if self.root is None:
                self.root = './data/colored_MNIST_0.02_env_2_c_5/%s/' % fold
            else:
                self.root = self.root + '%s/' % fold
            all_classes = os.listdir(self.root)
            for one_class in all_classes:
                for filename in os.listdir(os.path.join(self.root, one_class)):
                    self.u.append(float(filename[-10:-6]))
                    self.env.append(int(filename[-5:-4]))
                    self.image_path_list.append(
                        os.path.join(self.root, one_class, filename))
                    if int(one_class) <= 1:
                        self.y.append(0)
                    elif int(one_class) <= 3:
                        self.y.append(1)
                    elif int(one_class) <= 5:
                        self.y.append(2)
                    elif int(one_class) <= 7:
                        self.y.append(3)
                    else:
                        self.y.append(4)
        elif args.dataset == 'mnist_5_c5':
            if self.root is None:
                self.root = './data/colored_MNIST_0.02_env_2_c_5/%s/' % fold
            else:
                # ./data/colored_MNIST_0.02_env_2_0_c_2/train/
                self.root = self.root + '%s/' % fold
            all_classes = os.listdir(self.root)
            for one_class in all_classes:
                for filename in os.listdir(os.path.join(self.root, one_class)):
                    self.u.append(float(filename[-10:-6]))
                    self.env.append(int(filename[-5:-4]))
                    self.image_path_list.append(
                        os.path.join(self.root, one_class, filename))
                    if int(one_class) <= 1:
                        self.y.append(0)
                    elif int(one_class) <= 3:
                        self.y.append(1)
                    elif int(one_class) <= 5:
                        self.y.append(2)
                    elif int(one_class) <= 7:
                        self.y.append(3)
                    else:
                        self.y.append(4)
        print(self.root)

    def __getitem__(self, index):
        # print(self.image_path_list[index])
        with open(self.image_path_list[index], 'rb') as f:
            img_1 = Image.open(f)
            if '225. cat_mm8_2-min.png' in self.image_path_list[index]:
                img_1 = np.asarray(img_1.convert('RGBA'))[:, :, :3]
                img_1 = Image.fromarray(img_1.astype('uint8'))
            else:
                img_1 = Image.fromarray(np.asarray(
                    img_1.convert('RGB')).astype('uint8'))
        if self.transform is not None:
            img_1 = self.transform(img_1)
        return img_1, \
            torch.from_numpy(np.array(self.y[index]).astype('int')), \
            torch.from_numpy(np.array(self.env[index]).astype('int')), \
            torch.from_numpy(np.array(self.u[index]).astype(
                'float32').reshape((1)))

    def __len__(self):
        return len(self.image_path_list)


def VAE_loss(recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, args):
    """
    pred_y: predicted y
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    q_y_s: prior
    beta: tradeoff params
    """
    eps = 1e-5
    if args.dataset == 'AD':
        BCE = F.binary_cross_entropy(
            recon_x.view(-1, 48 ** 3), x.view(-1, 48 ** 3), reduction='mean')
    elif 'mnist' in args.dataset:
        x = x * 0.5 + 0.5
        BCE = F.binary_cross_entropy(
            recon_x.view(-1, 3 * 28 ** 2), x.view(-1, 3 * 28 ** 2), reduction='mean')
    else:
        x = x * 0.5 + 0.5
        BCE = F.binary_cross_entropy(
            recon_x.view(-1, 3 * 256 ** 2), x.view(-1, 3 * 256 ** 2), reduction='mean')
    # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = KLD_element.mul_(-0.5).mean()
    if args.fix_mu == 1:
        mu_prior = torch.zeros(mu_prior.size()).cuda()
    if args.fix_var == 1:
        logvar_prior = torch.ones(logvar_prior.size()).cuda()
    if args.KLD_type == 1:
        KLD_element = torch.log(logvar_prior.exp() ** 0.5 / logvar.exp() ** 0.5) + \
            0.5 * ((mu - mu_prior).pow(2) + logvar.exp()) / \
            logvar_prior.exp() - 0.5
        KLD = KLD_element.mul_(1).mean()
    else:
        log_p_zs = torch.log(eps + (1 / ((torch.exp(logvar_prior) ** 0.5) * np.sqrt(2 * np.pi))) *
                             torch.exp(-0.5 * ((zs - mu_prior) / (torch.exp(logvar_prior) ** 0.5)) ** 2))
        log_q_zs = torch.log(eps + (1 / ((torch.exp(logvar) ** 0.5) * np.sqrt(2 * np.pi))) *
                             torch.exp(-0.5 * ((zs - mu) / (torch.exp(logvar) ** 0.5)) ** 2))
        KLD = (log_q_zs - log_p_zs).sum() / x.size(0)

    return BCE, KLD


def train(epoch, model, optimizer, dataloader, args):
    all_zs = np.zeros((len(dataloader.dataset), args.zs_dim))
    RECON_loss = AverageMeter()
    KLD_loss = AverageMeter()
    classify_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    model.train()
    for batch_idx, (x, u, us, target) in enumerate(dataloader):
        if args.cuda:
            if 'mnist' in args.dataset:
                x, u, env, target = x.cuda(), target.cuda(), us.cuda(), u.cuda()
            else:
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda()

        # try:
        if args.model == 'VAE' or args.model == 'sVAE' or args.model == 'VAE_f' or args.model == 'sVAE_f':
            # print(u.size(), us.size())
            # args.model = VAE_f LACIM MNIST branch !!!
            _, recon_x, mu, logvar, mu_prior, logvar_prior, zs = model(
                x, u, us, feature=1)
            pred_y = model.get_pred_y(x, u, us)
        elif args.model == 'VAE_old':
            q_y_s, _, _, _, _, _ = model(x, u, us, feature=0)
            pred_y = model.get_pred_y(x, u, us)
            _, recon_x, mu, logvar, mu_prior, logvar_prior, zs = model(
                x, u, us, feature=1)
        else:
            pred_y = model(x, u, us)
        # except:
        #     continue
        if args.model == 'VAE' or args.model == 'sVAE' or args.model == 'VAE_f' or args.model == 'sVAE_f':
            if args.solve == 'IRM':
                recon_loss, kld_loss = VAE_loss(
                    recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, args)
                # cls_loss = F.nll_loss(torch.log(pred_y), target)
                cls_loss = torch.FloatTensor([0.0]).cuda()
                for ss in range(args.env_num):
                    if torch.sum(env == ss) == 0:
                        continue
                    cls_loss = torch.add(cls_loss,
                                         torch.sum(env == ss) * (
                                             F.cross_entropy(pred_y[env == ss, :], target[env == ss]) +
                                             args.alpha * penalty(pred_y[env == ss, :],
                                                                  target[env == ss])))
                    # print('IRM loss:', F.cross_entropy(pred_y[env == ss, :], target[env == ss]), args.alpha * penalty(pred_y[env == ss, :],target[env == ss]))
                cls_loss = cls_loss / pred_y.size(0)
                # print(recon_loss, args.beta * kld_loss, args.gamma * cls_loss, args.beta, args.gamma)
                loss = recon_loss + args.beta * kld_loss + args.gamma * cls_loss
            else:  # args.model=VAE_f args.solve=None LACIM MNIST branch !!!
                recon_loss, kld_loss = VAE_loss(
                    recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, args)
                cls_loss = F.nll_loss(torch.log(pred_y), target)
                loss = recon_loss + args.beta * kld_loss + args.alpha * cls_loss

            all_zs[batch_begin:batch_begin + x.size(0), :] = \
                zs.detach().view(x.size(0), args.zs_dim).cpu().numpy()
            batch_begin = batch_begin + x.size(0)
        elif args.model == 'VAE_old':
            recon_loss, kld_loss = VAE_old_loss(pred_y, recon_x, x, mu, logvar, mu_prior,
                                                logvar_prior, zs, q_y_s, target, args)
            cls_loss = F.nll_loss(torch.log(pred_y), target)
            loss = recon_loss + args.beta * kld_loss + args.alpha * cls_loss

            all_zs[batch_begin:batch_begin + x.size(0), :] = \
                zs.detach().view(x.size(0), args.zs_dim).cpu().numpy()
            batch_begin = batch_begin + x.size(0)
        else:
            loss = F.cross_entropy(pred_y, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' \
                or args.model == 'VAE_f' or args.model == 'sVAE_f':
            RECON_loss.update(recon_loss.item(), x.size(0))
            KLD_loss.update(kld_loss.item(), x.size(0))
            classify_loss.update(cls_loss.item(), x.size(0))
        all_loss.update(loss.item(), x.size(0))
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(),
                        target.detach().cpu().numpy()), x.size(0))

        if batch_idx % 10 == 0:
            args.logger.info(
                'epoch [{}/{}], batch: {}, rec_loss:{:.4f}, kld_loss:{:.4f} cls_loss:{:.4f}, overall_loss:{:.4f},acc:{:.4f}'
                .format(epoch,
                        args.epochs,
                        batch_idx,
                        RECON_loss.avg,
                        KLD_loss.avg * args.beta,
                        classify_loss.avg,
                        all_loss.avg,
                        accuracy.avg * 100))

    if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' or \
            args.model == 'VAE_f' or args.model == 'sVAE_f':
        all_zs = all_zs[:batch_begin]
    args.logger.info(
        'epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, all_loss.avg))

    return all_zs, accuracy.avg


def evaluate(model, dataloader, args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    for batch_idx, (x, u, us, target) in enumerate(dataloader):
        if args.cuda:
            if 'mnist' in args.dataset:
                x, u, us, target = x.cuda(), target.cuda(), us.cuda(), u.cuda()
            else:
                x, u, us, target = x.cuda(), u.cuda(), us.cuda(), target.cuda()

        if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' \
                or args.model == 'VAE_f' or args.model == 'sVAE_f':
            pred_y = model.get_pred_y(x, u, us)
        else:
            pred_y = model(x, u, us)
        pred[batch_begin:batch_begin + x.size(0), :] = pred_y.detach().cpu()
        batch_begin = batch_begin + x.size(0)
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(),
                        target.detach().cpu().numpy()), x.size(0))
    return pred, accuracy.avg


def checkpoint(epoch, save_folder, save_model, is_best=0, other_info=None, logger=None):
    if is_best:
        model_out_tar_path = os.path.join(save_folder, "best_acc.pth.tar")
    else:
        model_out_tar_path = os.path.join(save_folder, "checkpoints.pth.tar")

    torch.save({
        'state_dict': save_model.state_dict(),
        'epoch': epoch,
        'other_info': other_info
    }, model_out_tar_path)
    if logger is not None:
        logger.info("Checkpoint saved to {}".format(model_out_tar_path))
    else:
        print("Checkpoint saved to {}".format(model_out_tar_path))


def adjust_learning_rate(optimizer, epoch, lr, lr_decay, lr_controler):
    for param_group in optimizer.param_groups:
        new_lr = lr * lr_decay ** (epoch // lr_controler)
        param_group['lr'] = lr * lr_decay ** (epoch // lr_controler)
    print('current lr is ', new_lr)


def main():
    args = get_opt()
    args = make_dirs(args)
    logger = get_logger(args)
    args.logger = logger
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    if 'mnist' in args.dataset:
        train_loader = DataLoader(get_dataset_2D_env(root=args.root, args=args, fold='train',
                                                     transform=transforms.Compose([
                                                         transforms.RandomHorizontalFlip(
                                                             p=0.5),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                     ])),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True,
                                  drop_last=True)
        test_loader = DataLoaderX(get_dataset_2D_env(root=args.root, args=args, fold='test',
                                                     transform=transforms.Compose([
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                     ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True,)
        val_loader = None
    model = Generative_model_2D(in_channel=args.in_channel,
                                u_dim=args.u_dim,
                                us_dim=args.us_dim,
                                zs_dim=args.zs_dim,
                                num_classes=args.num_classes,
                                is_use_u=args.is_use_u,
                                is_sample=args.is_sample,
                                ).cuda()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.reg)

    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr,
                             args.lr_decay, args.lr_controler)
        _, _ = train(epoch, model, optimizer, train_loader, args)
        if val_loader is not None:
            pred_val, val_acc = evaluate(model, val_loader, args)
        else:
            pred_val = None
            val_acc = -1
        pred_test, test_acc = evaluate(model, test_loader, args)
        if test_acc >= best_acc:
            best_acc = copy.deepcopy(test_acc)
            best_acc_ep = copy.deepcopy(epoch)
            is_best = 1
        else:
            is_best = 0
        other_info = {
            'pred_val': pred_val,
            'pred_test': pred_test,
        }
        checkpoint(epoch, args.model_save_dir, model,
                   is_best, other_info, logger)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d, val_acc: %0.4f'
                    % (epoch, test_acc, best_acc, best_acc_ep, val_acc))
    logger.info('model save path: %s' % args.model_save_dir)
    logger.info('*' * 50)
    logger.info('*' * 50)


if __name__ == '__main__':
    main()
