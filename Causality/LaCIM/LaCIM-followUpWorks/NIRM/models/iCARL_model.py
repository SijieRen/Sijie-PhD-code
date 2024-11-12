# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from torchvision import transforms
from models import *
import torch.nn as nn
from torch.autograd import Function, grad
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from torch import distributions as dist


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, type='3d'):
        super(UnFlatten, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == '3d':
            return input.view(input.size(0), input.size(1), 1, 1, 1)
        else:
            return input.view(input.size(0), input.size(1), 1, 1)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class iCARL_2D_MNIST(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=0,
                 args=None,
                 prior=None,
                 decoder=None,
                 encoder=None,
                 device='cpu',
                 anneal=False,):
        super(iCARL_2D_MNIST, self).__init__()

        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.args = args

        self.latent_dim = zs_dim

        self.aux_dim = u_dim
        self.anneal_params = anneal

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        self.Enc_x_g = self.get_Enc_x_g()
        self.Enc_u_g = self.get_Enc_u_g()
        self.Enc_x_logv = self.get_Enc_x_logv()
        self.Enc_u_logv = self.get_Enc_u_logv()

        self.Dec_y = self.get_Dec_y()

        self.z_u = nn.Sequential(
            nn.Linear(32, self.latent_dim))
        self.z_xu_g = nn.Sequential(
            self.Fc_bn_ReLU(288, 64),
            nn.Linear(64, self.latent_dim))
        self.z_xu_logv = nn.Sequential(
            self.Fc_bn_ReLU(288, 64),
            nn.Linear(64, self.latent_dim))

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = self.get_Enc_u()
        # decoder params
        self.decoder_var = .01 * torch.ones(1).to(device)
        self.f = self.get_Dec_x()
        # encoder params

        self._training_hyperparams = [1., 1., 1., 1., 1]

    def encoder_params(self, x, u):
        # xu = torch.cat((x, u), 1)
        g = self.encode_xu_g(x, u)
        g = self.z_xu_g(g)
        logv = self.encode_xu_logv(x, u)
        logv = self.z_xu_logv(logv)
        return g, logv.exp()

    def decoder_params(self, z):
        f = self.f(z)
        return f[:, :, 2:30, 2:30].contiguous(), self.decoder_var

    def prior_params(self, u):
        logl = self.logl(u)
        logl = self.z_u(logl)
        return self.prior_mean, logl.exp()

    def forward(self, x, u, y):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        pred = self.Dec_y(z)
        return decoder_params, encoder_params, z, prior_params, pred

    def elbo_sm(self, x, u):
        decoder_params, (g, v), z, prior_params = self.forward(x, u)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder_dist.log_pdf(z, g, v)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder_dist.log_pdf(z.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                   v.view(1, M, self.latent_dim), reduce=False)
            log_qz = torch.logsumexp(log_qz_tmp.sum(
                dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1,
                        keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z

        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False

    def get_pred_y(self, x, u):
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        pred_y = self.Dec_y(z)
        return pred_y

    # def encode_prior(self, x, env_idx):
    #     temp = env_idx * torch.ones(x.size()[0], 1)
    #     temp = temp.long().cuda()
    #     y_onehot = torch.FloatTensor(x.size()[0], self.args.env_num).cuda()
    #     y_onehot.zero_()
    #     y_onehot.scatter_(1, temp, 1)
    #     u = self.Enc_u_prior(y_onehot)
    #     # us = self.Enc_us_prior(us)
    #     # concat = torch.cat([u, us], dim=1)
    #     return self.mean_zs_prior(u), self.sigma_zs_prior(u)

    def encode_xu_g(self, x, u):
        x = self.Enc_x_g(x)
        u = self.Enc_u_logv(u)
        concat = torch.cat([x, u], dim=1)
        return concat

    def encode_xu_logv(self, x, u):
        x = self.Enc_x_logv(x)
        u = self.Enc_u_logv(u)
        concat = torch.cat([x, u], dim=1)
        return concat

    # def decode_x(self, zs):
    #     return self.Dec_x(zs)

    # def decode_y(self, s):
    #     return self.Dec_y(s)

    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     if torch.cuda.is_available():
    #         eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     else:
    #         eps = torch.FloatTensor(std.size()).normal_()
    #     return eps.mul(std).add_(mu)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 16),
            self.Fc_bn_ReLU(16, 32)
        )

    def get_Enc_x_g(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u_g(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 16),
            self.Fc_bn_ReLU(16, 32)
        )

    def get_Enc_x_logv(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u_logv(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 16),
            self.Fc_bn_ReLU(16, 32)
        )

    # def get_Enc_us(self):
    #     return nn.Sequential(
    #         self.Fc_bn_ReLU(self.us_dim, 128),
    #         self.Fc_bn_ReLU(128, 256),
    #         self.Fc_bn_ReLU(256, 512),
    #     )
    def T_NN(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.latent_dim, self.latent_dim),
            self.Fc_bn_ReLU(self.latent_dim, self.latent_dim)
        )
    def lambda_NN(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 16),
            self.Fc_bn_ReLU(16, self.latent_dim)
        )
    def lambda_f(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.latent_dim, 128),
            self.Fc_bn_ReLU(128, self.latent_dim)
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


class IVAE_2D_NICO(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=0,
                 args=None,
                 prior=None,
                 decoder=None,
                 encoder=None,
                 device='cpu',
                 anneal=False,):
        super(IVAE_2D_NICO, self).__init__()

        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.args = args

        self.latent_dim = zs_dim

        self.aux_dim = u_dim
        self.anneal_params = anneal

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        self.Enc_x_g = self.get_Enc_x_g()
        self.Enc_u_g = self.get_Enc_u_g()
        self.Enc_x_logv = self.get_Enc_x_logv()
        self.Enc_u_logv = self.get_Enc_u_logv()

        self.Dec_y = self.get_Dec_y()

        self.z_u = nn.Sequential(
            nn.Linear(256, self.latent_dim))
        self.z_xu_g = nn.Sequential(
            self.Fc_bn_ReLU(1280, 512),
            nn.Linear(512, self.latent_dim))
        self.z_xu_logv = nn.Sequential(
            self.Fc_bn_ReLU(1280, 512),
            nn.Linear(512, self.latent_dim))

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = self.get_Enc_u()
        # decoder params
        self.decoder_var = .01 * torch.ones(1).to(device)
        self.f = self.get_Dec_x()
        # encoder params

        self._training_hyperparams = [1., 1., 1., 1., 1]

    def encoder_params(self, x, u):
        # xu = torch.cat((x, u), 1)
        g = self.encode_xu_g(x, u)
        g = self.z_xu_g(g)
        logv = self.encode_xu_logv(x, u)
        logv = self.z_xu_logv(logv)
        return g, logv.exp()

    def decoder_params(self, z):
        f = self.f(z)
        return f, self.decoder_var

    def prior_params(self, u):
        logl = self.logl(u)
        logl = self.z_u(logl)
        return self.prior_mean, logl.exp()

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        pred = self.Dec_y(z)
        return decoder_params, encoder_params, z, prior_params, pred

    def elbo(self, x, u):
        decoder_params, (g, v), z, prior_params = self.forward(x, u)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder_dist.log_pdf(z, g, v)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder_dist.log_pdf(z.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                   v.view(1, M, self.latent_dim), reduce=False)
            log_qz = torch.logsumexp(log_qz_tmp.sum(
                dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1,
                        keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z

        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False

    def get_pred_y(self, x, u):
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        pred_y = self.Dec_y(z)
        return pred_y

    # def encode_prior(self, x, env_idx):
    #     temp = env_idx * torch.ones(x.size()[0], 1)
    #     temp = temp.long().cuda()
    #     y_onehot = torch.FloatTensor(x.size()[0], self.args.env_num).cuda()
    #     y_onehot.zero_()
    #     y_onehot.scatter_(1, temp, 1)
    #     u = self.Enc_u_prior(y_onehot)
    #     # us = self.Enc_us_prior(us)
    #     # concat = torch.cat([u, us], dim=1)
    #     return self.mean_zs_prior(u), self.sigma_zs_prior(u)

    def encode_xu_g(self, x, u):
        x = self.Enc_x_g(x)
        u = self.Enc_u_logv(u)
        concat = torch.cat([x, u], dim=1)
        return concat

    def encode_xu_logv(self, x, u):
        x = self.Enc_x_logv(x)
        u = self.Enc_u_logv(u)
        concat = torch.cat([x, u], dim=1)
        return concat

    # def decode_x(self, zs):
    #     return self.Dec_x(zs)

    # def decode_y(self, s):
    #     return self.Dec_y(s)

    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     if torch.cuda.is_available():
    #         eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     else:
    #         eps = torch.FloatTensor(std.size()).normal_()
    #     return eps.mul(std).add_(mu)

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
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            # self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_x_g(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
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

    def get_Enc_u_g(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_x_logv(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
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

    def get_Enc_u_logv(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    # def get_Enc_us(self):
    #     return nn.Sequential(
    #         self.Fc_bn_ReLU(self.us_dim, 128),
    #         self.Fc_bn_ReLU(128, 256),
    #         self.Fc_bn_ReLU(256, 512),
    #     )

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


class IVAE_3D_AD(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=0,
                 args=None,
                 prior=None,
                 decoder=None,
                 encoder=None,
                 device='cpu',
                 anneal=False,):
        super(IVAE_3D_AD, self).__init__()

        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.args = args

        self.latent_dim = zs_dim

        self.aux_dim = u_dim
        self.anneal_params = anneal

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        self.Enc_x_g = self.get_Enc_x_g()
        self.Enc_u_g = self.get_Enc_u_g()
        self.Enc_x_logv = self.get_Enc_x_logv()
        self.Enc_u_logv = self.get_Enc_u_logv()

        self.Dec_y = self.get_Dec_y()

        self.z_u = nn.Sequential(
            nn.Linear(512, self.latent_dim))
        self.z_xu_g = nn.Sequential(
            self.Fc_bn_ReLU(1536, 512),
            nn.Linear(512, self.latent_dim))
        self.z_xu_logv = nn.Sequential(
            self.Fc_bn_ReLU(1536, 512),
            nn.Linear(512, self.latent_dim))

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = self.get_Enc_u()
        # decoder params
        self.decoder_var = .01 * torch.ones(1).to(device)
        self.f = self.get_Dec_x()
        # encoder params

        self._training_hyperparams = [1., 1., 1., 1., 1]

    def encoder_params(self, x, u):
        # xu = torch.cat((x, u), 1)
        g = self.encode_xu_g(x, u)
        g = self.z_xu_g(g)
        logv = self.encode_xu_logv(x, u)
        logv = self.z_xu_logv(logv)
        return g, logv.exp()

    def decoder_params(self, z):
        f = self.f(z)
        return f, self.decoder_var

    def prior_params(self, u):
        logl = self.logl(u)
        logl = self.z_u(logl)
        return self.prior_mean, logl.exp()

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        pred = self.Dec_y(z)
        return decoder_params, encoder_params, z, prior_params, pred

    def elbo(self, x, u):
        decoder_params, (g, v), z, prior_params = self.forward(x, u)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder_dist.log_pdf(z, g, v)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder_dist.log_pdf(z.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                   v.view(1, M, self.latent_dim), reduce=False)
            log_qz = torch.logsumexp(log_qz_tmp.sum(
                dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1,
                        keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z

        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False

    def get_pred_y(self, x, u):
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        pred_y = self.Dec_y(z)
        return pred_y

    # def encode_prior(self, x, env_idx):
    #     temp = env_idx * torch.ones(x.size()[0], 1)
    #     temp = temp.long().cuda()
    #     y_onehot = torch.FloatTensor(x.size()[0], self.args.env_num).cuda()
    #     y_onehot.zero_()
    #     y_onehot.scatter_(1, temp, 1)
    #     u = self.Enc_u_prior(y_onehot)
    #     # us = self.Enc_us_prior(us)
    #     # concat = torch.cat([u, us], dim=1)
    #     return self.mean_zs_prior(u), self.sigma_zs_prior(u)

    def encode_xu_g(self, x, u):
        x = self.Enc_x_g(x)
        u = self.Enc_u_logv(u)
        concat = torch.cat([x, u], dim=1)
        return concat

    def encode_xu_logv(self, x, u):
        x = self.Enc_x_logv(x)
        u = self.Enc_u_logv(u)
        concat = torch.cat([x, u], dim=1)
        return concat

    # def decode_x(self, zs):
    #     return self.Dec_x(zs)

    # def decode_y(self, s):
    #     return self.Dec_y(s)

    # def reparametrize(self, mu, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     if torch.cuda.is_available():
    #         eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     else:
    #         eps = torch.FloatTensor(std.size()).normal_()
    #     return eps.mul(std).add_(mu)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(),
            nn.Upsample(6),
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
            nn.Conv3d(in_channels=64, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_x_g(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool3d(1),
            Flatten(),
        )

    def get_Enc_u_g(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_x_logv(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool3d(1),
            Flatten(),
        )

    def get_Enc_u_logv(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    # def get_Enc_us(self):
    #     return nn.Sequential(
    #         self.Fc_bn_ReLU(self.us_dim, 128),
    #         self.Fc_bn_ReLU(128, 256),
    #         self.Fc_bn_ReLU(256, 512),
    #     )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        return layer

    def TConv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0,
                      bias=True, groups=1):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                               groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(
            self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        # compute batch cov from its "pseudo sqrt"
        cov = torch.einsum('bik,bjk->bij', v, v)
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False).to(self.device)
        logabsdets = torch.empty(
            batch_size, requires_grad=False).to(self.device)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError(
                'Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError(
                'Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(
                    lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(
                    nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(
                nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass
