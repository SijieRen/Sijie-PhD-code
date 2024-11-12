import torch
import torch.nn as nn
import math
import torch.optim as optim
import numpy as np
from torch.autograd import grad
from .LaCIM_model import MM_F_2D_Cmnist, MM_F_2D_NICO, MM_F_3D_AD
from .utils_baseline import *


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.weight_init()

    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"

# Feature selection part


class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(
            0.00 * torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma
        self.input_dim = input_dim

    def renew(self):
        self.mu = torch.nn.Parameter(
            0.00 * torch.randn(self.input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z).cuda()
        # print(stochastic_gate.size())
        # print(prev_x.size())
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self


class MpModel:
    def __init__(self,
                 input_dim,
                 output_dim,
                 sigma=1.0,
                 lam=0.1,
                 alpha=0.5,
                 hard_sum=1.0,
                 penalty='Ours',
                 args=None):
        self.args = args
        if 'NICO' in self.args.dataset:
            self.backmodel = MM_F_2D_NICO(in_channel=self.args.in_channel,
                                          u_dim=self.args.u_dim,
                                          us_dim=self.args.us_dim,
                                          num_classes=self.args.num_classes,
                                          is_use_u=self.args.is_use_u,
                                          zs_dim=self.args.zs_dim,
                                          ).cuda()
        if 'mnist' in self.args.dataset:
            self.backmodel = MM_F_2D_Cmnist(in_channel=self.args.in_channel,
                                            u_dim=self.args.u_dim,
                                            us_dim=self.args.us_dim,
                                            num_classes=self.args.num_classes,
                                            is_use_u=self.args.is_use_u,
                                            zs_dim=self.args.zs_dim,
                                            ).cuda()
        if 'AD' in self.args.dataset:
            self.backmodel = MM_F_3D_AD(in_channel=self.args.in_channel,
                                        u_dim=self.args.u_dim,
                                        us_dim=self.args.us_dim,
                                        num_classes=self.args.num_classes,
                                        is_use_u=self.args.is_use_u,
                                        zs_dim=self.args.zs_dim,
                                        ).cuda()
        self.loss = nn.CrossEntropyLoss()
        self.featureSelector = FeatureSelector(input_dim, sigma).cuda()
        self.reg = self.featureSelector.regularizer
        self.lam = lam
        self.mu = self.featureSelector.mu
        self.sigma = self.featureSelector.sigma
        self.alpha = alpha
        self.optimizer = optim.Adam([{'params': self.backmodel.parameters(), 'lr': 1e-3},
                                     {'params': self.mu, 'lr': 3e-4}])
        self.penalty = penalty
        self.hard_sum = hard_sum
        self.input_dim = input_dim
        self.accumulate_mip_penalty = torch.tensor(
            np.zeros(10, dtype=np.float32)).cuda()

    def renew(self):
        self.featureSelector.renew()
        self.mu = self.featureSelector.mu
        # self.backmodel.weight_init()
        self.optimizer = optim.Adam([{'params': self.backmodel.parameters(), 'lr': self.args.lr2},  # 1e-3
                                     {'params': self.mu, 'lr': self.args.lr3}])  # 3e-4

    def combine_envs(self, envs):
        X = []
        y = []
        for env in envs:
            X.append(env[0])
            y.append(env[1])
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        return X.reshape(-1, X.shape[1]), y.reshape(-1, 1)

    def pretrain(self, x, y, pretrain_epoch=100):
        pre_lr = 1e-3
        pre_optimizer = optim.Adam(
            [{'params': self.backmodel.parameters(), 'lr': pre_lr}])
        X = x

        for i in range(pretrain_epoch):
            adjust_learning_rate(self.optimizer, i, pre_lr, 0.5, 100)
            self.optimizer.zero_grad()
            pred = self.backmodel(X)
            loss = self.loss(pred, y)  # .reshape(pred.shape))
            loss.backward()
            pre_optimizer.step()

    def single_forward(self, x, regularizer_flag=False):
        output_x = self.featureSelector(x)
        if regularizer_flag == True:
            x = output_x.clone().detach()
        else:
            x = output_x
        return self.backmodel(x)

    def single_iter_mip(self, x, y):
        # assert type(envs) == list
        num_envs = len(x)
        loss_avg = 0.0
        grad_avg = 0.0
        grad_list = []
        # for [x, y] in envs:
        pred = self.single_forward(x)
        loss = self.loss(pred, y)  # .reshape(pred.shape))
        loss_avg += loss  # /num_envs LaCIM 数据在进入前已经按env分好

        # for [x, y] in envs:
        pred = self.single_forward(x, True)
        loss = self.loss(pred, y)  # .reshape(pred.shape))
        # grad_single = grad(loss, self.backmodel.parameters(), create_graph=True)[
        #     0].reshape(-1)
        # print(grad(loss, self.backmodel.parameters(), create_graph=True)[
        # 0].size())
        # print(self.backmodel.parameters().numel())
        # grad_avg += grad_single  # / num_envs
        # grad_list.append(grad_single)

        # penalty = torch.tensor(
        #     np.zeros(grad(loss, self.backmodel.parameters(), create_graph=True)[
        #         0].reshape(-1).size(), dtype=np.float32)).cuda()
        # for gradient in grad_list:
        #     print(gradient.size(), grad_avg.size())
        #     print(penalty.size())
        #     penalty += (gradient - grad_avg)**2
        # print(self.mu.shape)
        penalty_detach = 0.0
        # penalty_detach = torch.sum(
        #     penalty.reshape(self.mu.shape)*(self.mu+0.5))
        reg = torch.sum(self.reg((self.mu + 0.5) / self.sigma))
        reg = (reg-self.hard_sum)**2
        total_loss = loss_avg  # + self.alpha * (penalty_detach)
        total_loss = total_loss + self.lam * reg
        return total_loss, penalty_detach, self.reg((self.mu + 0.5) / self.sigma)

    def get_gates(self):
        return pretty(self.mu+0.5)

    def get_paras(self):
        return pretty(self.backmodel.linear.weight)

    def train(self, x, y, epochs):
        self.renew()
        self.pretrain(x, y, epochs)
        for epoch in range(1, epochs+1):
            adjust_learning_rate(self.optimizer, epoch,
                                 self.args.lr2, 0.5, 100)
            adjust_learning_rate(self.optimizer, epoch,
                                 self.args.lr3, 0.5, 100)
            self.optimizer.zero_grad()
            loss, penalty, reg = self.single_iter_mip(x, y)
            loss.backward()
            self.optimizer.step()
            if epoch % epochs == 0:
                print("Epoch %d | Loss = %.4f | Gates %s | " %  # Theta = %s" %
                      (epoch, loss, torch.mean(self.mu+0.5)))  # , pretty(self.backmodel.linear.weight)))
        return self.mu + 0.5, reg
