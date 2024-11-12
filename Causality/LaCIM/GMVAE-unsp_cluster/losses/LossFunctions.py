# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Loss functions used for training our model

"""
import math
from tkinter import Y
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics.cluster import (entropy,
                                     mutual_info_score,
                                     normalized_mutual_info_score,
                                     adjusted_mutual_info_score)


def MI(x, y): return mutual_info_score(x, y)


def NMI(x, y): return normalized_mutual_info_score(
    x, y, average_method='arithmetic')


def AMI(x, y): return adjusted_mutual_info_score(
    x, y, average_method='arithmetic')


class LossFunctions:
    eps = 1e-8

    def mean_squared_error(self, real, predictions):
        """Mean Squared Error between the true and predicted outputs
           loss = (1/n)*Σ(real - predicted)^2

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        loss = (real - predictions).pow(2)
        return loss.sum(-1).mean()

    def reconstruction_loss(self, real, predicted, rec_type='mse'):
        """Reconstruction loss between the true and predicted outputs
           mse = (1/n)*Σ(real - predicted)^2
           bce = (1/n) * -Σ(real*log(predicted) + (1 - real)*log(1 - predicted))

        Args:
            real: (array) corresponding array containing the true labels
            predictions: (array) corresponding array containing the predicted labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        if rec_type == 'mse':
            loss = (real - predicted).pow(2)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none')
        else:
            raise "invalid loss function... try bce or mse..."
        return loss.sum(-1).mean()

    def log_normal(self, x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
           log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
           x: (array) corresponding array containing the input
           mu: (array) corresponding array containing the mean 
           var: (array) corresponding array containing the variance

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        """Variational loss when using labeled data without considering reconstruction loss
           loss = log q(z|x,y) - log p(z) - log p(y)

        Args:
           z: (array) array containing the gaussian latent variable
           z_mu: (array) array containing the mean of the inference model
           z_var: (array) array containing the variance of the inference model
           z_mu_prior: (array) array containing the prior mean of the generative model
           z_var_prior: (array) array containing the prior variance of the generative mode

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        loss = self.log_normal(z, z_mu, z_var) - \
            self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def entropy(self, logits, targets):
        """Entropy loss
            loss = (1/n) * -Σ targets*log(predicted)

        Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                  of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))

    def mutual_loss(self, x, y):
        """
        mutual information loss for balance
        """
        # print("x", x)
        # print(x.size())
        # print("x", x.argmax(dim=1, keepdim=False))
        # print(x.argmax(dim=1, keepdim=False).size()) np.array(target_ppa).astype('int')
        return torch.from_numpy(np.array(MI(x.argmax(dim=1, keepdim=False).detach().cpu().numpy(),
                                            y.detach().cpu().numpy())).astype("float32")).cuda()


class MINE(nn.Module):
    def __init__(self, data_dim=64, hidden_size=10):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2 * data_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x, y):
        x = torch.unsqueeze(x, dim=1)
        y = torch.unsqueeze(y, dim=1)
        # print("x", x)
        # print("y", y)
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x, ], dim=0)
        idx = torch.randperm(batch_size)

        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        # print("x", tiled_x.size())
        # print("y", concat_y.size())
        inputs = torch.cat([tiled_x, concat_y], dim=1).float()
        # print("input", inputs.size())
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = -(torch.mean(pred_xy) -
                 torch.log(torch.mean(torch.exp(pred_x_y))))  # np.log2(np.exp(1)) *
        # compute loss, you'd better scale exp to bit
        return loss

# 二分类E


def _mask_2(labels, E_classification, num_classes):
    if num_classes == 2:
        return torch.nn.functional.one_hot(labels, num_classes=num_classes).bool()
    else:  # multi-classification for E
        return torch.nn.functional.one_hot(labels*E_classification, num_classes=num_classes).bool()


def _mine_2(logits, labels):
    batch_size, classes = logits.shape
    E_classification = logits.argmax(dim=1, keepdim=True)
    joints = torch.masked_select(
        logits, _mask_2(labels, E_classification, classes))
    t = torch.mean(joints)
    et = torch.logsumexp(logits, dim=(0, 1)) - np.log(batch_size * classes)
    return t, et, joints, logits.flatten()


def _regularized_loss(mi, reg):
    loss = mi - reg
    with torch.no_grad():
        mi_loss = mi - loss
    return loss + mi_loss


def mine_2(logits, labels):
    t, et, joints, marginals = _mine_2(logits, labels)
    return et - t, joints, marginals


# 多分类E
def _mask_M(labels, E_classification, num_classes):
    if num_classes == 3:
        return torch.nn.functional.one_hot(labels, num_classes=num_classes).bool()
    else:  # multi-classification for E
        return torch.nn.functional.one_hot(labels*E_classification, num_classes=num_classes).bool()


def _mine_M(logits, labels):
    batch_size, classes = logits.shape
    E_classification = logits.argmax(dim=1, keepdim=True)
    joints = torch.masked_select(
        logits, _mask_M(labels, E_classification, classes))
    t = torch.mean(joints)
    et = torch.logsumexp(logits, dim=(0, 1)) - np.log(batch_size * classes)
    return t, et, joints, logits.flatten()


def _regularized_loss(mi, reg):
    loss = mi - reg
    with torch.no_grad():
        mi_loss = mi - loss
    return loss + mi_loss


def mine_M(logits, labels):
    t, et, joints, marginals = _mine_M(logits, labels)
    return et - t, joints, marginals
#outputs = net(inputs)
#loss, joints, marginals = criterion(outputs, labels)

#  # MiNE方法主要用于模型的训练阶段
#     for epoch in tqdm(range(n_epoch)):
#         x_sample = gen_x() # 调用gen_x()函数生成样本x_Sample。X_sample代表X的边缘分布P(X)
#         y_sample = gen_y(x_sample) # 将生成的×_sample样本放到gen_x()函数中，生成样本y_sample。y_sample代表条件分布P(Y|X)。
#         y_shuffle = np.random.permutation(y_sample) # )将 y_sample按照批次维度打乱顺序得到y_shuffle，y_shuffle是Y的经验分布，近似于Y的边缘分布P(Y)。
#         # 转化为张量
#         x_sample = torch.from_numpy(x_sample).type(torch.FloatTensor)
#         y_sample = torch.from_numpy(y_sample).type(torch.FloatTensor)
#         y_shuffle = torch.from_numpy(y_shuffle).type(torch.FloatTensor)

#         model.zero_grad()
#         pred_xy = model(x_sample, y_sample)  # 式(8-49）中的第一项联合分布的期望:将x_sample和y_sample放到模型中，得到联合概率（P(X,Y)=P(Y|X)P(X)）关于神经网络的期望值pred_xy。
#         pred_x_y = model(x_sample, y_shuffle)  # 式(8-49)中的第二项边缘分布的期望:将x_sample和y_shuffle放到模型中，得到边缘概率关于神经网络的期望值pred_x_y 。

#         ret = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))) # 将pred_xy和pred_x_y代入式（8-49）中，得到互信息ret。
#         loss = - ret  # 最大化互信息：在训练过程中，因为需要将模型权重向着互信息最大的方向优化，所以对互信息取反，得到最终的loss值。
#         plot_loss.append(loss.data)  # 收集损失值
#         loss.backward()  # 反向传播：在得到loss值之后，便可以进行反向传播并调用优化器进行模型优化。
#         optimizer.step()  # 调用优化器
#     plot_y = np.array(plot_loss).reshape(-1, )  # 可视化
#     plt.plot(np.arange(len(plot_loss)), -plot_y, 'r') # 直接将|oss值取反，得到最大化互信息的值。
#     plt.show()
