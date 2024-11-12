import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from .LaCIM_model import MM_F_2D_Cmnist, MM_F_2D_NICO, MM_F_3D_AD
from .utils_baseline import *


def pretty(vector):
    if type(vector) is list:
        vlist = vector
    elif type(vector) is np.ndarray:
        vlist = vector.reshape(-1).tolist()
    else:
        vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.4f}".format(vi) for vi in vlist) + "]"


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.weight_init()

    def weight_init(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class WeightedLasso:
    def __init__(self, X, y, weight, lam, model, lr):

        self.lr = lr
        self.model = model
        self.X = X
        self.y = y
        self.weight = weight.reshape(-1, 1)
        self.loss = nn.CrossEntropyLoss()
        self.lam = lam
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, ep):
        # self.model.weight_init()
        epochs = ep
        all_loss = AverageMeter()
        l1_loss = AverageMeter()
        for epoch in range(1, 1+epochs):
            adjust_learning_rate(self.optimizer, epoch, self.lr, 0.5, 200)
            self.optimizer.zero_grad()
            if self.X.size(0) == 1 or self.X.size(0) == 0:
                break
            pred = self.model(self.X)
            regularization_l1 = 0
            for param in self.model.parameters():
                regularization_l1 += torch.sum(abs(param))
            # todo zhege loss xuyao chongixn queren
            loss = self.loss(pred, self.y)  # + self.lam * regularization_l1
            # print("pred: ", pred.size(), pred)
            # print("self.y: ", self.y.size(), self.y)

            # print("X, ", self.X.size())
            # print("all loss: ", all_loss)
            # print("loss: , reg loss: ", loss, self.lam * regularization_l1)
            all_loss.update(loss.item(), self.X.size(0))
            l1_loss.update(regularization_l1.item())
            if epoch % 500 == 0:
                print("WeightedLasso loss: {:.4f}, l1 loss : {:.4f}".format(
                    all_loss.avg, l1_loss.avg))
            loss.backward(retain_graph=True)
            self.optimizer.step()
        return self.model


class McModel:
    def __init__(self, num_classes, args=None):
        self.args = args
        if 'NICO' in self.args.dataset:
            self.frontmodel = MM_F_2D_NICO(in_channel=self.args.in_channel,
                                           u_dim=self.args.u_dim,
                                           us_dim=self.args.us_dim,
                                           num_classes=self.args.num_classes,
                                           is_use_u=self.args.is_use_u,
                                           zs_dim=self.args.zs_dim,
                                           ).cuda()
        if 'mnist' in self.args.dataset:
            self.frontmodel = MM_F_2D_Cmnist(in_channel=self.args.in_channel,
                                             u_dim=self.args.u_dim,
                                             us_dim=self.args.us_dim,
                                             num_classes=self.args.num_classes,
                                             is_use_u=self.args.is_use_u,
                                             zs_dim=self.args.zs_dim,
                                             ).cuda()
        if 'AD' in self.args.dataset:
            self.frontmodel = MM_F_3D_AD(in_channel=self.args.in_channel,
                                         u_dim=self.args.u_dim,
                                         us_dim=self.args.us_dim,
                                         num_classes=self.args.num_classes,
                                         is_use_u=self.args.is_use_u,
                                         zs_dim=self.args.zs_dim,
                                         ).cuda()
        self.num_classes = num_classes
        self.X = None
        self.y = None
        self.center = None
        self.bias = None
        self.domain = None
        self.weights = None
        self.shape_0 = self.args.batch_size
        if 'NICO' in self.args.dataset:
            self.shape_1 = (3, 256, 256)
        if 'mnist' in self.args.dataset:
            self.shape_1 = (3, 28, 28)
        if 'AD' in self.args.dataset:
            self.shape_1 = (48, 48, 48)

    def ols(self):
        for i in range(self.num_classes):
            index = torch.where(self.domain == i)[0]
            # tempx = (self.X[index, :]).reshape(-1, self.shape_1)
            # tempy = (self.y[index, :]).reshape(-1, 1)
            tempx = (self.X[index, :])  # .reshape(-1, self.shape_1)
            tempy = (self.y[index])  # .reshape(-1, 1)
            clf = WeightedLasso(tempx, tempy, self.weights,
                                1e-3, self.frontmodel, self.args.lr)
            self.model_list.append(clf.train(ep=self.args.epochs))

    def cluster(self, weight, past_domains, X, y, reuse=False):
        # self.center = torch.tensor(np.zeros(
        #     (self.num_classes, self.shape_1), dtype=np.float32)).cuda()  # in lacim not on using
        # in lacim not on using
        self.bias = torch.tensor(
            np.zeros(self.num_classes, dtype=np.float32)).cuda()
        self.model_list = []
        self.X = X
        self.y = y

        if past_domains is None or not reuse:
            self.domain = torch.tensor(np.random.randint(
                0, self.num_classes, self.shape_0)).cuda()
        else:
            self.domain = past_domains
        assert self.domain.shape[0] == self.shape_0
        self.weights = weight

        iter = 0
        end_flag = False
        delta_threshold = 250
        # delta = 1e4

        while not end_flag:
            # print("iter", iter)
            # print("end_flag: ", end_flag)
            iter += 1
            # print("iter", iter)
            self.ols()
            ols_error = []
            for i in range(self.num_classes):
                # coef = self.center[i].reshape(-1, 1)
                # error = torch.abs(torch.mm(self.X, coef) + self.bias[i] - self.y)
                # print(self.model_list[i](self.X).size())
                # print(self.y.size())
                error = torch.abs(torch.argmax(self.model_list[i](
                    self.X), dim=1) - self.y).reshape((self.shape_0, 1))  # TODO shi yi shi
                assert error.shape == (self.shape_0, 1)
                ols_error.append(error)
            ols_error = torch.stack(ols_error, dim=0).reshape(
                self.num_classes, self.shape_0)
            new_domain = torch.argmin(ols_error, dim=0)
            assert new_domain.shape[0] == self.shape_0
            diff = self.domain.reshape(-1, 1) - new_domain.reshape(-1, 1)
            diff[diff != 0] = 1
            delta = torch.sum(diff)
            # print("delta: ", delta)
            if iter % 10 == 9:
                pass
                # print("*-* "*20)
                # print("Iter %d | Delta = %d" % (iter, delta))
                # print("*-* "*20)
            if delta <= delta_threshold:
                end_flag = True
                # print("end flag: ", end_flag)
                print("Iter %d | Delta = %d" % (iter, delta))

            self.domain = new_domain
            # print("Ending!!!9")
        # environments = []
        # for i in range(self.num_classes):
        #     index = torch.where(self.domain == i)[0]
        #     tempx = (self.X[index, :]).reshape(-1, self.shape_1)
        #     tempy = (self.y[index, :]).reshape(-1, 1)
        #     environments.append([tempx, tempy])
        return self.domain


def comobine_envs(envs):
    X = []
    y = []
    for env in envs:
        X.append(env[0])
        y.append(env[1])
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)
    return X.reshape(-1, X.shape[1]), y.reshape(-1, 1)
