"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""
from cgi import print_arguments
from operator import concat
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from networks.Layers import *

# Inference Network


class InferenceNet_MI(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(InferenceNet_MI, self).__init__()
        self.enc_y = torch.nn.ModuleList([
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 64),
            nn.ReLU(),
            # nn.Linear(64, 128),
            # nn.ReLU()
        ])

        self.enc_x = torch.nn.ModuleList([
            nn.Linear(x_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Linear(64, 128),
            # nn.ReLU()
        ])

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList([
            # nn.Linear(x_dim + 128, 512),
            # nn.ReLU(),
            nn.Linear(576, 512),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            nn.ReLU(),
            GumbelSoftmax(512, y_dim)
        ])

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(576 + y_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            nn.ReLU(),
            Gaussian(512, z_dim)
        ])

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def MI_Y_in(self, Y_class):
        num_layers = len(self.enc_y)
        for i, layer in enumerate(self.enc_y):
            # if i == num_layers - 1:
            #     # last layer is gumbel softmax
            #     x = layer(x, temperature, hard)
            # else:
            Y_class = layer(Y_class)
        return Y_class

    def MI_x_in(self, x):
        num_layers = len(self.enc_x)
        for i, layer in enumerate(self.enc_x):
            # if i == num_layers - 1:
            #     # last layer is gumbel softmax
            #     x = layer(x, temperature, hard)
            # else:
            x = layer(x)
        return x

    def forward(self, x, Y_class, temperature=1.0, hard=0):
        #x = Flatten(x)
        Y_class = self.MI_Y_in(Y_class)
        x = self.MI_x_in(x)
        x = torch.cat((x, Y_class), dim=1)
        # q(y|x)
        logits, prob, y = self.qyx(x, temperature, hard)
        #print("logit:", logits)
        #print("prob", prob)
        # q(z|x,y)
        mu, var, z = self.qzxy(x, y)

        output = {'mean': mu, 'var': var, 'gaussian': z,
                  'logits': logits, 'prob_cat': prob, 'categorical': y}
        return output


# Generative Network
class GenerativeNet_MI(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GenerativeNet_MI, self).__init__()

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, x_dim),
            torch.nn.Sigmoid()
        ])

    # p(z|y)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    # p(x|z)
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)

        # p(x|z)
        x_rec = self.pxz(z)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output

# dududu
# GMVAE Network


class GMVAENet_MI(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GMVAENet_MI, self).__init__()

        self.inference = InferenceNet_MI(x_dim, z_dim, y_dim)
        self.generative = GenerativeNet_MI(x_dim, z_dim, y_dim)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, Y_class, temperature=1.0, hard=0):
        x = x.view(x.size(0), -1)
        out_inf = self.inference(x, Y_class, temperature, hard)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative(z, y)

        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output


############## 大模型 ###################
class InferenceNet_big_MI(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(InferenceNet_big_MI, self).__init__()
        self.MI = torch.nn.ModuleList([
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU()
        ])

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList([
            nn.Linear(x_dim + 1024, 1024),
            nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            GumbelSoftmax(512, y_dim)
        ])

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(x_dim + y_dim + 1024, 1024),
            nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            Gaussian(512, z_dim)
        ])

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)
        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    def MI_Y_in(self, Y_class):
        num_layers = len(self.MI)
        for i, layer in enumerate(self.MI):
            # if i == num_layers - 1:
            #     # last layer is gumbel softmax
            #     x = layer(x, temperature, hard)
            # else:
            Y_class = layer(Y_class)
        return Y_class

    # q(z|x,y)
    def qzxy(self, x, y):
        concat = torch.cat((x, y), dim=1)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, Y_class, temperature=1.0, hard=0):
        #x = Flatten(x)
        Y_class = self.MI_Y_in(Y_class)
        x = torch.cat((x, Y_class), dim=1)
        # q(y|x)
        logits, prob, y = self.qyx(x, temperature, hard)
        #print("logit:", logits)
        #print("prob", prob)
        # q(z|x,y)
        mu, var, z = self.qzxy(x, y)

        output = {'mean': mu, 'var': var, 'gaussian': z,
                  'logits': logits, 'prob_cat': prob, 'categorical': y}
        return output


# Generative Network
class GenerativeNet_big_MI(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GenerativeNet_big_MI, self).__init__()

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, x_dim),
            torch.nn.Sigmoid()
        ])

    # p(z|y)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    # p(x|z)
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(self, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)

        # p(x|z)
        x_rec = self.pxz(z)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output

# dududu
# GMVAE Network


class GMVAENet_big_MI(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(GMVAENet_big_MI, self).__init__()

        self.inference = InferenceNet_big_MI(x_dim, z_dim, y_dim)
        self.generative = GenerativeNet_big_MI(x_dim, z_dim, y_dim)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, Y_class, temperature=1.0, hard=0):
        x = x.view(x.size(0), -1)
        out_inf = self.inference(x, Y_class, temperature, hard)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative(z, y)

        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output
