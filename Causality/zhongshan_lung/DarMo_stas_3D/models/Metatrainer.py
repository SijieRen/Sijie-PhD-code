import torch
from torch import nn
from torch.nn import functional as F

from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
import math
from torch.nn import Parameter
import numpy as np
import pickle
from .ma_learning import GraphConvolution, gen_A, gen_adj
from .Basenet import Basenet
from .feature_learning import BasicBlock, conv3x3, ResNetbasic, BasicBlock2, ResNetbasic2

from .vit_pytorch_main.vit_pytorch import ViT, ViT_d

# models

Tensor = TypeVar('torch.tensor')
args = parse_opt()

class ViT_d_Enc(nn.Module):
    def __init__(self,
                 ViT,
                image_size = 256,
                patch_size = 32,
                num_classes = 32,
                dim = 32,# ori 1024 corresponding to dim-z=32
                depth = 6,
                heads = 16,
                mlp_dim = 512,# ori 2048
                dropout = 0.1,
                emb_dropout = 0.1):
        super(ViT_d_Enc).__init__()
        self.model = ViT_d(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            dropout = dropout,
            emb_dropout = emb_dropout
        )

    def forward(self,x, d_id):
        x = self.model(x, d_id)

        return x

class ViT_Enc(nn.Module):
    def __init__(self,
                 ViT,
                image_size = 256,
                patch_size = 32,
                num_classes = 32,
                dim = 32,# ori 1024 coresponding to dim-z=32
                depth = 6,
                heads = 16,
                mlp_dim = 512,# ori 2048
                dropout = 0.1,
                emb_dropout = 0.1):
        super(ViT_Enc).__init__()
        self.model = ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            dropout = dropout,
            emb_dropout = emb_dropout
        )

    def forward(self,x, d_id):
        x = self.model(x, d_id)

        return x

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super(ResizeConv2d).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class ResNet18Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet18Enc, self).__init__()
        self.z_dim = z_dim
        #self.ResNet34 = models.resnet34(pretrained=True)
        self.ResNet34 = ResNetbasic(BasicBlock, [3, 4, 6, 3], self.z_dim)
        #self.num_feature = self.ResNet34.fc.in_features
        #self.ResNet34.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

    def forward(self, x):
        x = self.ResNet34(x)

        return x


class ResNet18Enc2(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet18Enc2, self).__init__()
        self.z_dim = z_dim
        #self.ResNet34 = models.resnet34(pretrained=True)
        self.ResNet34_2 = ResNetbasic2(BasicBlock2, [3, 4, 6, 3], self.z_dim)
        #self.num_feature = self.ResNet34.fc.in_features
        #self.ResNet34.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

    def forward(self, x, d_id):
        x = self.ResNet34_2(x, d_id)

        return x

class DarMo(Basenet):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 gcndir: str,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(DarMo, self).__init__()


        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        if args.if_transformer:
            self.Resnetencoder = ViT_Enc()
        else:
            self.Resnetencoder = ResNet18Enc()

        if args.if_transformer:
            self.Resnetencoder2 = ViT_d_Enc()
        else:
            self.Resnetencoder2 = ResNet18Enc2()

        self.fc_d = nn.Linear(hidden_dims[-1]*49, latent_dim)
        self.fc_mu = nn.Linear(hidden_dims[-1]*49, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*49, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim*2, hidden_dims[-1] * 49)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        self.cls_fc = nn.Linear(256, 2)
        self.gc1 = GraphConvolution(12, 128)
        self.gc2 = GraphConvolution(128, 224)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(12, 0.4, str(gcndir))
        self.A = Parameter(torch.from_numpy(_adj).float())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.Resnetencoder(input)

        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 7, 7)
        result = self.decoder(result)

        result = self.final_layer(result)


        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, d_id: int, state: str, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)


        cls = self.cls_fc(z)
        #d
        if 'train' in state:
            d = self.Resnetencoder2(input, d_id)


            d = torch.flatten(d, start_dim=1)
            d = self.fc_d(d)

            # gcn
            inp = torch.eye(12).cuda()
            adj = gen_adj(self.A).detach()
            gcn = self.gc1(inp, adj)
            gcn = self.relu(gcn)
            gcn = self.gc2(gcn, adj)

            gcn = gcn.transpose(0, 1)
            # gcn_x = torch.matmul(z[:,:64], gcn)
            gcn_x = torch.matmul(z[:, :224], gcn)

        if 'val' in state:
            d = z
            gcn_x = 0
        return  [self.decode(torch.cat((z, d), 1)), input, mu, log_var, gcn_x, cls]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        loss_l1 = torch.nn.SmoothL1Loss()
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = loss_l1(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


