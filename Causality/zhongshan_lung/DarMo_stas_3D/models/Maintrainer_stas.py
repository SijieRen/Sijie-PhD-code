import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import tensor as Tensor
import math
from torch.nn import Parameter
import numpy as np
import pickle
from .ma_learning import GraphConvolution, gen_A, gen_adj
from .Basenet import Basenet
from .feature_learning import BasicBlock, ResNetbasic, BasicBlock2, ResNetbasic2
from .vit_pytorch_main.vit_pytorch.vit_3d import ViT
from .vit_pytorch_main.vit_pytorch.vit_3d_d import ViT_d
# from .vit_pytorch_main.vit_pytorch import ViT, ViT_d
import copy


from utils.opts import parse_opt

def get_inplanes():
    return [64, 128, 256, 512]

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
# models

Tensor = TypeVar('torch.tensor')
args = parse_opt()

class ViT_d_Enc(nn.Module):
    def __init__(self,
                image_size = 64,
                patch_size = args.patch_size,
                frames = 64, 
                frame_patch_size = args.frame_patch_size, 
                num_classes = 64,# dim-z
                dim = 512,# ori 1024
                depth = 2,
                heads = 8,
                mlp_dim = 1024,# ori 2048
                pool = args.pool,
                dropout = 0,
                emb_dropout = 0):
        super(ViT_d_Enc, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.frames = frames
        self.frame_patch_size = frame_patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.pool = pool
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.model = ViT_d(
            image_size = self.image_size,
            image_patch_size = self.patch_size,
            frames = self.frames,
            frame_patch_size= self.frame_patch_size,
            num_classes = self.num_classes,
            dim = self.dim,
            depth = self.depth,
            heads = self.heads,
            mlp_dim = self.mlp_dim,
            pool = self.pool,
            dropout = self.dropout,
            emb_dropout = self.emb_dropout
        )

    def forward(self,x, m_id, h_id):
        x = self.model(x, m_id, h_id)

        return x

class ViT_Enc(nn.Module):
    def __init__(self,
                image_size = 64,
                patch_size = args.patch_size,
                frames = 64, 
                frame_patch_size = args.frame_patch_size, 
                num_classes = 64, # dim-z
                dim = 512,# ori 1024
                depth = 10,
                heads = 16,
                mlp_dim = 2048,# ori 2048
                pool = args.pool,
                dropout = 0,
                emb_dropout = 0):
        super(ViT_Enc, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.frames = frames
        self.frame_patch_size = frame_patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.pool = pool
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.model = ViT( image_size = self.image_size, 
                         patch_size = self.patch_size, 
                         frames = self.frames,
                        frame_patch_size= self.frame_patch_size,
                         num_classes = self.num_classes, 
                         dim = self.dim, 
                         depth = self.depth, 
                         heads = self.heads, 
                         mlp_dim = self.mlp_dim, 
                         pool = self.pool, 
                         dropout = self.dropout, 
                         emb_dropout = self.emb_dropout)

    def forward(self,x):
        x = self.model(x)

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
        self.ResNet34 = ResNetbasic(BasicBlock, [3, 4, 6, 3], self.z_dim, get_inplanes())
        self.mlp_latent = nn.Sequential(
                            nn.Linear(512*1*1*1, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Linear(512, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Linear(128, 64)
                            )
        #self.num_feature = self.ResNet34.fc.in_features
        #self.ResNet34.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

    def forward(self, x):
        x = self.ResNet34(x)
        # print("res latent111: ", x.size())

        x = x.view(-1, 512*1*1*1)
        # print("res latent222: ", x.size())

        x = self.mlp_latent(x)

        return x


class ResNet18Enc2(nn.Module):  #res_d
    def __init__(self, z_dim=32):
        super(ResNet18Enc2, self).__init__()
        self.z_dim = z_dim
        #self.ResNet34 = models.resnet34(pretrained=True)
        self.ResNet34_2 = ResNetbasic2(BasicBlock2, [3, 4, 6, 3], self.z_dim, get_inplanes())
        self.mlp_latent = nn.Sequential(
                            nn.Linear(512*1*1*1, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Linear(512, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Linear(128, 64)
                            )
        #self.num_feature = self.ResNet34.fc.in_features
        #self.ResNet34.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

    def forward(self, x, d_id):
        x = self.ResNet34_2(x, d_id)
        x = x.view(-1, 512*1*1*1)
        x = self.mlp_latent(x)
        # print("res2 latent: ", x.size())

        return x

class Model_STAS_3D_bs(Basenet): 


    def __init__(self,
                 in_channels: int,
                 latent_num: int,
                 num_classes: int,
                 if_mmf: int,
                 **kwargs) -> None:
        super(Model_STAS_3D_bs, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.if_mmf = if_mmf
        self.latent_num = latent_num

        if "trans" in args.if_transformer:
            self.Resnetencoder = ViT_Enc()
            # self.Resnetencoder2 = ViT_d_Enc()
        else:
            self.Resnetencoder = ResNet18Enc()
            # self.Resnetencoder2 = ResNet18Enc2()
        if self.if_mmf:
            self.mlp = nn.Sequential(self.Fc_bn_ReLU(16, 128),
                                    self.Fc_bn_ReLU(128, 256),
                                    self.Fc_bn_ReLU(256, self.latent_num),
                                    )
            self.decode_y = nn.Sequential(
                            self.Fc_bn_ReLU(int(self.latent_num * 2), 512), 
                            self.Fc_bn_ReLU(512, 256),
                            nn.Linear(256, self.num_classes),
                            nn.Softmax(dim=1),
                        )
        else:
            self.decode_y = nn.Sequential(
                            self.Fc_bn_ReLU(self.latent_num, 512), 
                            self.Fc_bn_ReLU(512, 256),
                            nn.Linear(256, self.num_classes),
                            nn.Softmax(dim=1),
                        )
            
    def forward(self, x, A):
        x = self.Resnetencoder(x)
        # print("x1", x.size())
        if self.if_mmf:
            A = self.mlp(A)
            x = torch.cat((x, A), 1)    
            out = self.decode_y(x)
        else:
            out = self.decode_y(x)
        return out
    

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer
                


class Model_STAS_3D(Basenet): 


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 gcndir: str,
                #  hidden_dims: List = None,
                 args = None,

                 **kwargs) -> None:
        super(Model_STAS_3D, self).__init__()
        self.args = args


        self.latent_dim = latent_dim

        # modules = []
        # if hidden_dims is None:
        #     hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels=h_dim,
        #                       kernel_size= 3, stride= 2, padding  = 1),
        #             nn.BatchNorm2d(h_dim),
        #             nn.LeakyReLU())
        #     )
        #     in_channels = h_dim

        # self.encoder = nn.Sequential(*modules)
        # print("self.args.if_transformer:", self.args.if_transformer)
        if "trans" in self.args.if_transformer:
            self.Resnetencoder = ViT_Enc()
            self.Resnetencoder2 = ViT_d_Enc()
        else:
            self.Resnetencoder = ResNet18Enc()
            self.Resnetencoder2 = ResNet18Enc2()


            

        self.fc_d = nn.Linear(latent_dim, latent_dim)
        self.fc_mu = nn.Sequential(self.Fc_bn_ReLU(64, 128),
                                   nn.Linear(128, 64),
                                   )
        self.fc_var = nn.Sequential(self.Fc_bn_ReLU(64, 128),
                                   nn.Linear(128, 64),
                                   )


        # Build Decoder
        modules = []

        self.gc1 = GraphConvolution(14, 128)
        self.gc2 = GraphConvolution(128, 64)
        self.relu = nn.LeakyReLU(0.2)
        _adj = gen_A(14, 0.4, str(gcndir))
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.enc_A1 = self.encode_A1()
        self.dec_y_by_vdis = self.decode_y_by_vdis()
        self.dec_x = self.decode_x()
        self.get_y = nn.Sequential(self.Fc_bn_ReLU(32, 128), 
                                    self.Fc_bn_ReLU(128, 64),
                                    nn.Linear(64, 2),
                                    nn.Softmax(dim=1),
                                    )

    def encode_x(self, input: Tensor, A1: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.Resnetencoder(input)
        # print("result", result.size())
        # print("EncodeA1", self.enc_A1(A1).size())
        result += self.enc_A1(A1)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var
    
    def encode_A1(self) -> Tensor: #TODO encode A1 to mu and sigma

        return  nn.Sequential(
            self.Fc_bn_ReLU(2, 32), #TODO dim modify
            self.Fc_bn_ReLU(32, 64),
            nn.Linear(64, self.latent_dim),
        )


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

    def forward(self, input: Tensor, m_id: int, h_id: int, A1: Tensor, gcn_target: Tensor, criterion_gcn, state: str, **kwargs) -> List[Tensor]:
        # m = 2
        # h = 2
        # for machine in range(m):

        mu, log_var = self.encode_x(input, A1)
        
        
        # pred_A2 = self.decode_A2_by_vdis_y(z[:, (self.latent_dim // 2):], cls)
        #d
        if 'train' in state:
            z = self.reparameterize(mu, log_var)


            cls = self.dec_y_by_vdis(z[:, (self.latent_dim // 2):])
            if "trans" in args.if_transformer:
                d = self.Resnetencoder2(input, str(m_id), str(h_id))
            else:
                d = self.Resnetencoder2(input, str(m_id)) # 0627 xiugai


            # d = torch.flatten(d, start_dim=1)
            # d = self.fc_d(d)

            # gcn
            inp = torch.eye(14).cuda()
            adj = gen_adj(self.A).detach()
            gcn = self.gc1(inp, adj)
            gcn = self.relu(gcn)
            gcn = self.gc2(gcn, adj)

            gcn = gcn.transpose(0, 1)
            # print("z", z.size())
            # print(torch.cat((z[:,32:], cls), dim=1).size())
            # print(gcn.size())
            gcn_x = torch.matmul(torch.cat((z[:,32:], cls), dim=1), gcn)
            # gcn_x = torch.matmul(z[:, :224], gcn)
            cls = self.get_y(cls)
            return  [self.dec_x(torch.cat((z, d), 1)), input, mu, log_var, gcn_x, cls]

        if 'val' in state:
            with torch.no_grad():
                z = self.reparameterize(mu, log_var)
                cls = self.dec_y_by_vdis(z[:, (self.latent_dim // 2):])

            cls_init = copy.deepcopy(cls)
            v_dis = z[:,32:]
            v_dis = v_dis
            cls.requires_grad = True
            v_dis.requires_grad = True
            d = z
            

            optimizer_cls = optim.Adam(params=[cls, v_dis], lr=self.args.lr2, weight_decay=self.args.wd2)

            for i in range(self.args.val_ep):
                optimizer_cls.zero_grad()
                inp = torch.eye(14).cuda()
                adj = gen_adj(self.A).detach()
                gcn = self.gc1(inp, adj)
                gcn = self.relu(gcn)
                gcn = self.gc2(gcn, adj)

                gcn = gcn.transpose(0, 1)
                gcn_x = torch.matmul(torch.cat((v_dis, cls), dim=1), gcn)
                loss_gcn = criterion_gcn(gcn_x, gcn_target)
                loss_gcn.backward(retain_graph=True)
                optimizer_cls.step()

            cls = self.get_y(cls)



        # v_all = torch.cat((z, d), 1)
            return  [self.dec_x(torch.cat((z, d), 1)), input, mu, log_var, gcn_x, self.get_y(cls_init), cls]

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
        # pred_A2 = args[4]
        loss_l1 = torch.nn.SmoothL1Loss()
        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        #recons_loss =F.mse_loss(recons, input)
        recons_loss = loss_l1(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}
    
    def loss_A2(self, pred_A2, A2):
        loss_A2_1 = torch.nn.CrossEntropyLoss()
        loss_A2_2 = torch.nn.mse_loss()
        loss_list = []
        loss = 0
        for ii in range(10):
            loss_list.append(loss_A2_1)
        loss_list.append(loss_A2_2)
        pred_A2_list = [pred_A2[:,:5], pred_A2[:,5:8], pred_A2[:,8],pred_A2[:,9],pred_A2[:,10],pred_A2[:,11],
                        pred_A2[:,12],pred_A2[:,13], pred_A2[:,14],pred_A2[:,15], pred_A2[:,-1]
                        ]
        for ii in range(len(loss_list)):
            loss += loss_list[ii](pred_A2_list[ii], A2[ii])
        return loss

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
    
    def decode_y_by_vdis(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.latent_dim // 2), 512), 
            self.Fc_bn_ReLU(512, 256),
            self.Fc_bn_ReLU(256, 128),
            nn.Linear(128, 32),
            # nn.Softmax(dim=1),
        )
    # def decode_A2_by_vdis_y(self, v_dis, y):
    #     decode_A2_by_vis =  nn.Sequential(
    #         self.Fc_bn_ReLU(int(self.zs_dim / 2), 512), 
    #         self.Fc_bn_ReLU(512, 256),
    #         nn.Linear(256, 17),
    #     )
    #     decode_A2_by_y = nn.Sequential(
    #         self.Fc_bn_ReLU(2, 512), 
    #         self.Fc_bn_ReLU(512, 256),
    #         nn.Linear(256, 17),
    #     )
    #     return decode_A2_by_vis(v_dis) + decode_A2_by_y(y)

    def decode_x(self):
        return nn.Sequential(
            UnFlatten(type="3d"),
            nn.Upsample(4),  # *4
            self.TConv_bn_ReLU(
                in_channels=64 * 2, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.Conv_bn_ReLU(in_channels=256, out_channels=512,
                              kernel_size=3, stride=1, padding=1),
            # self.Conv_bn_ReLU(in_channels=256, out_channels=256,
            #                   kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=512, out_channels=256,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            # self.Conv_bn_ReLU(in_channels=128, out_channels=128,
            #                   kernel_size=3, stride=1, padding=1),
            # self.Conv_bn_ReLU(in_channels=128, out_channels=128,
            #                   kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            # self.Conv_bn_ReLU(in_channels=64, out_channels=64,
            #                   kernel_size=3, stride=1, padding=1),
            # self.TConv_bn_ReLU(in_channels=64, out_channels=64, # *2^4
            #                    kernel_size=2, stride=2, padding=0),
            # self.Conv_bn_ReLU(in_channels=64, out_channels=64,
            #                   kernel_size=3, stride=1, padding=1),
            self.Conv_bn_ReLU(in_channels=64, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            self.Conv_bn_ReLU(in_channels=32, out_channels=16,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=16, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )


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


