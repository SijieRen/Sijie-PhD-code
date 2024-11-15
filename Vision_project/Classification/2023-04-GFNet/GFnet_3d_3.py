import torch
from torch import dtype, float32
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from torch.nn.modules.module import Module
import logging
import math
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


'''
class Global_Filter(nn.Module):
    def __init__(self, dim, h, w, d ):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h,w,d,dim,2, dtype=torch.float32)*0.02)
        self.w = w
        self.h = h
        self.d = d
    def forward(self, x, spatial_size = None):
'''
#dim = 768


# class Global_Filter(nn.Module):
#     def __init__(self, h=18, w=21, d=10, dim=768):
#         super().__init__()
#         self.complex_weight = nn.Parameter(torch.randn(
#             h, w, d, dim, 2, dtype=torch.float32) * 0.02)
#         self.h = h
#         self.w = w
#         self.d = d

#     def forward(self, x):
#         B, N, C = x.shape
#         x = x.to(torch.float32)
#         x = x.view(B, 18, 21, 18, 768)
#         x = torch.fft.rfftn(x, dim=(1, 2, 3), norm='ortho')
#         weight = torch.view_as_complex(self.complex_weight)
#         x = x * weight
#         x = torch.fft.irfftn(x, s=(18, 21, 18), dim=(1, 2, 3), norm='ortho')
#         x = x.reshape(B, N, C)
#         return x

# Block [N x 3888 x 768] ----> [N x 3888 x 768]
# 2d sijei modify
class Global_Filter(nn.Module):
    def __init__(self, h=16, w=9, dim=768):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(
            h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.h = h
        self.w = w
        #self.d = d

    def forward(self, x):
        # print(x.shape)
        B, N, C = x.shape
        x = x.to(torch.float32)
        x = x.view(B, 16, 16, 768)
        x = torch.fft.rfftn(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        # print(x.size())
        # print(weight.size())
        x = x * weight
        x = torch.fft.irfftn(x, s=(16, 16), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x


class Block(nn.Module):
    def __init__(self, dim=768, mlp_ratio=8., drop=0.5, drop_path=0.6, act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=16, w=9, d=10):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.filter = Global_Filter(dim=768, h=h, w=w, d=d)
        self.filter = Global_Filter(dim=768, h=h, w=w)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + \
            self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class PatchEmbed(nn.Module):
    # image to patch embedding
    # def __init__(self, img_size=(181, 217, 181), patch_size=(10, 10, 10), num_classes=2, in_channels=1):
    # 2d sijie modify
    def __init__(self, img_size=(256, 256), patch_size=(16, 16), num_classes=2, in_channels=1):
        super().__init__()
        #img_size = to_2tuple(img_size)
        #patch_size = to_2tuple(patch_size)

        # num_patches = (img_size[2] // patch_size[2]) * (img_size[1] //
        #                                                 patch_size[1]) * (img_size[0] // patch_size[0])
        # 2d sijie modify
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # self.patch_dim = in_channels * \
        #     patch_size[0]*patch_size[1]*patch_size[2]

        # 2d sijie modify
        self.patch_dim = in_channels * patch_size[0]*patch_size[1]
        #self.proj = nn.Conv3d(in_channels, embedd_dim, kernel_size=patch_size, stride=patch_size)
        # self.to_patch_embedding = nn.Sequential(
        #    Rearrange('b c (h p1) (w p2) (d p3) -> b (h w 。。。d) (p1 p2 p3 c)',
        #             p1=img_size[0], p2=img_size[1], p3=img_size[2]),
        #    nn.Linear(patch_dim, dim),
        # )
        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        # self.proj = nn.Conv3d(in_channels, self.patch_dim,
        #                       kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(in_channels, self.patch_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # N, C, H, W, D = x.size()  # 这里用C表示的是通道的信息,D表示的是Z轴的信息，即slice数
        # assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2],\
        #     f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # return x

        N, C, H, W = x.size()  # 这里用C表示的是通道的信息,D表示的是Z轴的信息，即slice数
        assert H == self.img_size[0] and W == self.img_size[1],\
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class GFNet(nn.Module):
    # def __init__(self, img_size=(181, 217, 181), patch_size=(10, 10, 10), embed_dim=768, num_classes=2, in_channels=1, drop_rate=0.5, depth=8, mlp_ratio=2., representation_size=None, uniform_drop=False, drop_path_rate=0.6, norm_layer=False, dropcls=0.25):
    # 2d sijie modify
    def __init__(self, img_size=(256, 256), patch_size=(16, 16), embed_dim=768, num_classes=2, in_channels=3, drop_rate=0.5, depth=8, mlp_ratio=2., representation_size=None, uniform_drop=False, drop_path_rate=0.6, norm_layer=False, dropcls=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes)
        num_patches = self.patch_embed.num_patches  # 3888
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, 768))
        self.pos_drop = nn.Dropout(p=drop_rate)
        h = 16
        w = 9
        d = 10

        if uniform_drop:
            print('using uniform droppath with expected rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            print('using linear droppath with expected rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w, d=d)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(
            self.num_features, self.num_classes) if num_classes > 0 else nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    # x = torch.randn(2, 1, 181, 217, 181)
    # 2d sijie modify
    x = torch.randn(2, 3, 256, 256)
    net = GFNet()
    y = net(x)
    print(y.shape)
