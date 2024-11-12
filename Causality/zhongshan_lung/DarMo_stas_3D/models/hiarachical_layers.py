import torch
from torch import nn
import math

def bn_selector(channel_ic):

    return {'0': nn.BatchNorm2d(channel_ic).cuda(), '1': nn.BatchNorm2d(channel_ic).cuda(), '2': nn.BatchNorm2d(channel_ic).cuda()}

def bn_selector_3d(channel_ic):

    return {'0': nn.BatchNorm3d(channel_ic).cuda(), '1': nn.BatchNorm3d(channel_ic).cuda(), '2': nn.BatchNorm3d(channel_ic).cuda(), '3': nn.BatchNorm3d(channel_ic).cuda(), '4': nn.BatchNorm3d(channel_ic).cuda()}

def downsample_selector(self_inplanes, _planes, _block, _stride):

    return {'0':nn.Sequential(nn.Conv2d(self_inplanes, _planes * _block.expansion, kernel_size=1, stride=_stride, bias=False).cuda(),nn.BatchNorm2d(_planes * _block.expansion).cuda(),),
            '1':nn.Sequential(nn.Conv2d(self_inplanes, _planes * _block.expansion, kernel_size=1, stride=_stride, bias=False).cuda(),nn.BatchNorm2d(_planes * _block.expansion).cuda(),),
            '2':nn.Sequential(nn.Conv2d(self_inplanes, _planes * _block.expansion, kernel_size=1, stride=_stride, bias=False).cuda(),nn.BatchNorm2d(_planes * _block.expansion).cuda(),)}

def downsample_selector_3d(self_inplanes, _planes, _block, _stride):

    return {'0':nn.Sequential(nn.Conv3d(self_inplanes, _planes * _block.expansion, kernel_size=1, stride=_stride, bias=False).cuda(),nn.BatchNorm3d(_planes * _block.expansion).cuda(),),
            '1':nn.Sequential(nn.Conv3d(self_inplanes, _planes * _block.expansion, kernel_size=1, stride=_stride, bias=False).cuda(),nn.BatchNorm3d(_planes * _block.expansion).cuda(),),
            '2':nn.Sequential(nn.Conv3d(self_inplanes, _planes * _block.expansion, kernel_size=1, stride=_stride, bias=False).cuda(),nn.BatchNorm3d(_planes * _block.expansion).cuda(),),
            '3':nn.Sequential(nn.Conv3d(self_inplanes, _planes * _block.expansion, kernel_size=1, stride=_stride, bias=False).cuda(),nn.BatchNorm3d(_planes * _block.expansion).cuda(),),
            '4':nn.Sequential(nn.Conv3d(self_inplanes, _planes * _block.expansion, kernel_size=1, stride=_stride, bias=False).cuda(),nn.BatchNorm3d(_planes * _block.expansion).cuda(),)}

def ln_selector(channel_ic):

    return {'0': nn.LayerNorm(channel_ic).cuda(), 
            '1': nn.LayerNorm(channel_ic).cuda(), 
            '2': nn.LayerNorm(channel_ic).cuda(), 
            '3': nn.LayerNorm(channel_ic).cuda(),  
            '4': nn.LayerNorm(channel_ic).cuda() }
