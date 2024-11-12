from sklearn.metrics import roc_auc_score
import math

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from FFC_complexLayers import *
from FFC_complexFunctions import *
from FFC_shift import *
from FFC_complexUtils import Sequential_complex


def get_inplanes():
    return [32, 64, 128, 256]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = Sequential_complex(
            ComplexConv2d(in_channels, out_channels, kernel_size=3,
                          stride=stride, padding=1, bias=False),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(inplace=True),  # inplace = true
            ComplexConv2d(out_channels, out_channels * BasicBlock.expansion,
                          kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = Sequential_complex()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = Sequential_complex(
                ComplexConv2d(in_channels, out_channels * BasicBlock.expansion,
                              kernel_size=1, stride=stride, bias=False),
                ComplexBatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x1, x2):
        x3, x4 = self.residual_function(x1, x2)
        x1, x2 = complex_relu(x3, x4, inplace=True)
        return x1, x2

class BasicBlock_3d(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = Sequential_complex(
            ComplexConv3d(in_channels, out_channels, kernel_size=3,
                          stride=stride, padding=1, bias=False),
            ComplexBatchNorm3d(out_channels),
            ComplexReLU(inplace=True),  # inplace = true
            ComplexConv3d(out_channels, out_channels * BasicBlock_3d.expansion,
                          kernel_size=3, padding=1, bias=False),
            ComplexBatchNorm3d(out_channels * BasicBlock_3d.expansion)
        )

        # shortcut
        self.shortcut = Sequential_complex()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock_3d.expansion * out_channels:
            self.shortcut = Sequential_complex(
                ComplexConv3d(in_channels, out_channels * BasicBlock_3d.expansion,
                              kernel_size=1, stride=stride, bias=False),
                ComplexBatchNorm3d(out_channels * BasicBlock_3d.expansion)
            )

    def forward(self, x1, x2):
        x3, x4 = self.residual_function(x1, x2)
        x1, x2 = complex_relu(x3, x4, inplace=True)
        return x1, x2



class RN18_FFC_2d(nn.Module):

    def __init__(self, block, num_block, num_classes=2, inputchannel=1):
        super().__init__()

        self.in_channels = 64
        self.conv1 = ComplexConv2d(
            in_channels=inputchannel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = ComplexBatchNorm2d(num_features=64)
        self.relu = ComplexReLU(inplace=True)
        self.maxpool = ComplexMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = Complex_AdaptiveAvgPool2d((1, 1))

        self.fc_r_1 = nn.Sequential(nn.Linear(512 * block.expansion, 4096),
                                    nn.ReLU(inplace=True))
        self.fc_r_2 = nn.Sequential(nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=True))
        self.fc_r_3 = nn.Sequential(nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=True))
        self.fc_r_4 = nn.Sequential(nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=True))
        # self.fc_r_5 = nn.Sequential(nn.Linear(4096, 4096),
        #                             nn.ReLU(inplace=True))
        self.fc_r_6 = nn.Sequential(nn.Linear(4096, 512 * block.expansion),
                                    nn.ReLU(inplace=True))

        self.fc_i_1 = nn.Sequential(nn.Linear(512 * block.expansion, 4096),
                                    nn.ReLU(inplace=True))
        self.fc_i_2 = nn.Sequential(nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=True))
        self.fc_i_3 = nn.Sequential(nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=True))
        self.fc_i_4 = nn.Sequential(nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=True))
        # self.fc_i_5 = nn.Sequential(nn.Linear(4096, 4096),
        #                             nn.ReLU(inplace=True))
        self.fc_i_6 = nn.Sequential(nn.Linear(4096, 512 * block.expansion),
                                    nn.ReLU(inplace=True))

        self.fc = ComplexLinear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return Sequential_complex(*layers)

    def forward(self, x, target=None):  # sijie FFCNet modification
        x = torch.cat((x, x), dim=1) #TODO 可能需要修改维度dim
        # print(x.size())
        xr = x[:, [0, 1, 2], :, :]
        xi = x[:, [3, 4, 5], :, :]
        # print(x.size())
        # print("xr: ", xr.size())
        # print("xi: ", xi.size())
        xr, xi = self.conv1(xr, xi)
        xr, xi = self.bn1(xr, xi)
        xr, xi = self.relu(xr, xi)
        xr, xi = self.maxpool(xr, xi)
        xr, xi = self.conv2_x(xr, xi)
        xr, xi = self.conv3_x(xr, xi)
        xr, xi = self.conv4_x(xr, xi)
        xr, xi = self.conv5_x(xr, xi)
        xr, xi = self.avg_pool(xr, xi)
        xr = xr.view(xr.size(0), -1)
        xi = xi.view(xi.size(0), -1)
        xr = self.fc_r_1(xr)
        xr = self.fc_r_2(xr)
        xr = self.fc_r_3(xr)
        xr = self.fc_r_4(xr)
        # xr = self.fc_r_5(xr)
        xr = self.fc_r_6(xr)
        xi = self.fc_i_1(xi)
        xi = self.fc_i_2(xi)
        xi = self.fc_i_3(xi)
        xi = self.fc_i_4(xi)
        # xi = self.fc_i_5(xi)
        xi = self.fc_i_6(xi)
        # print(xr.size())
        # print(xi.size())
        xr, xi = self.fc(xr, xi)
        # print(xr.size())
        # print(xi.size())
        x = torch.sqrt(torch.pow(xr, 2)+torch.pow(xi, 2))
        # print(x.size())
        return x


class RN18_FFC_3d(nn.Module):

    def __init__(self, 
                 block=BasicBlock_3d, 
                 num_block=[2, 2, 2, 2], 
                 num_classes=2, 
                 inputchannel=1):
        super().__init__()

        self.in_channels = 64
        self.conv1 = ComplexConv3d(
            in_channels=inputchannel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = ComplexBatchNorm3d(num_features=64)
        self.relu = ComplexReLU(inplace=True)
        self.maxpool = ComplexMaxPool3d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = Complex_AdaptiveAvgPool3d((1, 1))

        self.fc_r_1 = nn.Sequential(nn.Linear(512 * block.expansion, 4096),
                                    nn.ReLU(inplace=True))
        self.fc_r_2 = nn.Sequential(nn.Linear(4096, 512 * block.expansion),
                                    nn.ReLU(inplace=True))

        self.fc_i_1 = nn.Sequential(nn.Linear(512 * block.expansion, 4096),
                                    nn.ReLU(inplace=True))
        self.fc_i_2 = nn.Sequential(nn.Linear(4096, 512 * block.expansion),
                                    nn.ReLU(inplace=True))

        self.fc = ComplexLinear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return Sequential_complex(*layers)

    def forward(self, x, target=None):  # sijie FFCNet modification
        x = torch.cat((x, x), dim=1) #TODO 可能需要修改维度dim
        # print(x.size())
        xr = x[:, [0, 1, 2], :, :]
        xi = x[:, [3, 4, 5], :, :]
        # print(x.size())
        # print("xr: ", xr.size())
        # print("xi: ", xi.size())
        xr, xi = self.conv1(xr, xi)
        xr, xi = self.bn1(xr, xi)
        xr, xi = self.relu(xr, xi)
        xr, xi = self.maxpool(xr, xi)
        xr, xi = self.conv2_x(xr, xi)
        xr, xi = self.conv3_x(xr, xi)
        xr, xi = self.conv4_x(xr, xi)
        xr, xi = self.conv5_x(xr, xi)
        xr, xi = self.avg_pool(xr, xi)
        xr = xr.view(xr.size(0), -1)
        xi = xi.view(xi.size(0), -1)

        xr = self.fc_r_1(xr)
        xr = self.fc_r_2(xr)

        xi = self.fc_i_1(xi)
        xi = self.fc_i_2(xi)
        # print(xr.size())
        # print(xi.size())
        xr, xi = self.fc(xr, xi)
        # print(xr.size())
        # print(xi.size())
        x = torch.sqrt(torch.pow(xr, 2)+torch.pow(xi, 2))
        # print(x.size())
        return x




if __name__ == '__main__':
    # model1 = RN18_extrator()
    # model2 = RN18_generator()
    model3 = RN18_FFC_3d()
    batch = torch.rand([4, 1, 256, 256, 256])
    output = model3(batch)
    print(output.shape)
    # output = model2(output)
    # print(output.shape)

    # a = np.array([[0, 1, 0], [0, 0, 1]])
    # b = np.array([[0.05, 0.9, 0.05], [0, 0.5, 0.5]])
    # c = roc_auc_score(a, b, average='micro')
    # print(c)
