from sklearn.metrics import roc_auc_score
import math

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# RN18_extractor + RN18_generator = RN18
class RN18_extrator(nn.Module):

    def __init__(self,
                 block=BasicBlock,
                 layers=None,
                 block_inplanes=None,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0
                 ):
        super().__init__()

        if layers is None:
            layers = [2, 2, 2, 2]
        if block_inplanes is None:
            block_inplanes = get_inplanes()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes *
                              block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        return x


class RN18_generator(nn.Module):

    def __init__(self,
                 block=BasicBlock,
                 layers=None,
                 block_inplanes=None,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=3):
        super().__init__()

        if layers is None:
            layers = [2, 2, 2, 2]
        if block_inplanes is None:
            block_inplanes = get_inplanes()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[1]
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, 32)
        self.fc_e = nn.Linear(124, n_classes)


        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 64)
        self.bn1 = nn.BatchNorm1d(num_features=32)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.fc_01 = nn.Linear(64, 32)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        ##########  sijie 1021daixiugai
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes *
                              block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, x1):
        # 构建一个简单的MLP
        x1 = self.fc1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        # x = torch.cat((x, x1), 1)
        x1_32 = self.fc_01(x1)  # -> 32
        #x_final = x1 + x_final

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x_32 = self.fc(x)


        # print((x_32.unsqueeze(2)).size())
        x_intra_1 = torch.matmul(((x_32.unsqueeze(2))), (x_32.unsqueeze(
            2).reshape(x_32.size(0), 1, x_32.size(1))))
        x1_intra_2 = torch.matmul(((x1_32.unsqueeze(2))), (x1_32.unsqueeze(
            2).reshape(x1_32.size(0), 1, x1_32.size(1))))
        x_inter = torch.matmul(((x_32.unsqueeze(2))), (x1_32.unsqueeze(
            2).reshape(x1_32.size(0), 1, x1_32.size(1))))
        # print(torch.matmul(((x_32.unsqueeze(2))), (x_32.unsqueeze(
        # 2).reshape(x_32.size(0), 1, x_32.size(1)))).size())
        #Inter-BFEM & Intra-BFEM
        intra_1 = self.fc_1024_to_20_1(x_intra_1.reshape(x.size(0), 1024))
        inter = self.fc_1024_to_20_2(x_inter.reshape(x.size(0), 1024))
        intra_2 = self.fc_1024_to_20_3(x1_intra_2.reshape(x.size(0), 1024))

        #x = torch.cat([x_32, intra_1, inter, intra_2, x1_32], 1)
        x = self.relu(torch.cat([x_32, intra_1, inter, intra_2, x1_32], 1))

        # 124 input
        
        x = self.fc_e(x)
        return x


class RN18_GPDBN(nn.Module):
    def __init__(self,
                 block=BasicBlock,
                 layers=None,
                 block_inplanes=None,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=3):
        super().__init__()
        self.extractor = RN18_extrator(block=block, layers=layers, block_inplanes=block_inplanes,
                                       shortcut_type=shortcut_type, widen_factor=widen_factor)
        self.generator = RN18_generator(block=block, layers=layers, block_inplanes=block_inplanes,
                                        shortcut_type=shortcut_type, widen_factor=widen_factor,
                                        n_classes=n_classes)

    def forward(self, x, t1):
        x = self.extractor(x)
        x = self.generator(x, t1)

        return x


if __name__ == '__main__':
    model1 = RN18_extrator()
    model2 = RN18_generator()
    model3 = RN18_GPDBN()
    batch = torch.rand([4, 1, 256, 256, 256])
    output = model3(batch)
    print(output.shape)
    # output = model2(output)
    # print(output.shape)

    # a = np.array([[0, 1, 0], [0, 0, 1]])
    # b = np.array([[0.05, 0.9, 0.05], [0, 0.5, 0.5]])
    # c = roc_auc_score(a, b, average='micro')
    # print(c)
