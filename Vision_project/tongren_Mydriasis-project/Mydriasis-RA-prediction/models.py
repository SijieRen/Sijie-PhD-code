import torch
import math
from torch import nn
from torch.nn import init
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, final_relu=1):
        super(BasicBlock, self).__init__()
        self.final_relu = final_relu
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.final_relu:
            out = self.relu(out)

        return out


class ResNet_front(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, final_tanh=1):
        super(ResNet_front, self).__init__()
        self.leaky_relu = nn.Sequential(
            nn.LeakyReLU(),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       final_relu=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, final_relu=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for ss in range(1, blocks):
            if ss == blocks - 1 and final_relu == 0:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, final_relu=final_relu))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        return x


class ResNet_last(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout=0, dp=0.5):
        super(ResNet_last, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if dropout == 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        elif dropout == 1:
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, 256),
                nn.Dropout(dp),
                nn.Linear(256, num_classes),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet_last_double(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout=0, dp=0.5):
        super(ResNet_last_double, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if dropout == 0:
            self.fc_0 = nn.Linear(512 * block.expansion, num_classes)
            self.fc_00 = nn.Linear(512 * block.expansion, num_classes)
        elif dropout == 1:
            self.fc_0 = nn.Sequential(
                nn.Linear(512 * block.expansion, 256),
                nn.Dropout(dp),
                nn.Linear(256, num_classes),
            )
            self.fc_00 = nn.Sequential(
                nn.Linear(512 * block.expansion, 256),
                nn.Dropout(dp),
                nn.Linear(256, num_classes),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_finla_1 = self.fc_0(x)
        x_finla_2 = self.fc_00(x)
        return x_finla_1, x_finla_2


class ResNet_last_attr(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, feature_dim=6, dropout=0, dp=0.5, concat_num=1):
        super(ResNet_last_attr, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.concat_num = concat_num

        self.inplanes = 128 * concat_num
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if dropout == 0:
            self.fc_0 = nn.Linear(512 * block.expansion, num_classes)
        elif dropout == 1:
            self.fc_0 = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.Dropout(dp),
                nn.Linear(512, num_classes),
            )

        if dropout == 0:
            self.fc_01 = nn.Linear(256, num_classes)
        elif dropout == 1:
            self.fc_01 = nn.Sequential(
                nn.Linear(256, 256),
                nn.Dropout(dp),
                nn.Linear(256, num_classes),
            )

        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, x1):
        x = self.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_final = self.fc_0(x)

        if x1 is None:
            pass
        else:
            x1 = self.fc1(x1)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.fc2(x1)
            x1 = self.bn2(x1)
            x1 = self.relu(x1)
            # x = torch.cat((x, x1), 1)
            x1 = self.fc_01(x1)
            x_final = x1 + x_final

        return x_final

class ResNet_last_attr_double(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, feature_dim=6, dropout=0, dp=0.5, concat_num=1):
        super(ResNet_last_attr_double, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.concat_num = concat_num

        self.inplanes = 128 * concat_num
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if dropout == 0:
            self.fc_0 = nn.Linear(512 * block.expansion, num_classes)
            self.fc_00 = nn.Linear(512 * block.expansion, num_classes)
        elif dropout == 1:
            self.fc_0 = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.Dropout(dp),
                nn.Linear(512, num_classes),
            )
            self.fc_00 = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.Dropout(dp),
                nn.Linear(512, num_classes),
            )

        if dropout == 0:
            self.fc_01 = nn.Linear(256, num_classes)
            self.fc_11 = nn.Linear(256, num_classes)
        elif dropout == 1:
            self.fc_01 = nn.Sequential(
                nn.Linear(256, 256),
                nn.Dropout(dp),
                nn.Linear(256, num_classes),
            )
            self.fc_11 = nn.Sequential(
                nn.Linear(256, 256),
                nn.Dropout(dp),
                nn.Linear(256, num_classes),
            )

        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, x1):
        x = self.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_final_1 = self.fc_0(x)
        x_final_2 = self.fc_00(x)

        if x1 is None:
            pass
        else:
            x1 = self.fc1(x1)
            x1 = self.bn1(x1)
            x1 = self.relu(x1)
            x1 = self.fc2(x1)
            x1 = self.bn2(x1)
            x1 = self.relu(x1)
            # x = torch.cat((x, x1), 1)
            x_1 = self.fc_01(x1)
            x_2 = self.fc_11(x1)
            x_final_1 += x_1
            x_final_2 += x_2

        return x_final_1, x_final_2


class ResNet_last_multi(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_last_multi, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x), self.fc1(x)


def RN18_front(**kwargs):
    return ResNet_front(BasicBlock, [2, 2, 2, 2], **kwargs)


def RN18_last(**kwargs):
    return ResNet_last(BasicBlock, [2, 2, 2, 2], **kwargs)

def RN18_last_double(**kwargs):
    return ResNet_last_double(BasicBlock, [2, 2, 2, 2], **kwargs)


def RN18_last_attr(**kwargs):
    return ResNet_last_attr(BasicBlock, [2, 2, 2, 2], **kwargs)

def RN18_last_attr_double(**kwargs):
    return ResNet_last_attr_double(BasicBlock, [2, 2, 2, 2], **kwargs)


def RN18_last_multi(**kwargs):
    return ResNet_last_multi(BasicBlock, [2, 2, 2, 2], **kwargs)


class ResNet_last_attr_ind(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, concat_num=1, task_num=4, feature_dim=6):
        super(ResNet_last_attr_ind, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.concat_num = concat_num
        self.task_num = task_num

        self.inplanes = 128 * concat_num
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = []
        for _ in range(task_num):
            self.fc.append(nn.Linear(512 * block.expansion + 256, num_classes))
        self.fc = nn.ModuleList(self.fc)

        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def mapping_func(self, idx1, idx2):
        if self.task_num == 10:
            if idx1 == 1:
                return int(idx2) - 2
            elif idx1 == 2:
                return int(idx2) + 1
            elif idx1 == 3:
                return int(idx2) + 3
            elif idx1 == 4:
                return 9
        else:
            print('it should not be appear in mapping func')

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, x1, x2=None, idx1=0, idx2=-1, feature=0):
        if x2 is not None:
            x = self.relu(torch.cat([x, x2], dim=1))
        else:
            x = self.relu(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x1 = self.fc1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x = torch.cat((x, x1), 1)

        if self.task_num == 1:
            if feature:
                return self.fc[0](x), x
            else:
                return self.fc[0](x)
        else:
            if type(idx2) == int:
                pred = self.fc[idx1[0]](x[0].unsqueeze(0))
                for s in range(1, x.size(0)):
                    pred = torch.cat([pred, self.fc[idx1[s]](x[s].unsqueeze(0))], dim=0)
                if feature:
                    return pred, x
                else:
                    return pred
            else:
                true_idx = self.mapping_func(idx1[0], idx2[0])
                pred = self.fc[true_idx](x[0].unsqueeze(0))
                for s in range(1, x.size(0)):
                    true_idx = self.mapping_func(idx1[s], idx2[s])
                    pred = torch.cat([pred, self.fc[true_idx](x[s].unsqueeze(0))], dim=0)
                if feature:
                    return pred, x
                else:
                    return pred


def RN18_last_attr_ind(**kwargs):
    return ResNet_last_attr_ind(BasicBlock, [2, 2, 2, 2], **kwargs)


class Generator(nn.Module):
    def __init__(self, feature_num=6, is_ESPCN=0, mid_channel=1024):
        super(Generator, self).__init__()
        self.conv_blocks_feature = nn.Sequential(
            # 128*64*64 - 5256*32*32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # 256*32*32 - 512*32*32
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
        )
        self.conv_blocks1 = nn.Sequential(

            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=100 + feature_num, out_channels=2048, kernel_size=4, stride=1, padding=0),
            # up4
            nn.BatchNorm2d(num_features=2048),
            nn.ReLU(inplace=True),

            # input batch_size*1024*4*4 to btc*512*8*8
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=4, stride=2, padding=1),
            # up2:4-2-1 up4:4-4-0 8-4-2
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),

            # 8*8 - 16*16
            nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=4, stride=2, padding=1),
            # up2:4-2-1 up4:4-4-0 8-4-2
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),

            # 16*16 - 32*32
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            # up2:4-2-1 up4:4-4-0 8-4-2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            # 32*32 - 64*64
            nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1),
            # up2:4-2-1 up4:4-4-0 8-4-2
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            # 64*64 - 64*64
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        )
        self.is_ESPCN = is_ESPCN
        self.leaky_relu = nn.Sequential(
            nn.LeakyReLU(),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        if self.is_ESPCN == 1:
            self.ESPCN = ESPCN(mid_channel=mid_channel)

    def forward(self, z, feature):
        mid_feature_1 = self.conv_blocks1(z)
        mid_feature_2 = self.conv_blocks_feature(feature)
        x = torch.cat((mid_feature_1, mid_feature_2), 1)
        featuremap = self.conv_blocks2(x)

        if self.is_ESPCN == 1:
            featuremap = self.ESPCN(featuremap)

        return featuremap




class MLP(nn.Module):
    def __init__(self, in_channels=128, out_channel=1):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=False),
            nn.Linear(256, out_channel),
        )

    def forward(self, x):
        x = self.mlp(x)
        
        return x

