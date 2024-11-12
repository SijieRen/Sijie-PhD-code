import torch

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
                 base_width=64, dilation=1, norm_layer=None, final_relu = 1):
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
        self.final_tanh = final_tanh
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


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, final_relu = 1):
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
        if self.final_tanh == 1:
            x = torch.tanh(x)
        elif self.final_tanh == 2 :
            x = torch.sigmoid(x)
        elif self.final_tanh == 3 :
            x = self.leaky_relu(x)
        elif self.final_tanh == 4 :
            x = self.relu(x)
        return x

class ResNet_last(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
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
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

class ResNet_last_attr(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, feature_dim=6):
        super(ResNet_last_attr, self).__init__()
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
        self.fc = nn.Linear(512 * block.expansion + 256, num_classes)

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

        x1 = self.fc1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu(x1)
        x = torch.cat((x, x1), 1)
        x = self.fc(x)
        
        return x

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
    return ResNet_front(BasicBlock, [2, 2, 2, 2],**kwargs)

def RN18_last(**kwargs):
    return ResNet_last(BasicBlock, [2, 2, 2, 2], **kwargs)

def RN18_last_attr(**kwargs):
    return ResNet_last_attr(BasicBlock, [2, 2, 2, 2], **kwargs)

def RN18_last_multi(**kwargs):
    return ResNet_last_multi(BasicBlock, [2, 2, 2, 2], **kwargs)


class Generator_paper1order1(nn.Module):
    def __init__(self):
        super(Generator_paper1order1, self).__init__()
        # Filter 1024 512 256
        # self.init_size = 200
        # self.l1 = nn.Sequential(nn.Linear(1,  (self.init_size // 8) ** 2))#todo size needed tobe modified(adjust size)
        # self.in_channels = in_channels
        self.conv_blocks = nn.Sequential(
            
            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=108, out_channels=512, kernel_size=4, stride=1, padding=0),  # up4
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            # input batch_size*1024*4*4
            nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            # input batch_size*512*16*16//////512*8*8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            # input batch_size*256*64*64//////256*16*16
            nn.ConvTranspose2d(in_channels=256, out_channels=6, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            # nn.LeakyReLU(inplace=True),
            
            # output batch_size*3*256*256//////3*32*32
            nn.Tanh(),
        )
        self.conv_y = nn.Sequential(
            
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 2),
            nn.ReLU(inplace=True),
        
        )
    
    def forward(self, z, y):
        # out = self.l1(z)
        #out = z.view(z.shape[0], 108, 1, 1)
        out = z
        img = self.conv_blocks(out)
        img1, img2 = img.chunk(2, 1)
        # y = y.view(y.shape[0], 100, 1, 1)
        y_hat = self.conv_y(y)
        y_hat1, y_hat2 = y_hat.chunk(2, 1)
        return img1, img2, y_hat1, y_hat2


class Generator_paper1order1_half(nn.Module):
    def __init__(self):
        super(Generator_paper1order1_half, self).__init__()
        # Filter 1024 512 256
        # self.init_size = 200
        # self.l1 = nn.Sequential(nn.Linear(1,  (self.init_size // 8) ** 2))#todo size needed tobe modified(adjust size)
        # self.in_channels = in_channels
        self.conv_blocks = nn.Sequential(
            
            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=108, out_channels=256, kernel_size=4, stride=1, padding=0),  # up4
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            # input batch_size*1024*4*4
            nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            # input batch_size*512*16*16//////512*8*8
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            
            # input batch_size*256*64*64//////256*16*16
            nn.ConvTranspose2d(in_channels=128, out_channels=6, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            # nn.LeakyReLU(inplace=True),
            
            # output batch_size*3*256*256//////3*32*32
            nn.Tanh(),
        )
        self.conv_y = nn.Sequential(
            
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 2),
            nn.ReLU(inplace=True),
        
        )
    
    def forward(self, z, y):
        # out = self.l1(z)
        out = z.view(z.shape[0], 108, 1, 1)
        img = self.conv_blocks(out)
        img1, img2 = img.chunk(2, 1)
        # y = y.view(y.shape[0], 100, 1, 1)
        y_hat = self.conv_y(y)
        y_hat1, y_hat2 = y_hat.chunk(2, 1)
        return img1, img2, y_hat1, y_hat2


class Generator_paper1order2(nn.Module):
    def __init__(self):
        super(Generator_paper1order2, self).__init__()
        # Filter 1024 512 256
        # self.init_size = 200
        # self.l1 = nn.Sequential(nn.Linear(1,  (self.init_size // 8) ** 2))#todo size needed tobe modified(adjust size)
        # self.in_channels = in_channels
        self.conv_blocks = nn.Sequential(
            
            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0),  # up4
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            # input batch_size*1024*4*4
            nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            # input batch_size*512*16*16//////512*8*8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            # input batch_size*256*64*64//////256*16*16
            nn.ConvTranspose2d(in_channels=256, out_channels=9, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            # nn.LeakyReLU(inplace=True),
            
            # output batch_size*3*256*256//////3*32*32
            nn.Tanh(),
        )
        self.conv_y = nn.Sequential(
            
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 3),
            nn.ReLU(inplace=True),
        
        )
    
    def forward(self, z, y):
        # out = self.l1(z)
        out = z.view(z.shape[0], 100, 1, 1)
        img = self.conv_blocks(out)
        img1, img2, img3 = img.chunk(3, 1)
        # y = y.view(y.shape[0], 100, 1, 1)
        y_hat = self.conv_y(y)
        y_hat1, y_hat2, y_hat3 = y_hat.chunk(3, 1)
        return img1, img2, img3, y_hat1, y_hat2, y_hat3


class Generator_paper1order2_half(nn.Module):
    def __init__(self):
        super(Generator_paper1order2_half, self).__init__()
        # Filter 1024 512 256
        # self.init_size = 200
        # self.l1 = nn.Sequential(nn.Linear(1,  (self.init_size // 8) ** 2))#todo size needed tobe modified(adjust size)
        # self.in_channels = in_channels
        self.conv_blocks = nn.Sequential(
            
            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=100, out_channels=256, kernel_size=4, stride=1, padding=0),  # up4
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            # input batch_size*1024*4*4
            nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            # input batch_size*512*16*16//////512*8*8
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            
            # input batch_size*256*64*64//////256*16*16
            nn.ConvTranspose2d(in_channels=128, out_channels=9, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            # nn.LeakyReLU(inplace=True),
            
            # output batch_size*3*256*256//////3*32*32
            nn.Tanh(),
        )
        self.conv_y = nn.Sequential(
            
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 3),
            nn.ReLU(inplace=True),
        
        )
    
    def forward(self, z, y):
        # out = self.l1(z)
        out = z.view(z.shape[0], 100, 1, 1)
        img = self.conv_blocks(out)
        img1, img2, img3 = img.chunk(3, 1)
        # y = y.view(y.shape[0], 100, 1, 1)
        y_hat = self.conv_y(y)
        y_hat1, y_hat2, y_hat3 = y_hat.chunk(3, 1)
        return img1, img2, img3, y_hat1, y_hat2, y_hat3

class Generator_paper1order3(nn.Module):
    def __init__(self):
        super(Generator_paper1order3, self).__init__()
        # Filter 1024 512 256
        # self.init_size = 200
        # self.l1 = nn.Sequential(nn.Linear(1,  (self.init_size // 8) ** 2))#todo size needed tobe modified(adjust size)
        # self.in_channels = in_channels
        self.conv_blocks = nn.Sequential(
            
            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0),  # up4
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            # input batch_size*1024*4*4
            nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            # input batch_size*512*16*16//////512*8*8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            # input batch_size*256*64*64//////256*16*16
            nn.ConvTranspose2d(in_channels=256, out_channels=12, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            # nn.LeakyReLU(inplace=True),
            
            # output batch_size*3*256*256//////3*32*32
            nn.Tanh(),
        )
        self.conv_y = nn.Sequential(
            
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 4),
            nn.ReLU(inplace=True),
        
        )
    
    def forward(self, z, y):
        # out = self.l1(z)
        out = z.view(z.shape[0], 100, 1, 1)
        img = self.conv_blocks(out)
        img1, img2, img3, img4 = img.chunk(4, 1)
        # y = y.view(y.shape[0], 100, 1, 1)
        y_hat = self.conv_y(y)
        y_hat1, y_hat2, y_hat3, y_hat4 = y_hat.chunk(4, 1)
        return img1, img2, img3, img4, y_hat1, y_hat2, y_hat3, y_hat4


class Generator_paper1order3_half(nn.Module):
    def __init__(self):
        super(Generator_paper1order3_half, self).__init__()
        # Filter 1024 512 256
        # self.init_size = 200
        # self.l1 = nn.Sequential(nn.Linear(1,  (self.init_size // 8) ** 2))#todo size needed tobe modified(adjust size)
        # self.in_channels = in_channels
        self.conv_blocks = nn.Sequential(
            
            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=100, out_channels=256, kernel_size=4, stride=1, padding=0),  # up4
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            # input batch_size*1024*4*4
            nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),  # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            # input batch_size*512*16*16//////512*8*8
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            
            # input batch_size*256*64*64//////256*16*16
            nn.ConvTranspose2d(in_channels=128, out_channels=12, kernel_size=8, stride=4, padding=2),  # 4
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            # nn.LeakyReLU(inplace=True),
            
            # output batch_size*3*256*256//////3*32*32
            nn.Tanh(),
        )
        self.conv_y = nn.Sequential(
            
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 4),
            nn.ReLU(inplace=True),
        
        )
    
    def forward(self, z, y):
        # out = self.l1(z)
        out = z.view(z.shape[0], 100, 1, 1)
        img = self.conv_blocks(out)
        img1, img2, img3, img4 = img.chunk(4, 1)
        # y = y.view(y.shape[0], 100, 1, 1)
        y_hat = self.conv_y(y)
        y_hat1, y_hat2, y_hat3, y_hat4 = y_hat.chunk(4, 1)
        return img1, img2, img3, img4, y_hat1, y_hat2, y_hat3, y_hat4

class Generator_paper2(nn.Module):
    def __init__(self):
        super(Generator_paper2, self).__init__()
        # Filter 1024 512 256
        # Filter 1024 512 256
        # self.init_size = 200
        # self.l1 = nn.Sequential(nn.Linear(1,  (self.init_size // 8) ** 2))#todo size needed tobe modified(adjust size)
        # self.in_channels = in_channels
        self.conv_blocks = nn.Sequential(

            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),  # up4
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),

            # input batch_size*1024*4*4
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=8, stride=4, padding=2),
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # input batch_size*512*16*16//////512*8*8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=8, stride=4, padding=2),
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            # input batch_size*256*64*64//////256*16*16
            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=8, stride=4, padding=2),
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            # nn.LeakyReLU(inplace=True),

            # output batch_size*3*256*256//////3*32*32
            nn.Tanh(),
        )
        self.conv_y = nn.Sequential(
    
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
    
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
    
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
    
            nn.Linear(1024, 1),
            nn.ReLU(inplace=True),
    
        )

    def forward(self, z, y):
        # out = self.l1(z)
        out = z.view(z.shape[0], 100, 1, 1)
        img = self.conv_blocks(out)
        #y = y.view(y.shape[0], 100, 1, 1)
        y_hat = self.conv_y(y)
        return img, y_hat

class Generator_paper(nn.Module):
    def __init__(self):
        super(Generator_paper, self).__init__()
        # Filter 1024 512 256
        # self.init_size = 200
        # self.l1 = nn.Sequential(nn.Linear(1,  (self.init_size // 8) ** 2))#todo size needed tobe modified(adjust size)
        # self.in_channels = in_channels
        self.conv_blocks = nn.Sequential(

            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),  # up4
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),

            # input batch_size*1024*4*4
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=8, stride=4, padding=2),
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            # input batch_size*512*16*16//////512*8*8
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=8, stride=4, padding=2),
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
    
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),#up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            
            # input batch_size*256*64*64//////256*16*16
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),#up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            # nn.LeakyReLU(inplace=True),

            # output batch_size*3*256*256//////3*32*32
            nn.Tanh(),
        )
        self.conv_y = nn.Sequential(
    
            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=0),  # up4
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
    
            # input batch_size*1024*4*4
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), # up2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
    
            # input batch_size*512*16*16//////512*8*8
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), #down2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),  # down2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1),  # down2
            # up2:4-2-1 up4:4-4-0 up4:8 4 2
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True),
            
            

        )
        
        
    def forward(self, z, y):
        # out = self.l1(z)
        out = z.view(z.shape[0], 100, 1, 1)
        img = self.conv_blocks(out)
        y = y.view(y.shape[0], 1, 1, 1)
        y_hat = self.conv_y(y)
        return img, y_hat

class Generator(nn.Module):
    def __init__(self, feature_num=6, final_tanh = 0, is_ESPCN=0, scale_factor=2, mid_channel=1024, dw_type='conv'):
        super(Generator, self).__init__()
        self.final_tanh = final_tanh
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
            self.ESPCN = ESPCN(scale_factor=scale_factor, mid_channel=mid_channel, dw_type=dw_type, final_tanh=final_tanh)
    
    def forward(self, z, feature):
        mid_feature_1 = self.conv_blocks1(z)
        mid_feature_2 = self.conv_blocks_feature(feature)
        x = torch.cat((mid_feature_1, mid_feature_2), 1)
        featuremap = self.conv_blocks2(x)

        if self.is_ESPCN == 1:
            featuremap = self.ESPCN(featuremap)

        if self.final_tanh == 1:
            featuremap = torch.tanh(featuremap)
        elif self.final_tanh == 2 :
            featuremap = torch.sigmoid(featuremap)
        elif self.final_tanh == 3 :
            featuremap = self.leaky_relu(featuremap)
        elif self.final_tanh == 4 :
            featuremap = self.relu(featuremap)

        return featuremap


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.single_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, before_channel, bilinear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels + before_channel, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)  # in_channels
        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 bilinear=True,
                 feature_num=6,
                 final_tanh = 0,
                 is_ESPCN=0, scale_factor=2, mid_channel=1024, dw_type='conv'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.final_tanh = final_tanh
        self.leaky_relu = nn.Sequential(
            nn.LeakyReLU(),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        self.is_ESPCN = is_ESPCN
        
        self.inc = DoubleConv(n_channels, 64) #x1
        self.down1 = Down(64, 128) #x2
        self.down2 = Down(128, 256) #x3
        self.down3 = Down(256, 512) #x4
        self.down4 = Down(512, 1024) #x5
        self.up1 = Up(1024, 512, 512, bilinear)
        self.up2 = Up(512, 256, 256, bilinear)
        self.up3 = Up(256, 128, 128, bilinear)
        self.up4 = Up(128, 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.conv_blocks1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100 + feature_num, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
        )
        self.conv_tanh = nn.Sequential(
            nn.Tanh()
        )

        if self.is_ESPCN == 1:
            self.ESPCN = ESPCN(scale_factor=scale_factor, mid_channel=mid_channel, dw_type=dw_type, final_tanh=final_tanh)
    
    def forward(self,  x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)  # x3 256 # 256 x 64 x 64
        x4 = self.down3(x3)  # x4 512 # 512 x 32 x 32
        x5 = self.down4(x4)  # x5 512 # 512 x 16 x 16
        #z = self.conv_blocks1(z)  # 512 x 16 x 16
        #x6 = torch.cat((x5, z), 1)  # 1024 x 16 x 16
        x = self.up1(x5, x4)  # x 256 # (1536)
        x = self.up2(x, x3)  # x 128
        x = self.up3(x, x2)  # x
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.conv_tanh(x)
        
        return x


class UNet_half(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 bilinear=True,
                 feature_num=6,
                 final_tanh=0,
                 is_ESPCN=0, scale_factor=2, mid_channel=1024, dw_type='conv',
                 inch=32,outch=256,):
        super(UNet_half, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.final_tanh = final_tanh
        self.leaky_relu = nn.Sequential(
            nn.LeakyReLU(),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        self.is_ESPCN = is_ESPCN
        
        self.inc = SingleConv(n_channels, inch)  # x1
        self.down1 = Down(inch, 64)  # x2
        self.down2 = Down(64, 128)  # x3
        self.down3 = Down(128, 256)  # x4
        self.down4 = Down(256, outch)  # x5
        self.up1 = Up(outch, 256, 256, bilinear)
        self.up2 = Up(256, 128, 128, bilinear)
        self.up3 = Up(128, 64, 64, bilinear)
        self.up4 = Up(64, inch, inch, bilinear)
        self.outc = OutConv(inch, n_classes)
        self.conv_blocks1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100 + feature_num, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
        )
        self.conv_tanh = nn.Sequential(
            nn.Tanh()
        )
        
        
        if self.is_ESPCN == 1:
            self.ESPCN = ESPCN(scale_factor=scale_factor, mid_channel=mid_channel, dw_type=dw_type,
                               final_tanh=final_tanh)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)  # x3 256 # 256 x 64 x 64
        x4 = self.down3(x3)  # x4 512 # 512 x 32 x 32
        x5 = self.down4(x4)  # x5 512 # 512 x 16 x 16
        # z = self.conv_blocks1(z)  # 512 x 16 x 16
        # x6 = torch.cat((x5, z), 1)  # 1024 x 16 x 16
        x = self.up1(x5, x4)  # x 256 # (1536)
        x = self.up2(x, x3)  # x 128
        x = self.up3(x, x2)  # x
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.conv_tanh(x)
        
        return x


class UNet_half_minus1(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 bilinear=True,
                 feature_num=6,
                 final_tanh=0,
                 is_ESPCN=0, scale_factor=2, mid_channel=1024, dw_type='conv',
                 inch=32,outch=256):
        super(UNet_half_minus1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.final_tanh = final_tanh
        self.leaky_relu = nn.Sequential(
            nn.LeakyReLU(),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        self.is_ESPCN = is_ESPCN
        
        self.inc = DoubleConv(n_channels, inch)  # x1
        self.down1 = Down(inch, 64)  # x2
        self.down2 = Down(64, 128)  # x3
        self.down3 = Down(128, outch)  # x4
        #self.down4 = Down(256, 512)  # x5
        #self.up1 = Up(512, 256, 256, bilinear)
        self.up2 = Up(outch, 128, 128, bilinear)
        self.up3 = Up(128, 64, 64, bilinear)
        self.up4 = Up(64, inch, inch, bilinear)
        self.outc = OutConv(inch, n_classes)
        self.conv_blocks1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100 + feature_num, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
        )
        self.conv_tanh = nn.Sequential(
            nn.Tanh()
        )
        
        if self.is_ESPCN == 1:
            self.ESPCN = ESPCN(scale_factor=scale_factor, mid_channel=mid_channel, dw_type=dw_type,
                               final_tanh=final_tanh)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)  # x3 256 # 256 x 64 x 64
        x4 = self.down3(x3)  # x4 512 # 512 x 32 x 32
        #x5 = self.down4(x4)  # x5 512 # 512 x 16 x 16
        # z = self.conv_blocks1(z)  # 512 x 16 x 16
        # x6 = torch.cat((x5, z), 1)  # 1024 x 16 x 16
        #x = self.up1(x5, x4)  # x 256 # (1536)
        x = self.up2(x4, x3)  # x 128
        x = self.up3(x, x2)  # x
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.conv_tanh(x)
        
        return x


class UNet_half_minus2(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 bilinear=True,
                 feature_num=6,
                 final_tanh=0,
                 is_ESPCN=0, scale_factor=2, mid_channel=1024, dw_type='conv',
                 ch=32   ):
        super(UNet_half_minus2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.final_tanh = final_tanh
        self.leaky_relu = nn.Sequential(
            nn.LeakyReLU(),
        )
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        self.is_ESPCN = is_ESPCN
        
        self.inc = DoubleConv(n_channels, ch)  # x1
        #self.down1 = Down(32, 64)  # x2
        self.down2 = Down(ch, 2*ch)  # x3
        self.down3 = Down(2*ch, 4*ch)  # x4
        self.down4 = Down(4*ch, 8*ch)  # x5
        self.up1 = Up(8*ch, 4*ch, 4*ch, bilinear)
        self.up2 = Up(4*ch, 2*ch, 2*ch, bilinear)
        self.up3 = Up(2*ch, ch, ch, bilinear)
        #self.up4 = Up(64, 32, 32, bilinear)
        self.outc = OutConv(ch, n_classes)
        self.conv_blocks1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100 + feature_num, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
        )
        self.conv_tanh = nn.Sequential(
            nn.Tanh()
        )
        
        if self.is_ESPCN == 1:
            self.ESPCN = ESPCN(scale_factor=scale_factor, mid_channel=mid_channel, dw_type=dw_type,
                               final_tanh=final_tanh)
    
    def forward(self, x):
        x1 = self.inc(x)
        #x2 = self.down1(x1)
        x3 = self.down2(x1)  # x3 256 # 256 x 64 x 64
        x4 = self.down3(x3)  # x4 512 # 512 x 32 x 32
        x5 = self.down4(x4)  # x5 512 # 512 x 16 x 16
        # z = self.conv_blocks1(z)  # 512 x 16 x 16
        # x6 = torch.cat((x5, z), 1)  # 1024 x 16 x 16
        x = self.up1(x5, x4)  # x 256 # (1536)
        x = self.up2(x, x3)  # x 128
        x = self.up3(x, x1)  # x
        #x = self.up4(x, x1)
        x = self.outc(x)
        x = self.conv_tanh(x)
        
        return x


class Discriminator_paper(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = 3 (3x256x256)(3*32*32)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (batch_size*C*256*256)(c*32*32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=8, stride=4, padding=2),#down4:4-4-0 #down2:4-2-1 up4:8 4 2
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # Image (batch_size*256*64*64)(256*16*16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=8, stride=4, padding=2),#down4:4-4-0 #down2:4-2-1 up4:8 4 2
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (batch_size*512*16*16)(512*8*8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=8, stride=4, padding=2),#down4:4-4-0 #down2:4-2-1 up4:8 4 2
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True)
            # output of main module --> State (1024*4*4)(1024*4*4)
             )

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            # Image (batch_size*128*64*64)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),  # modify
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # Image (batch_size*512*32*32)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(num_features=1024),
            #nn.LeakyReLU(0.2, inplace=True),
        )

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0)
        )

    # 1024*1*1

    def forward(self, x):
        x = self.main_module(x)
        x = self.output(x)
        return x

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), input.size(1), 1, 1, 1)
class Generative_model_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 ):
        super(Generative_model_2D, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        
        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()
        
        if self.is_use_u == 1:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1536, 256),
                nn.Linear(256, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1536, 256),
                nn.Linear(256, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 256),
                nn.Linear(256, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 256),
                nn.Linear(256, self.zs_dim))
        
        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()
    
    def forward(self, x, u, feature=0):
        mu, logvar = self.encode(x, u)
        mu_prior, logvar_piror = self.encode_prior(u)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        if feature == 1:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror
    
    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        pred_y = self.Dec_y(mu[:, int(self.zs_dim / 2):])
        return pred_y
    
    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            us = self.Enc_us(us)
            concat = torch.cat([x, u, us], dim=1)
        else:
            # print('not use u and us')
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)
    
    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        us = self.Enc_us_prior(us)
        concat = torch.cat([u, us], dim=1)
        return self.mean_zs_prior(concat), self.sigma_zs_prior(concat)
    
    def decode_x(self, zs):
        return self.Dec_x(zs)
    
    def decode_y(self, s):
        return self.Dec_y(s)
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)
    
    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(),
            nn.Upsample(6),
            self.TConv_bn_ReLU(in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim / 2), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )
    
    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool3d(1),
            Flatten(),
        )
    
    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )
    
    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer
    
    def TConv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0,
                      bias=True, groups=1):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                               groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer
    
    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


import math
from torch import nn



class ESPCN(nn.Module):
    def __init__(self, scale_factor, num_channels=128, mid_channel=2048, dw_type='conv',final_tanh=0):
        super(ESPCN, self).__init__()
        if final_tanh == 0:
            self.first_part = nn.Sequential(
                nn.Conv2d(num_channels, mid_channel, kernel_size=5, padding=5 // 2),
                nn.ReLU(),
                nn.Conv2d(mid_channel, mid_channel // 2, kernel_size=3, padding=3 // 2),
                nn.ReLU(),
            )
        elif final_tanh == 1:
            self.first_part = nn.Sequential(
                nn.Conv2d(num_channels, mid_channel, kernel_size=5, padding=5 // 2),
                nn.Tanh(),
                nn.Conv2d(mid_channel, mid_channel // 2, kernel_size=3, padding=3 // 2),
                nn.Tanh(),
            )
        elif final_tanh == 2:
            self.first_part = nn.Sequential(
                nn.Conv2d(num_channels, mid_channel, kernel_size=5, padding=5 // 2),
                nn.Sigmoid(),
                nn.Conv2d(mid_channel, mid_channel // 2, kernel_size=3, padding=3 // 2),
                nn.Sigmoid(),
            )
        elif final_tanh == 3:
            self.first_part = nn.Sequential(
                nn.Conv2d(num_channels, mid_channel, kernel_size=5, padding=5 // 2),
                nn.LeakyReLU(),
                nn.Conv2d(mid_channel, mid_channel // 2, kernel_size=3, padding=3 // 2),
                nn.LeakyReLU(),
            )
        elif final_tanh == 4:
            self.first_part = nn.Sequential(
                nn.Conv2d(num_channels, mid_channel, kernel_size=5, padding=5 // 2),
                nn.ReLU(),
                nn.Conv2d(mid_channel, mid_channel // 2, kernel_size=3, padding=3 // 2),
                nn.ReLU(),
            )
        self.last_part = nn.Sequential(
            nn.Conv2d(mid_channel // 2, num_channels * (pow(scale_factor, 2)), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
        )

        self.mid_ch = mid_channel
        self.dw_type = dw_type
        self._initialize_weights()
        self.scale_factor = scale_factor

        if self.dw_type == 'conv':
            a = 0
            if scale_factor == 2:
                self.down = nn.Sequential(
                    # input batch_size*128*256*256 to btc*128*128*128
                    nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1),
                    # down 2
                    nn.BatchNorm2d(num_features=num_channels),
                    nn.ReLU(inplace=True),

                )
            elif scale_factor == 4:
                self.down = nn.Sequential(
                    # input batch_size*128*256*256 to btc*128*128*128
                    nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1),
                    # down 2
                    nn.BatchNorm2d(num_features=num_channels),
                    nn.ReLU(inplace=True),

                    # input batch_size*128*128*128 to btc*128*64*64
                    nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=4, stride=2, padding=1),
                    # down 2
                    # up2:4-2-1 up4:4-4-0 8-4-2
                    nn.BatchNorm2d(num_features=num_channels),
                    nn.ReLU(inplace=True),
                )
                pass
        elif self.dw_type == 'avg':
            if scale_factor == 2:
                self.down = nn.Sequential(
                    # input btc*128*128*128 to btc*128*64*64
                    nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_features=num_channels),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(4, stride=2, padding=1),  # down 2
                )
            elif scale_factor == 4:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_features=num_channels),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(4, stride=2, padding=1),  # down 2

                    # input batch_size*1024*4*4 to btc*512*8*8
                    nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
                    # up2:4-2-1 up4:4-4-0 8-4-2
                    nn.BatchNorm2d(num_features=num_channels),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(4, stride=2, padding=1),  # down 2
                )
                pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == (self.mid_ch) // 2:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0,
                                    std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)

        x = self.last_part(x)
        if self.scale_factor == 1:
            pass
        else:
            x = self.down(x)

        return x