""" Full assembly of the parts to form the complete network """
import torch.nn as nn
import torch.nn.functional as F
import torch


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
    
class DoubleConv_DS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
	        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
	        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
	        nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pad1 = 0
        self.pad2 = 0
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, padding=[self.pad1, self.pad2]),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        '''
        if not (x.size(2) /2)% 2 == 0:
            print((x.size(2) /2)%2)
            self.pad1 = 1
        if not (x.size(3) /2)% 2 == 0:
            print((x.size(3) /2)%2)
            self.pad2 = 1
        '''
        return self.maxpool_conv(x)
    
class Down_DS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_DS(in_channels, out_channels)
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
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2) ###### kernel size is modified by sijie (2 -> 3)

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
        #print('x5 up :',x1.size())
        #print(x2.size())
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        ###print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)



class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        ###print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512, 512, 512, bilinear)
        self.up2 = Up(512, 256, 256, bilinear)
        self.up3 = Up(256, 128, 128, bilinear)
        self.up4 = Up(128, 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.out_activ = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x) #x1 64*128*128
        x2 = self.down1(x1)  # x2 128*64*64
        x3 = self.down2(x2)  # x3 256 x 32 x 32
        x4 = self.down3(x3)  # x4 512 x 16 x 16
        x5 = self.down4(x4)  # x5 512 x 8 x 8

        x = self.up1(x5, x4)  # x 256 # (1536)
        x = self.up2(x, x3)  # x 128
        x = self.up3(x, x2)  # x
        x = self.up4(x, x1) #x 128
        logits = self.outc(x)  # x 128*64*64
        logits = self.out_activ(logits)

        return logits



class UNet_SmaAt(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_SmaAt, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.CBAM1 = CBAM(64)
        self.down1 = Down(64, 128)
        self.CBAM2 = CBAM(128)
        self.down2 = Down(128, 256)
        self.CBAM3 = CBAM(256)
        self.down3 = Down(256, 512)
        self.CBAM4 = CBAM(512)
        self.down4 = Down(512, 512)
        self.CBAM5 = CBAM(512)
        self.up1 = Up(512, 512, 512, bilinear)
        self.up2 = Up(512, 256, 256, bilinear)
        self.up3 = Up(256, 128, 128, bilinear)
        self.up4 = Up(128, 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.out_activ = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x) #x1 64*128*128
        x11 = self.CBAM1(x1)
        x2 = self.down1(x1)  # x2 128*64*64
        x22 = self.CBAM2(x2)
        x3 = self.down2(x2)  # x3 256 x 32 x 32
        x33 = self.CBAM3(x3)
        x4 = self.down3(x3)  # x4 512 x 16 x 16
        x44 = self.CBAM4(x4)
        x5 = self.down4(x4)  # x5 512 x 8 x 8
        x55 = self.CBAM5(x5)
        #z = self.conv_blocks1(z)  # 512 x 16 x 16
        #x6 = torch.cat((x5, z), 1)  # 1024 x 16 x 16
        x = self.up1(x55, x44)  # x 256 # (1536)
        x = self.up2(x, x33)  # x 128
        x = self.up3(x, x22)  # x
        x = self.up4(x, x11) #x 128
        logits = self.outc(x)  # x 128*64*64
        logits = self.out_activ(logits)
        return logits


class UNet_residual(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_residual, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 128)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, 512, bilinear)
        self.up2 = Up(512, 256, 256, bilinear)
        self.up3 = Up(256, 128, 128, bilinear)
        self.up4 = Up(128, 128, 64, bilinear)
        self.outc = OutConv(128, n_classes)
        '''
        self.conv_blocks1 = nn.Sequential(

            # input batch_size*516*1*1
            nn.ConvTranspose2d(in_channels=106, out_channels=1024, kernel_size=4, stride=1, padding=0),  # up4
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),

            # input batch_size*1024*4*4 to btc*512*8*8
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            # up2:4-2-1 up4:4-4-0 8-4-2
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
        )
        '''
        self.conv_tanh = nn.Sequential(
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.inc(x) #x1 64*128*128
        x2 = self.down1(x1)  # x2 128*64*64
        x3 = self.down2(x2)  # x3 256 x 32 x 32
        x4 = self.down3(x3)  # x4 512 x 16 x 16
        x5 = self.down4(x4)  # x5 512 x 8 x 8
        #z = self.conv_blocks1(z)  # 512 x 16 x 16
        #x6 = torch.cat((x5, z), 1)  # 1024 x 16 x 16
        x = self.up1(x5, x4)  # x 256 # (1536)
        x = self.up2(x, x3)  # x 128
        x = self.up3(x, x2)  # x
        x = self.up4(x, x1) #x 128
        logits = self.outc(x)  # x 128*64*64
        #logits = self.conv_tanh(logits)
        return logits
    


