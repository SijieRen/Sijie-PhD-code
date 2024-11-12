import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


class Generative_model(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 ):

        super(Generative_model, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()
        self.Enc_us = self.get_Enc_us()

        if self.is_use_u == 1:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(2048, 128),
                nn.Linear(128, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(2048, 128),
                nn.Linear(128, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 128),
                nn.Linear(128, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 128),
                nn.Linear(128, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.Enc_us_prior = self.get_Enc_us()
        self.mean_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(1024, 128),
            nn.Linear(128, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(1024, 128),
            nn.Linear(128, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z[:, int(self.zs_dim/2):])
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
            #print('not use u and us')
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
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
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
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
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

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
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
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class Generative_model_u(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 ):

        super(Generative_model_u, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()

        if self.is_use_u == 1:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1536, 512),
                nn.Linear(512, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1536, 512),
                nn.Linear(512, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 512),
                nn.Linear(512, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 512),
                nn.Linear(512, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        # self.Enc_us_prior = self.get_Enc_us()
        self.mean_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(512, 512),
            nn.Linear(512, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(512, 512),
            nn.Linear(512, self.zs_dim))

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

    def get_pred_y(self, x, u):
        mu, logvar = self.encode(x, u)
        pred_y = self.Dec_y(mu[:, int(self.zs_dim / 2):])
        return pred_y

    def encode(self, x, u):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            concat = torch.cat([x, u], dim=1)
        else:
            # print('not use u and us')
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u):
        concat = self.Enc_u_prior(u)
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
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
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
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
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

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
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


class sVAE_u(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 ):

        super(sVAE_u, self).__init__()
        print('sVAE_u model with zs dim', zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()

        if self.is_use_u == 1:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1536, 512),
                nn.Linear(512, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1536, 512),
                nn.Linear(512, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 512),
                nn.Linear(512, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 512),
                nn.Linear(512, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        # self.Enc_us_prior = self.get_Enc_us()
        self.mean_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(512, 512),
            nn.Linear(512, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(512, 512),
            nn.Linear(512, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, feature=0):
        mu, logvar = self.encode(x, u)
        mu_prior, logvar_piror = self.encode_prior(u)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        if feature == 1:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u):
        mu, logvar = self.encode(x, u)
        pred_y = self.Dec_y(mu)
        return pred_y

    def encode(self, x, u):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            concat = torch.cat([x, u], dim=1)
        else:
            # print('not use u and us')
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u):
        concat = self.Enc_u_prior(u)
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
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.zs_dim, 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
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

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
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


class MM_F_u(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim=256
                 ):

        super(MM_F_u, self).__init__()
        print('MM_F_u model with zs dim', zs_dim)
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.relu = nn.ReLU()

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool3d(1),
            Flatten()
        )
        self.get_feature_u = nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512)
        )

        if self.is_use_u == 1:
            self.fc_1 = nn.Linear(1536, self.zs_dim)
            self.fc = nn.Linear(self.zs_dim, num_classes)
        else:
            self.fc_1 = nn.Linear(1024, self.zs_dim)
            self.fc = nn.Linear(self.zs_dim, num_classes)

    def forward(self, x, u, feature=0):
        x = self.get_feature(x)
        if self.is_use_u == 1:
            u = self.get_feature_u(u)
            fea = self.relu(self.fc_1(torch.cat([x, u], dim=1)))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)
        else:
            fea = self.relu(self.fc_1(x))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class sVAE(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 ):

        super(sVAE, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()
        self.Enc_us = self.get_Enc_us()

        if is_use_u == 1:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(2048, 128),
                nn.Linear(128, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(2048, 128),
                nn.Linear(128, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 128),
                nn.Linear(128, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 128),
                nn.Linear(128, self.zs_dim))

        self.Enc_u_prior = self.get_Enc_u()
        self.Enc_us_prior = self.get_Enc_us()
        self.mean_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(1024, 128),
            nn.Linear(128, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(1024, 128),
            nn.Linear(128, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        if feature == 1:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        pred_y = self.Dec_y(mu)
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
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
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

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
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


class MM_F(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 ):

        super(MM_F, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool3d(1),
            Flatten()
        )
        self.get_feature_u = nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim + self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512)
        )
        if self.is_use_u == 1:
            self.fc = nn.Linear(1536, num_classes)
        else:
            self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, u=None, us=None, feature=0):
        x = self.get_feature(x)
        if self.is_use_u == 1 and u is not None:
            u = self.get_feature_u(torch.cat([u, us], dim=1))
            return self.fc(torch.cat([x, u], dim=1))
        else:
            #print('not use u and us')
            return self.fc(x)

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class MM_F_f(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim=256
                 ):

        super(MM_F_f, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.relu = nn.ReLU()

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool3d(1),
            Flatten()
        )
        self.get_feature_u = nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim + self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512)
        )

        if self.is_use_u == 1:
            self.fc_1 = nn.Linear(1536, self.zs_dim)
            self.fc = nn.Linear(self.zs_dim, num_classes)
        else:
            self.fc_1 = nn.Linear(1024, self.zs_dim)
            self.fc = nn.Linear(self.zs_dim, num_classes)

    def forward(self, x, u, us, feature=0):
        x = self.get_feature(x)
        if self.is_use_u == 1:
            u = self.get_feature_u(torch.cat([u, us], dim=1))
            fea = self.relu(self.fc_1(torch.cat([x, u], dim=1)))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)
        else:
            fea = self.relu(self.fc_1(x))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class Generative_model_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 ):

        super(Generative_model_2D, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()

        if self.is_use_u == 1 and self.is_use_us == 1:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(2048, 128),
                nn.Linear(128, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(2048, 128),
                nn.Linear(128, self.zs_dim))
        elif self.is_use_u == 1 and self.is_use_us == 0:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1536, 128),
                nn.Linear(128, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1536, 128),
                nn.Linear(128, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 128),
                nn.Linear(128, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(1024, 128),
                nn.Linear(128, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(512, 128),
            nn.Linear(128, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(512, 128),
            nn.Linear(128, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        if feature == 1:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        if self.is_sample:
            z = self.reparametrize(mu, logvar)
            pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        else:
            pred_y = self.Dec_y(mu[:, int(self.zs_dim / 2):])
        return pred_y

    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            #us = self.Enc_us(us)
            concat = torch.cat([x, u], dim=1)
        else:
            # print('not use u and us')
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        concat = u
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
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
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
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 ):

        super(Generative_model_f_2D, self).__init__()
        print('model: Generative_model_f_2D, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()

        if self.is_use_u == 1 and self.is_use_us == 1:
            self.mean_zs = nn.Sequential(
                nn.Linear(2048, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(2048, self.zs_dim))
        elif self.is_use_u == 1 and self.is_use_us == 0:
            self.mean_zs = nn.Sequential(
                nn.Linear(1536, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(1536, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                nn.Linear(1024, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(1024, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))

        if decoder_type == 0:
            self.Dec_x = self.get_Dec_x()
        else:
            self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        if feature == 1:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        if self.is_sample:
            z = self.reparametrize(mu, logvar)
            pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        else:
            pred_y = self.Dec_y(mu[:, int(self.zs_dim / 2):])
        return pred_y

    def get_x_y(self, z, s):
        rec_x = self.Dec_x(torch.cat([z, s], dim=1))
        pred_y = self.Dec_y(s)
        return rec_x, pred_y

    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            #print(u.size(), self.Enc_u)
            u = self.Enc_u(u)

            # us = self.Enc_us(us)
            concat = torch.cat([x, u], dim=1)
        else:
            # print('not use u and us')
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        concat = u
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
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
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
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_ff_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 ):

        super(Generative_model_ff_2D, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()

        if self.is_use_u == 1 and self.is_use_us == 1:
            self.mean_zs = nn.Sequential(
                nn.Linear(2048, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(2048, self.zs_dim))
        elif self.is_use_u == 1 and self.is_use_us == 0:
            self.mean_zs = nn.Sequential(
                nn.Linear(1536, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(1536, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                nn.Linear(1024, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(1024, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        if feature == 1:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        if self.is_sample:
            z = self.reparametrize(mu, logvar)
            pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        else:
            pred_y = self.Dec_y(mu[:, int(self.zs_dim / 2):])
        return pred_y

    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            # us = self.Enc_us(us)
            concat = torch.cat([x, u], dim=1)
        else:
            # print('not use u and us')
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        concat = u
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
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            nn.Linear(int(self.zs_dim / 2), self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_28(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 ):

        super(Generative_model_f_2D_28, self).__init__()
        print('model: Generative_model_f_2D, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()

        if self.is_use_u == 1 and self.is_use_us == 1:
            self.mean_zs = nn.Sequential(
                nn.Linear(2048, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(2048, self.zs_dim))
        elif self.is_use_u == 1 and self.is_use_us == 0:
            self.mean_zs = nn.Sequential(
                nn.Linear(1536, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(1536, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                nn.Linear(1024, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(1024, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        if feature == 1:
            return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        if self.is_sample:
            z = self.reparametrize(mu, logvar)
            pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        else:
            pred_y = self.Dec_y(mu[:, int(self.zs_dim / 2):])
        return pred_y

    def get_x_y(self, z, s):
        rec_x = self.Dec_x(torch.cat([z, s], dim=1))
        pred_y = self.Dec_y(s)
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            # print(u.size(), self.Enc_u)
            u = self.Enc_u(u)

            # us = self.Enc_us(us)
            concat = torch.cat([x, u], dim=1)
        else:
            # print('not use u and us')
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        concat = u
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
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
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
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_shared=0,
                 more_layer=0,
                 ):

        super(Generative_model_f_2D_unpooled, self).__init__()
        print('model: Generative_model_f_2D_unpooled, zs_dim: %d, more shared' %
              zs_dim, more_shared, more_layer)
        self.more_shared = more_shared
        self.more_layer = more_layer
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        if decoder_type == 0:
            self.Enc_x = self.get_Enc_x()
        else:
            self.Enc_x = self.get_Enc_x_28()
        self.Enc_u = self.get_Enc_u()

        # construct the [env_idx][y_idx] model
        self.mean_zs = []
        self.logvar_zs = []
        self.phi_z = []
        self.phi_s = []
        for env_idx in range(self.total_env):
            one_mean_zs = []
            one_logvar_zs = []
            for y_idx in range(self.num_classes):
                if self.more_layer == 0:
                    one_mean_zs.append(
                        nn.Sequential(
                            nn.Linear(1024, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, self.zs_dim)
                        )
                    )
                    one_logvar_zs.append(
                        nn.Sequential(
                            nn.Linear(1024, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, self.zs_dim)
                        )
                    )
                elif self.more_layer == 1:
                    one_mean_zs.append(
                        nn.Sequential(
                            nn.Linear(1024, 2048),
                            nn.ReLU(),
                            nn.Linear(2048, 2048),
                            nn.ReLU(),
                            nn.Linear(2048, self.zs_dim)
                        )
                    )
                    one_logvar_zs.append(
                        nn.Sequential(
                            nn.Linear(1024, 2048),
                            nn.ReLU(),
                            nn.Linear(2048, 2048),
                            nn.ReLU(),
                            nn.Linear(2048, self.zs_dim)
                        )
                    )
                if self.more_layer == 0:
                    self.phi_z.append(
                        nn.Sequential(
                            #self.Fc_bn_ReLU(1536, 1024),
                            nn.Linear(self.zs_dim//2, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, self.zs_dim//2)
                        )
                    )
                    self.phi_s.append(
                        nn.Sequential(
                            #self.Fc_bn_ReLU(1536, 1024),
                            nn.Linear(self.zs_dim//2, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, self.zs_dim//2)
                        )
                    )
                elif self.more_layer == 1:
                    self.phi_z.append(
                        nn.Sequential(
                            # self.Fc_bn_ReLU(1536, 1024),
                            nn.Linear(self.zs_dim // 2, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, 2048),
                            nn.ReLU(),
                            nn.Linear(2048, self.zs_dim // 2)
                        )
                    )
                    self.phi_s.append(
                        nn.Sequential(
                            # self.Fc_bn_ReLU(1536, 1024),
                            nn.Linear(self.zs_dim // 2, 1024),
                            nn.ReLU(),
                            nn.Linear(1024, 2048),
                            nn.ReLU(),
                            nn.Linear(2048, self.zs_dim // 2)
                        )
                    )

            self.mean_zs.append(nn.ModuleList(one_mean_zs))
            self.logvar_zs.append(nn.ModuleList(one_logvar_zs))
        self.phi_z = nn.ModuleList(self.phi_z)
        self.phi_s = nn.ModuleList(self.phi_s)
        self.mean_zs = nn.ModuleList(self.mean_zs)
        self.logvar_zs = nn.ModuleList(self.logvar_zs)
        if self.more_shared == 1:
            self.shared_z_layer = nn.Sequential(
                self.Fc_bn_ReLU(self.zs_dim//2, 2048),
                nn.Linear(2048, self.zs_dim//2)
            )
            self.shared_s_layer = nn.Sequential(
                self.Fc_bn_ReLU(self.zs_dim // 2, 2048),
                nn.Linear(2048, self.zs_dim//2)
            )
        if decoder_type == 0:
            self.Dec_x = self.get_Dec_x()
        else:
            self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, x, env, y, feature=0):
        x = self.Enc_x(x)
        for i in range(x.size(0)):
            if i == 0:
                mu, logvar = self.encode(x[i].unsqueeze(0), env[i], y[i])
            else:
                mu_t, logvar_t = self.encode(x[i].unsqueeze(0), env[i], y[i])
                mu = torch.cat([mu, mu_t], dim=0)
                logvar = torch.cat([logvar, logvar_t], dim=0)
        zs = self.reparametrize(mu, logvar)
        for i in range(x.size(0)):
            if i == 0:
                z = self.phi_z[env[i]](zs[i, :self.zs_dim//2].unsqueeze(0))
                s = self.phi_s[env[i]](zs[i, self.zs_dim // 2:].unsqueeze(0))
            else:
                z_t = self.phi_z[env[i]](zs[i, :self.zs_dim // 2].unsqueeze(0))
                s_t = self.phi_s[env[i]](zs[i, self.zs_dim // 2:].unsqueeze(0))
                z = torch.cat([z, z_t], dim=0)
                s = torch.cat([s, s_t], dim=0)
        if self.more_shared == 1:
            zs = torch.cat([self.shared_z_layer(
                z), self.shared_s_layer(s)], dim=1)
        else:
            zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.zs_dim//2:])
        if feature == 1:
            return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s, zs
        else:
            return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s

    def get_x_y(self, z, s):
        if self.more_shared == 1:
            zs = torch.cat([self.shared_z_layer(
                z), self.shared_s_layer(s)], dim=1)
        else:
            zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def encode(self, x, env_idx, y_idx):
        return self.mean_zs[env_idx][y_idx](x), self.logvar_zs[env_idx][y_idx](x)

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
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
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
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_layer=0,
                 more_shared=0,
                 ):

        super(Generative_model_f_2D_unpooled_env, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.more_layer = more_layer
        self.more_shared = more_shared

        if decoder_type == 0:
            self.Enc_x = self.get_Enc_x()
        else:
            self.Enc_x = self.get_Enc_x_28()
        self.Enc_u = self.get_Enc_u()

        # construct the [env_idx][y_idx] model
        self.mean_zs = []
        self.logvar_zs = []
        self.phi_z = []
        self.phi_s = []
        for env_idx in range(self.total_env):
            if self.more_layer == 0:
                self.mean_zs.append(
                    nn.Sequential(
                        # nn.Linear(1024, 1024),
                        # nn.ReLU(),
                        self.Fc_bn_ReLU(1024, 1024),
                        nn.Linear(1024, self.zs_dim)
                    )
                )
                self.logvar_zs.append(
                    nn.Sequential(
                        # nn.Linear(1024, 1024),
                        # nn.ReLU(),
                        self.Fc_bn_ReLU(1024, 1024),
                        nn.Linear(1024, self.zs_dim)
                    )
                )
            elif self.more_layer == 1:
                self.mean_zs.append(
                    nn.Sequential(
                        # nn.Linear(1024, 2048),
                        # nn.ReLU(),
                        # nn.Linear(2048, 2048),
                        # nn.ReLU(),
                        self.Fc_bn_ReLU(1024, 2048),
                        self.Fc_bn_ReLU(2048, 2048),
                        nn.Linear(2048, self.zs_dim)
                    )
                )
                self.logvar_zs.append(
                    nn.Sequential(
                        # nn.Linear(1024, 2048),
                        # nn.ReLU(),
                        # nn.Linear(2048, 2048),
                        # nn.ReLU(),
                        self.Fc_bn_ReLU(1024, 2048),
                        self.Fc_bn_ReLU(2048, 2048),
                        nn.Linear(2048, self.zs_dim)
                    )
                )

            if self.more_layer == 0:
                self.phi_z.append(
                    nn.Sequential(
                        # self.Fc_bn_ReLU(1536, 1024),
                        nn.Linear(self.zs_dim // 2, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, self.zs_dim // 2)
                    )
                )
                self.phi_s.append(
                    nn.Sequential(
                        # self.Fc_bn_ReLU(1536, 1024),
                        nn.Linear(self.zs_dim // 2, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, self.zs_dim // 2)
                    )
                )
            elif self.more_layer == 1:
                self.phi_z.append(
                    nn.Sequential(
                        # self.Fc_bn_ReLU(1536, 1024),
                        nn.Linear(self.zs_dim // 2, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, self.zs_dim // 2)
                    )
                )
                self.phi_s.append(
                    nn.Sequential(
                        # self.Fc_bn_ReLU(1536, 1024),
                        nn.Linear(self.zs_dim // 2, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, self.zs_dim // 2)
                    )
                )

        self.phi_z = nn.ModuleList(self.phi_z)
        self.phi_s = nn.ModuleList(self.phi_s)
        self.mean_zs = nn.ModuleList(self.mean_zs)
        self.logvar_zs = nn.ModuleList(self.logvar_zs)
        if self.more_shared == 1:
            self.shared_z_layer = nn.Sequential(
                self.Fc_bn_ReLU(self.zs_dim // 2, 2048),
                nn.Linear(2048, self.zs_dim // 2)
            )
            self.shared_s_layer = nn.Sequential(
                self.Fc_bn_ReLU(self.zs_dim // 2, 2048),
                nn.Linear(2048, self.zs_dim // 2)
            )
        if decoder_type == 0:
            self.Dec_x = self.get_Dec_x()
        else:
            self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env, feature=0):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = self.phi_z[env](zs[:, :self.zs_dim // 2])
        s = self.phi_s[env](zs[:, self.zs_dim // 2:])
        if self.more_shared == 1:
            zs = torch.cat([self.shared_z_layer(
                z), self.shared_s_layer(s)], dim=1)
        else:
            zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        if self.decoder_type == 1:
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s
        else:
            if feature == 1:
                return pred_y, rec_x, mu, logvar, z, s, zs
            else:
                return pred_y, rec_x, mu, logvar, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = self.phi_z[env](zs[:, :self.zs_dim // 2])
        s = self.phi_s[env](zs[:, self.zs_dim // 2:])
        if self.more_shared == 1:
            zs = torch.cat([self.shared_z_layer(
                z), self.shared_s_layer(s)], dim=1)
        else:
            zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        return pred_y

    def get_x_y(self, z, s):
        if self.more_shared == 1:
            zs = torch.cat([self.shared_z_layer(
                z), self.shared_s_layer(s)], dim=1)
        else:
            zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        if self.decoder_type == 1:
            return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y
        else:
            return rec_x, pred_y

    def encode(self, x, env_idx):
        return self.mean_zs[env_idx](x), self.logvar_zs[env_idx](x)

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
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
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
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_y(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_layer=0,
                 more_shared=0,
                 ):

        super(Generative_model_f_2D_unpooled_y, self).__init__()
        print('model: Generative_model_f_2D_unpooled_y, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.more_layer = more_layer
        self.more_shared = more_shared
        if decoder_type == 0:
            self.Enc_x = self.get_Enc_x()
        else:
            self.Enc_x = self.get_Enc_x_28()
        self.Enc_u = self.get_Enc_u()

        # construct the [env_idx][y_idx] model
        self.mean_zs = []
        self.logvar_zs = []
        self.phi_z = []
        self.phi_s = []
        for env_idx in range(self.total_env):
            self.mean_zs.append(
                nn.Sequential(
                    nn.Linear(1024, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2048),
                    nn.ReLU(),
                    # self.Fc_bn_ReLU(1024, 2048),
                    # self.Fc_bn_ReLU(2048, 2048),
                    nn.Linear(2048, self.zs_dim)
                )
            )
            self.logvar_zs.append(
                nn.Sequential(
                    nn.Linear(1024, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2048),
                    nn.ReLU(),
                    # self.Fc_bn_ReLU(1024, 2048),
                    # self.Fc_bn_ReLU(2048, 2048),
                    nn.Linear(2048, self.zs_dim)
                )
            )

            self.phi_z.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    # self.Fc_bn_ReLU(self.zs_dim // 2, 1024),
                    # self.Fc_bn_ReLU(1024, 2048),
                    nn.Linear(self.zs_dim // 2, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, self.zs_dim // 2)
                )
            )
        for idx in range(self.num_classes):
            self.phi_s.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.zs_dim // 2, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 2048),
                    nn.ReLU(),
                    # self.Fc_bn_ReLU(self.zs_dim // 2, 1024),
                    # self.Fc_bn_ReLU(1024, 2048),
                    nn.Linear(2048, self.zs_dim // 2)
                )
            )
        self.phi_z = nn.ModuleList(self.phi_z)
        self.phi_s = nn.ModuleList(self.phi_s)
        self.mean_zs = nn.ModuleList(self.mean_zs)
        self.logvar_zs = nn.ModuleList(self.logvar_zs)
        if self.more_shared == 1:
            self.shared_z_layer = nn.Sequential(
                self.Fc_bn_ReLU(self.zs_dim // 2, 2048),
                nn.Linear(2048, self.zs_dim // 2)
            )
            self.shared_s_layer = nn.Sequential(
                self.Fc_bn_ReLU(self.zs_dim // 2, 2048),
                nn.Linear(2048, self.zs_dim // 2)
            )
        if decoder_type == 0:
            self.Dec_x = self.get_Dec_x()
        else:
            self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, x, env, y, feature=0):
        x = self.Enc_x(x)
        for i in range(x.size(0)):
            if i == 0:
                mu, logvar = self.encode(x[i].unsqueeze(0), env[i], y[i])
            else:
                mu_t, logvar_t = self.encode(x[i].unsqueeze(0), env[i], y[i])
                mu = torch.cat([mu, mu_t], dim=0)
                logvar = torch.cat([logvar, logvar_t], dim=0)
        zs = self.reparametrize(mu, logvar)
        for i in range(x.size(0)):
            if i == 0:
                z = self.phi_z[env[i]](zs[i, :self.zs_dim // 2].unsqueeze(0))
                s = self.phi_s[y[i]](zs[i, self.zs_dim // 2:].unsqueeze(0))
            else:
                z_t = self.phi_z[env[i]](zs[i, :self.zs_dim // 2].unsqueeze(0))
                s_t = self.phi_s[y[i]](zs[i, self.zs_dim // 2:].unsqueeze(0))
                z = torch.cat([z, z_t], dim=0)
                s = torch.cat([s, s_t], dim=0)
        if self.more_shared == 1:
            zs = torch.cat([self.shared_z_layer(
                z), self.shared_s_layer(s)], dim=1)
        else:
            zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        if feature == 1:
            return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s, zs
        else:
            return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s

    def get_x_y(self, z, s):
        if self.more_shared == 1:
            zs = torch.cat([self.shared_z_layer(
                z), self.shared_s_layer(s)], dim=1)
        else:
            zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def encode(self, x, env_idx, y_idx):
        return self.mean_zs[env_idx](x), self.logvar_zs[env_idx](x)

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
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
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
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env_t(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_layer=0,
                 more_shared=0,
                 is_bn=1,
                 args=None,
                 is_cuda=1
                 ):

        super(Generative_model_f_2D_unpooled_env_t, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_test, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.more_layer = more_layer
        self.more_shared = more_shared
        self.args = args
        self.is_cuda = is_cuda
        self.is_bn = is_bn

        if decoder_type == 0:
            self.Enc_x = self.get_Enc_x()
        else:
            self.Enc_x = self.get_Enc_x_28()
        self.Enc_u = self.get_Enc_u()

        # construct the [env_idx][y_idx] model
        self.mean_zs = []
        self.logvar_zs = []
        self.phi_z = []
        self.phi_s = []
        for env_idx in range(self.total_env):
            if is_bn == 0:
                self.mean_zs.append(
                    nn.Sequential(
                        nn.Linear(1024, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, self.zs_dim)
                    )
                )
                self.logvar_zs.append(
                    nn.Sequential(
                        nn.Linear(1024, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, self.zs_dim)
                    )
                )
            else:
                self.mean_zs.append(
                    nn.Sequential(
                        # nn.Linear(1024, 1024),
                        # nn.ReLU(),
                        self.Fc_bn_ReLU(1024, 1024),
                        nn.Linear(1024, self.zs_dim)
                    )
                )
                self.logvar_zs.append(
                    nn.Sequential(
                        # nn.Linear(1024, 1024),
                        # nn.ReLU(),
                        self.Fc_bn_ReLU(1024, 1024),
                        nn.Linear(1024, self.zs_dim)
                    )
                )

            self.phi_z.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.zs_dim // 2, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, self.zs_dim // 2)
                )
            )
            self.phi_s.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.zs_dim // 2, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, self.zs_dim // 2)
                )
            )

        self.phi_z = nn.ModuleList(self.phi_z)
        self.phi_s = nn.ModuleList(self.phi_s)
        self.mean_zs = nn.ModuleList(self.mean_zs)
        self.logvar_zs = nn.ModuleList(self.logvar_zs)
        if decoder_type == 0:
            self.Dec_x = self.get_Dec_x()
        else:
            self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:
            # test only
            # init
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        #pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = self.phi_z[env_idx](zs[:, :self.zs_dim // 2])
                        s = self.phi_s[env_idx](zs[:, self.zs_dim // 2:])
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        if self.decoder_type == 1:
                            recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:
                            z_init, s_init = z, s
                            min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                        else:
                            new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                              (raw_x * 0.5 + 0.5).view(-1,
                                                                                       3 * self.args.image_size ** 2),
                                                              reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True
            #print('self.args.lr2, self.args.reg2, self.args.test_ep,self.args.sample_num', self.args.lr2, self.args.reg2, self.args.test_ep,self.args.sample_num)
            optimizer = optim.Adam(
                params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)

            for i in range(self.args.test_ep):
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2), (raw_x * 0.5 + 0.5).view(-1, 3 * self.args.image_size ** 2),
                                             reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = self.phi_s[0](zs[:, self.zs_dim // 2:])
            return self.Dec_y(s)
        else:
            mu, logvar = self.encode(x, env)
            zs = self.reparametrize(mu, logvar)
            z = self.phi_z[env](zs[:, :self.zs_dim // 2])
            s = self.phi_s[env](zs[:, self.zs_dim // 2:])
            if self.more_shared == 1:
                zs = torch.cat([self.shared_z_layer(
                    z), self.shared_s_layer(s)], dim=1)
            else:
                zs = torch.cat([z, s], dim=1)
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
            if self.decoder_type == 1:
                if feature == 1:
                    return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s, zs
                else:
                    return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s
            else:
                if feature == 1:
                    return pred_y, rec_x, mu, logvar, z, s, zs
                else:
                    return pred_y, rec_x, mu, logvar, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = self.phi_z[env](zs[:, :self.zs_dim // 2])
        s = self.phi_s[env](zs[:, self.zs_dim // 2:])
        if self.more_shared == 1:
            zs = torch.cat([self.shared_z_layer(
                z), self.shared_s_layer(s)], dim=1)
        else:
            zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        return pred_y

    def get_x_y(self, z, s):
        if self.more_shared == 1:
            zs = torch.cat([self.shared_z_layer(
                z), self.shared_s_layer(s)], dim=1)
        else:
            zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        if self.decoder_type == 1:
            return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y
        else:
            return rec_x, pred_y

    def get_y(self, s):
        return self.Dec_y(s)

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = self.phi_s[env](zs[:, self.zs_dim // 2:])
        return self.Dec_y(s)

    def encode(self, x, env_idx):
        return self.mean_zs[env_idx](x), self.logvar_zs[env_idx](x)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        if self.is_bn == 1:
            return nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim / 2), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes),
                nn.Softmax(dim=1),
            )
        else:
            return nn.Sequential(
                # self.Fc_bn_ReLU(int(self.zs_dim / 2), 512),
                # self.Fc_bn_ReLU(512, 256),
                nn.Linear(int(self.zs_dim / 2), 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_classes),
                nn.Softmax(dim=1),
            )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env_t_AD(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_layer=0,
                 more_shared=0,
                 args=None,
                 is_cuda=1
                 ):

        super(Generative_model_f_2D_unpooled_env_t_AD, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_AD, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.more_layer = more_layer
        self.more_shared = more_shared
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 1024

        self.Enc_x = self.get_Enc_x()

        self.mean_zs = []
        self.logvar_zs = []
        self.phi_z = []
        self.phi_s = []
        for env_idx in range(self.total_env):
            self.mean_zs.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )
            self.logvar_zs.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )

            self.phi_z.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.zs_dim // 2, self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, self.zs_dim // 2)
                )
            )
            self.phi_s.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.zs_dim // 2, self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, self.zs_dim // 2)
                )
            )

        self.phi_z = nn.ModuleList(self.phi_z)
        self.phi_s = nn.ModuleList(self.phi_s)
        self.mean_zs = nn.ModuleList(self.mean_zs)
        self.logvar_zs = nn.ModuleList(self.logvar_zs)
        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = self.phi_z[env_idx](zs[:, :self.zs_dim // 2])
                        s = self.phi_s[env_idx](zs[:, self.zs_dim // 2:])
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)

                        if z_init is None:
                            z_init, s_init = z, s
                            min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                        else:
                            new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                              (raw_x * 0.5 + 0.5).view(-1,
                                                                                       3 * self.args.image_size ** 2),
                                                              reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True
            optimizer = optim.Adam(
                params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)

            for i in range(self.args.test_ep):
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                             (raw_x * 0.5 + 0.5).view(-1,
                                                                      3 * self.args.image_size ** 2),
                                             reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = self.phi_s[0](zs[:, self.zs_dim // 2:])
            return self.Dec_y(s)
        else:
            mu, logvar = self.encode(x, env)
            zs = self.reparametrize(mu, logvar)
            z = self.phi_z[env](zs[:, :self.zs_dim // 2])
            s = self.phi_s[env](zs[:, self.zs_dim // 2:])
            zs = torch.cat([z, s], dim=1)
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
            if feature == 1:
                return pred_y, rec_x, mu, logvar, z, s, zs
            else:
                return pred_y, rec_x, mu, logvar, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = self.phi_z[env](zs[:, :self.zs_dim // 2])
        s = self.phi_s[env](zs[:, self.zs_dim // 2:])
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        return pred_y

    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        return rec_x, pred_y

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = self.phi_s[env](zs[:, self.zs_dim // 2:])
        return self.Dec_y(s)

    def get_y(self, s):
        return self.Dec_y(s)

    def encode(self, x, env_idx):
        return self.mean_zs[env_idx](x), self.logvar_zs[env_idx](x)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(),
            nn.Upsample(6),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
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
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool3d(1),
            Flatten(),
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


class Generative_model_2D_mnist(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 ):

        super(Generative_model_2D_mnist, self).__init__()
        print('Generative_model_2D_mnist, zs_dim', zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.in_plane = 256

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()

        if self.is_use_u == 1 and self.is_use_us == 1:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(768, self.in_plane),
                nn.Linear(self.in_plane, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(768, self.in_plane),
                nn.Linear(self.in_plane, self.zs_dim))
        elif self.is_use_u == 1 and self.is_use_us == 0:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(512, self.in_plane),
                nn.Linear(self.in_plane, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(512, self.in_plane),
                nn.Linear(self.in_plane, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                self.Fc_bn_ReLU(256, self.in_plane),
                nn.Linear(self.in_plane, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                self.Fc_bn_ReLU(256, self.in_plane),
                nn.Linear(self.in_plane, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(256, self.in_plane),
            nn.Linear(self.in_plane, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(256, self.in_plane),
            nn.Linear(self.in_plane, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        rec_x = rec_x[:, :, 2:30, 2:30].contiguous()
        pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        if feature == 1:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        if self.is_sample:
            z = self.reparametrize(mu, logvar)
            pred_y = self.Dec_y(z[:, int(self.zs_dim / 2):])
        else:
            pred_y = self.Dec_y(mu[:, int(self.zs_dim / 2):])
        return pred_y

    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            # us = self.Enc_us(us)
            concat = torch.cat([x, u], dim=1)
        else:
            # print('not use u and us')
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        concat = u
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
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
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
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env_t_mnist(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_layer=0,
                 more_shared=0,
                 args=None,
                 is_cuda=1,
                 z_ratio=0.5
                 ):

        super(Generative_model_f_2D_unpooled_env_t_mnist, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.more_layer = more_layer
        self.more_shared = more_shared
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256
        self.z_dim = int(round(zs_dim * args.z_ratio))
        self.Enc_x = self.get_Enc_x_28()
        print('z_dim is ', self.z_dim)
        self.mean_zs = []
        self.logvar_zs = []
        self.phi_z = []
        self.phi_s = []
        for env_idx in range(self.total_env):
            self.mean_zs.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )
            self.logvar_zs.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )

            self.phi_z.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.z_dim, self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )
            self.phi_s.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(int(self.zs_dim - self.z_dim), self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, int(self.zs_dim - self.z_dim))
                )
            )

        self.phi_z = nn.ModuleList(self.phi_z)
        self.phi_s = nn.ModuleList(self.phi_s)
        self.mean_zs = nn.ModuleList(self.mean_zs)
        self.logvar_zs = nn.ModuleList(self.logvar_zs)
        self.Dec_x = self.get_Dec_x_28()

        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:
            # test only
            # init
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = self.phi_z[env_idx](zs[:, :self.z_dim])
                        s = self.phi_s[env_idx](zs[:, self.z_dim:])
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:
                            z_init, s_init = z, s
                            if self.args.mse_loss:
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x * 0.5 + 0.5).view(-1,
                                                                                   3 * self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                                               3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                               3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True

            optimizer = optim.Adam(
                params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)

            for i in range(self.args.test_ep):
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                if self.args.mse_loss:
                    BCE = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                     (raw_x * 0.5 + 0.5).view(-1,
                                                              3 * self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x * 0.5 + 0.5).view(-1,
                                                                          3 * self.args.image_size ** 2),
                                                 reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = self.phi_s[0](zs[:, self.z_dim:])
            return self.Dec_y(s)
        else:
            mu, logvar = self.encode(x, env)
            zs = self.reparametrize(mu, logvar)
            z = self.phi_z[env](zs[:, :self.z_dim])
            s = self.phi_s[env](zs[:, self.z_dim:])
            zs = torch.cat([z, s], dim=1)
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(zs[:, self.z_dim:])
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = self.phi_z[env](zs[:, :self.z_dim])
        s = self.phi_s[env](zs[:, self.z_dim:])
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return pred_y

    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def get_y(self, s):
        return self.Dec_y(s)

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = self.phi_s[env](zs[:, self.z_dim:])
        return self.Dec_y(s)

    def encode(self, x, env_idx):
        return self.mean_zs[env_idx](x), self.logvar_zs[env_idx](x)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim - self.z_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env_t_mnist_more_share(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_layer=0,
                 more_shared=0,
                 args=None,
                 is_cuda=1,
                 z_ratio=0.5
                 ):

        super(Generative_model_f_2D_unpooled_env_t_mnist_more_share, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist_more_share, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.more_layer = more_layer
        self.more_shared = more_shared
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256
        self.z_dim = int(round(zs_dim * args.z_ratio))
        self.Enc_x = self.get_Enc_x_28()
        print('z_dim is ', self.z_dim)
        self.mean_zs = []
        self.logvar_zs = []
        self.phi_z = []
        self.phi_s = []
        for env_idx in range(self.total_env):
            temp = nn.Sequential(
                self.Conv_bn_ReLU(128, 256),
                nn.AdaptiveAvgPool2d(1),
                Flatten()
            )
            self.mean_zs.append(
                nn.Sequential(
                    temp,
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )
            self.logvar_zs.append(
                nn.Sequential(
                    temp,
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )

            self.phi_z.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.z_dim, self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )
            self.phi_s.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(int(self.zs_dim - self.z_dim), self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, int(self.zs_dim - self.z_dim))
                )
            )

        self.phi_z = nn.ModuleList(self.phi_z)
        self.phi_s = nn.ModuleList(self.phi_s)
        self.mean_zs = nn.ModuleList(self.mean_zs)
        self.logvar_zs = nn.ModuleList(self.logvar_zs)
        self.Dec_x = self.get_Dec_x_28()

        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:
            # test only
            # init
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = self.phi_z[env_idx](zs[:, :self.z_dim])
                        s = self.phi_s[env_idx](zs[:, self.z_dim:])
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:
                            z_init, s_init = z, s
                            if self.args.mse_loss:
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x * 0.5 + 0.5).view(-1,
                                                                                   3 * self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                                               3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                               3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True

            optimizer = optim.Adam(
                params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)

            for i in range(self.args.test_ep):
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                if self.args.mse_loss:
                    BCE = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                     (raw_x * 0.5 + 0.5).view(-1,
                                                              3 * self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x * 0.5 + 0.5).view(-1,
                                                                          3 * self.args.image_size ** 2),
                                                 reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = self.phi_s[0](zs[:, self.z_dim:])
            return self.Dec_y(s)
        else:
            mu, logvar = self.encode(x, env)
            zs = self.reparametrize(mu, logvar)
            z = self.phi_z[env](zs[:, :self.z_dim])
            s = self.phi_s[env](zs[:, self.z_dim:])
            zs = torch.cat([z, s], dim=1)
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(zs[:, self.z_dim:])
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = self.phi_z[env](zs[:, :self.z_dim])
        s = self.phi_s[env](zs[:, self.z_dim:])
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return pred_y

    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def get_y(self, s):
        return self.Dec_y(s)

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = self.phi_s[env](zs[:, self.z_dim:])
        return self.Dec_y(s)

    def encode(self, x, env_idx):
        return self.mean_zs[env_idx](x), self.logvar_zs[env_idx](x)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim - self.z_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),

        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env_t_mnist_shared_s(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_layer=0,
                 more_shared=0,
                 args=None,
                 is_cuda=1,
                 z_ratio=0.5
                 ):

        super(Generative_model_f_2D_unpooled_env_t_mnist_shared_s, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist_shared_s, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.more_layer = more_layer
        self.more_shared = more_shared
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256
        self.z_dim = int(round(zs_dim * args.z_ratio))
        self.Enc_x = self.get_Enc_x_28()
        print('z_dim is ', self.z_dim)
        self.mean_z = []
        self.logvar_z = []
        self.phi_z = []
        self.phi_s = nn.Sequential(
            # self.Fc_bn_ReLU(1536, 1024),
            nn.Linear(int(self.zs_dim - self.z_dim), self.in_plane),
            nn.ReLU(),
            nn.Linear(self.in_plane, int(self.zs_dim - self.z_dim))
        )
        self.mean_s = nn.Sequential(
            self.Fc_bn_ReLU(self.in_plane, self.in_plane),
            nn.Linear(self.in_plane, self.zs_dim - self.z_dim)
        )
        self.logvar_s = nn.Sequential(
            self.Fc_bn_ReLU(self.in_plane, self.in_plane),
            nn.Linear(self.in_plane, self.zs_dim - self.z_dim)
        )
        for env_idx in range(self.total_env):
            self.mean_z.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )
            self.logvar_z.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )

            self.phi_z.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.z_dim, self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )
        self.phi_z = nn.ModuleList(self.phi_z)
        self.mean_z = nn.ModuleList(self.mean_z)
        self.logvar_z = nn.ModuleList(self.logvar_z)
        self.Dec_x = self.get_Dec_x_28()

        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = self.phi_z[env_idx](zs[:, :self.z_dim])
                        s = self.phi_s(zs[:, self.z_dim:])
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:
                            z_init, s_init = z, s
                            if self.args.mse_loss:
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x * 0.5 + 0.5).view(-1,
                                                                                   3 * self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                                               3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                               3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True

            optimizer = optim.Adam(
                params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)

            for i in range(self.args.test_ep):
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                if self.args.mse_loss:
                    BCE = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                     (raw_x * 0.5 + 0.5).view(-1,
                                                              3 * self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x * 0.5 + 0.5).view(-1,
                                                                          3 * self.args.image_size ** 2),
                                                 reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = self.phi_s(zs[:, self.z_dim:])
            return self.Dec_y(s)
        else:
            mu, logvar = self.encode(x, env)
            zs = self.reparametrize(mu, logvar)
            z = self.phi_z[env](zs[:, :self.z_dim])
            s = self.phi_s(zs[:, self.z_dim:])
            zs = torch.cat([z, s], dim=1)
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(zs[:, self.z_dim:])
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = self.phi_z[env](zs[:, :self.z_dim])
        s = self.phi_s(zs[:, self.z_dim:])
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return pred_y

    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def get_y(self, s):
        return self.Dec_y(s)

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = self.phi_s(zs[:, self.z_dim:])
        return self.Dec_y(s)

    def encode(self, x, env_idx):
        return torch.cat([self.mean_z[env_idx](x), self.mean_s(x)], dim=1), torch.cat([self.logvar_z[env_idx](x), self.logvar_s(x)], dim=1)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim - self.z_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env_t_mnist_prior(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 decoder_type=0,
                 total_env=2,
                 args=None,
                 is_cuda=1
                 ):

        super(Generative_model_f_2D_unpooled_env_t_mnist_prior, self).__init__()
        print(
            'model: Generative_model_f_2D_unpooled_env_t_mnist_prior, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256
        self.z_dim = int(round(zs_dim * args.z_ratio))
        self.Enc_x = self.get_Enc_x_28()
        self.u_dim = total_env
        print('z_dim is ', self.z_dim)
        self.mean_zs = []
        self.logvar_zs = []
        for env_idx in range(self.total_env):
            self.mean_zs.append(
                nn.Sequential(
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )
            self.logvar_zs.append(
                nn.Sequential(
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )

        self.mean_zs = nn.ModuleList(self.mean_zs)
        self.logvar_zs = nn.ModuleList(self.logvar_zs)

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(32, self.zs_dim))
        self.logvar_zs_prior = nn.Sequential(
            nn.Linear(32, self.zs_dim))

        self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:
            # test only
            # init
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = zs[:, :self.z_dim]
                        s = zs[:, self.z_dim:]
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:
                            z_init, s_init = z, s
                            if self.args.mse_loss:
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x * 0.5 + 0.5).view(-1,
                                                                                   3 * self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                                               3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                               3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True

            optimizer = optim.Adam(
                params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)

            for i in range(self.args.test_ep):
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                if self.args.mse_loss:
                    BCE = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                     (raw_x * 0.5 + 0.5).view(-1,
                                                              3 * self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x * 0.5 + 0.5).view(-1,
                                                                          3 * self.args.image_size ** 2),
                                                 reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = zs[:, self.z_dim:]
            return self.Dec_y(s)
        else:
            mu, logvar = self.encode(x, env)
            mu_prior, logvar_prior = self.encode_prior(x, env)
            zs = self.reparametrize(mu, logvar)
            z = zs[:, :self.z_dim]
            s = zs[:, self.z_dim:]
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(s)
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = zs[:, :self.z_dim]
        s = zs[:, self.z_dim:]
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return pred_y

    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def get_y(self, s):
        return self.Dec_y(s)

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = zs[:, self.z_dim:]
        return self.Dec_y(s)

    def encode(self, x, env_idx):

        return self.mean_zs[env_idx](x), self.logvar_zs[env_idx](x)

    def encode_prior(self, x, env_idx):
        temp = env_idx * torch.ones(x.size()[0], 1)
        temp = temp.long().cuda()
        y_onehot = torch.FloatTensor(x.size()[0], self.args.env_num).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, temp, 1)
        # print(env_idx, y_onehot, 'onehot')
        u = self.Enc_u_prior(y_onehot)
        return self.mean_zs_prior(u), self.logvar_zs_prior(u)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim - self.z_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 16),
            self.Fc_bn_ReLU(16, 32)
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


# sijie NIPS 2021
class Generative_model_f_2D_unpooled_env_t_mnist_prior_share(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 decoder_type=0,
                 total_env=2,
                 args=None,
                 is_cuda=1
                 ):

        super(Generative_model_f_2D_unpooled_env_t_mnist_prior_share, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist_prior_share, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256
        self.z_dim = int(round(zs_dim * args.z_ratio))
        self.Enc_x = self.get_Enc_x_28()
        self.u_dim = total_env
        print('z_dim is ', self.z_dim)
        self.s_dim = int(self.zs_im - self.z_dim)
        self.mean_z = []
        self.logvar_z = []
        self.mean_s = []
        self.logvar_s = []
        self.shared_s = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
        for env_idx in range(self.total_env):
            self.mean_z.append(
                nn.Sequential(
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )
            self.logvar_z.append(
                nn.Sequential(
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.z_dim)
                )
            )
            self.mean_s.append(
                nn.Sequential(
                    self.shared_s,
                    nn.Linear(self.in_plane, self.s_dim)
                )
            )
            self.logvar_s.append(
                nn.Sequential(
                    self.shared_s,
                    nn.Linear(self.in_plane, self.s_dim)
                )
            )

        self.mean_z = nn.ModuleList(self.mean_z)
        self.logvar_z = nn.ModuleList(self.logvar_z)
        self.mean_s = nn.ModuleList(self.mean_s)
        self.logvar_s = nn.ModuleList(self.logvar_s)

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(32, self.zs_dim))
        self.logvar_zs_prior = nn.Sequential(
            nn.Linear(32, self.zs_dim))

        self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:  # sijie evaluate_22 is_train=0 is_debug=1
            # test only
            # init
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = zs[:, :self.z_dim]
                        s = zs[:, self.z_dim:]
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:  # sijiez_init, s_init = None, None
                            z_init, s_init = z, s
                            if self.args.mse_loss:  # sijie mse_loss=1
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x * 0.5 + 0.5).view(-1,
                                                                                   3 * self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                                               3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                               3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True

            optimizer = optim.Adam(
                params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)

            for i in range(self.args.test_ep):  # sijie test_ep=100 --> Eq(6)
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                if self.args.mse_loss:  # sijie mse_loss=1
                    BCE = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                     (raw_x * 0.5 + 0.5).view(-1,
                                                              3 * self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x * 0.5 + 0.5).view(-1,
                                                                          3 * self.args.image_size ** 2),
                                                 reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = zs[:, self.z_dim:]
            return self.Dec_y(s)
        else:  # sijie train is_train=1
            mu, logvar = self.encode(x, env)
            mu_prior, logvar_prior = self.encode_prior(x, env)
            zs = self.reparametrize(mu, logvar)
            z = zs[:, :self.z_dim]
            s = zs[:, self.z_dim:]
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(s)
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = zs[:, :self.z_dim]
        s = zs[:, self.z_dim:]
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return pred_y

    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def get_y(self, s):
        return self.Dec_y(s)

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = zs[:, self.z_dim:]
        return self.Dec_y(s)

    def encode(self, x, env_idx):
        return torch.cat([self.mean_z[env_idx](x), self.mean_s[env_idx](x)], dim=1), \
            torch.cat([self.logvar_z[env_idx](x),
                      self.logvar_s[env_idx](x)], dim=1)

    def encode_prior(self, x, env_idx):
        temp = env_idx * torch.ones(x.size()[0], 1)
        temp = temp.long().cuda()
        y_onehot = torch.FloatTensor(x.size()[0], self.args.env_num).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, temp, 1)
        # print(env_idx, y_onehot, 'onehot')
        u = self.Enc_u_prior(y_onehot)
        return self.mean_zs_prior(u), self.logvar_zs_prior(u)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim - self.z_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 16),
            self.Fc_bn_ReLU(16, 32)
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env_t_mnist_prior_more(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 decoder_type=0,
                 total_env=2,
                 args=None,
                 is_cuda=1
                 ):

        super(Generative_model_f_2D_unpooled_env_t_mnist_prior_more, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist_prior_more, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256
        self.z_dim = int(round(zs_dim * args.z_ratio))
        self.Enc_x = self.get_Enc_x_28()
        self.u_dim = total_env
        print('z_dim is ', self.z_dim)
        self.mean_zs = []
        self.logvar_zs = []
        self.env_zs = []
        for env_idx in range(self.total_env):
            temp = nn.Sequential(
                self.Conv_bn_ReLU(128, 256),
                nn.AdaptiveAvgPool2d(1),
                Flatten()
            )
            self.mean_zs.append(
                nn.Sequential(
                    temp,
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )
            self.logvar_zs.append(
                nn.Sequential(
                    temp,
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )
        # self.env_zs = nn.ModuleList(self.env_zs)
        self.mean_zs = nn.ModuleList(self.mean_zs)
        self.logvar_zs = nn.ModuleList(self.logvar_zs)

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(32, self.zs_dim))
        self.logvar_zs_prior = nn.Sequential(
            nn.Linear(32, self.zs_dim))

        self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:
            # test only
            # init
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = zs[:, :self.z_dim]
                        s = zs[:, self.z_dim:]
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:
                            z_init, s_init = z, s
                            if self.args.mse_loss:
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x * 0.5 + 0.5).view(-1,
                                                                                   3 * self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                                               3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                               3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True

            optimizer = optim.Adam(
                params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)

            for i in range(self.args.test_ep):
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                if self.args.mse_loss:
                    BCE = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                     (raw_x * 0.5 + 0.5).view(-1,
                                                              3 * self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x * 0.5 + 0.5).view(-1,
                                                                          3 * self.args.image_size ** 2),
                                                 reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = zs[:, self.z_dim:]
            return self.Dec_y(s)
        else:
            mu, logvar = self.encode(x, env)
            mu_prior, logvar_prior = self.encode_prior(x, env)
            zs = self.reparametrize(mu, logvar)
            z = zs[:, :self.z_dim]
            s = zs[:, self.z_dim:]
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(s)
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = zs[:, :self.z_dim]
        s = zs[:, self.z_dim:]
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return pred_y

    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def get_y(self, s):
        return self.Dec_y(s)

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = zs[:, self.z_dim:]
        return self.Dec_y(s)

    def encode(self, x, env_idx):

        return self.mean_zs[env_idx](x), self.logvar_zs[env_idx](x)

    def encode_prior(self, x, env_idx):
        temp = env_idx * torch.ones(x.size()[0], 1)
        temp = temp.long().cuda()
        y_onehot = torch.FloatTensor(x.size()[0], self.args.env_num).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, temp, 1)
        # print(env_idx, y_onehot, 'onehot')
        u = self.Enc_u_prior(y_onehot)
        return self.mean_zs_prior(u), self.logvar_zs_prior(u)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim - self.z_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 16),
            self.Fc_bn_ReLU(16, 32)
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2)
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env_t_mnist_more(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_layer=0,
                 more_shared=0,
                 args=None,
                 is_cuda=1
                 ):

        super(Generative_model_f_2D_unpooled_env_t_mnist_more, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist_more, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.more_layer = more_layer
        self.more_shared = more_shared
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256

        self.Enc_x = self.get_Enc_x_28()

        self.mean_zs = []
        self.logvar_zs = []
        self.phi_z = []
        self.phi_s = []
        for env_idx in range(self.total_env):
            self.mean_zs.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Conv_bn_ReLU(256, self.in_plane),
                    nn.AdaptiveAvgPool2d(1),
                    Flatten(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )
            self.logvar_zs.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Conv_bn_ReLU(256, self.in_plane),
                    nn.AdaptiveAvgPool2d(1),
                    Flatten(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim)
                )
            )

            self.phi_z.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.zs_dim // 2, self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, self.zs_dim // 2)
                )
            )
            self.phi_s.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.zs_dim // 2, self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, self.zs_dim // 2)
                )
            )

        self.phi_z = nn.ModuleList(self.phi_z)
        self.phi_s = nn.ModuleList(self.phi_s)
        self.mean_zs = nn.ModuleList(self.mean_zs)
        self.logvar_zs = nn.ModuleList(self.logvar_zs)
        self.Dec_x = self.get_Dec_x_28()

        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        raw_x = x
        x = self.Enc_x(x)
        if is_train == 0:
            # test only
            # init
            with torch.no_grad():
                z_init, s_init = None, None
                for env_idx in range(self.args.env_num):
                    mu, logvar = self.encode(x, env_idx)
                    for ss in range(self.args.sample_num):
                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        z = self.phi_z[env_idx](zs[:, :self.zs_dim // 2])
                        s = self.phi_s[env_idx](zs[:, self.zs_dim // 2:])
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:
                            z_init, s_init = z, s
                            if self.args.mse_loss:
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x * 0.5 + 0.5).view(-1,
                                                                                   3 * self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                                               3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x * 0.5 + 0.5).view(-1,
                                                                               3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x * 0.5 + 0.5).view(-1,
                                                                                           3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            if is_debug:
                pred_y_init = self.get_y(s)
            z.requires_grad = True
            s.requires_grad = True

            optimizer = optim.Adam(
                params=[z, s], lr=self.args.lr2, weight_decay=self.args.reg2)

            for i in range(self.args.test_ep):
                optimizer.zero_grad()
                recon_x, _ = self.get_x_y(z, s)
                if self.args.mse_loss:
                    BCE = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                     (raw_x * 0.5 + 0.5).view(-1,
                                                              3 * self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x * 0.5 + 0.5).view(-1,
                                                                          3 * self.args.image_size ** 2),
                                                 reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                return pred_y_init, pred_y
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = self.phi_s[0](zs[:, self.zs_dim // 2:])
            return self.Dec_y(s)
        else:
            mu, logvar = self.encode(x, env)
            zs = self.reparametrize(mu, logvar)
            z = self.phi_z[env](zs[:, :self.zs_dim // 2])
            s = self.phi_s[env](zs[:, self.zs_dim // 2:])
            zs = torch.cat([z, s], dim=1)
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        z = self.phi_z[env](zs[:, :self.zs_dim // 2])
        s = self.phi_s[env](zs[:, self.zs_dim // 2:])
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        return pred_y

    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.zs_dim // 2:])
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def get_y(self, s):
        return self.Dec_y(s)

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        s = self.phi_s[env](zs[:, self.zs_dim // 2:])
        return self.Dec_y(s)

    def encode(self, x, env_idx):
        return self.mean_zs[env_idx](x), self.logvar_zs[env_idx](x)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim / 2), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env_t_mnist_more_indvidual(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_layer=0,
                 more_shared=0,
                 args=None,
                 is_cuda=1,
                 softmax=1,
                 ):

        super(Generative_model_f_2D_unpooled_env_t_mnist_more_indvidual,
              self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist_more_indvidual, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.more_layer = more_layer
        self.more_shared = more_shared
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256
        self.softmax = softmax

        self.Enc_x = self.get_Enc_x_28()

        self.mean_s = nn.Sequential(
            self.Conv_bn_ReLU(256, self.in_plane),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            self.Fc_bn_ReLU(self.in_plane, self.in_plane),
            nn.Linear(self.in_plane, self.zs_dim//2)
        )

        self.logvar_s = nn.Sequential(
            self.Conv_bn_ReLU(256, self.in_plane),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            self.Fc_bn_ReLU(self.in_plane, self.in_plane),
            nn.Linear(self.in_plane, self.zs_dim//2)
        )

        self.phi_s = nn.Sequential(
            # self.Fc_bn_ReLU(1536, 1024),
            nn.Linear(self.zs_dim // 2, self.in_plane),
            nn.ReLU(),
            nn.Linear(self.in_plane, self.zs_dim // 2)
        )

        self.mean_z = []
        self.logvar_z = []
        self.phi_z = []

        for env_idx in range(self.total_env):
            self.mean_z.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Conv_bn_ReLU(256, self.in_plane),
                    nn.AdaptiveAvgPool2d(1),
                    Flatten(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim//2)
                )
            )
            self.logvar_z.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Conv_bn_ReLU(256, self.in_plane),
                    nn.AdaptiveAvgPool2d(1),
                    Flatten(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim//2)
                )
            )

            self.phi_z.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.zs_dim // 2, self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, self.zs_dim // 2)
                )
            )

        self.phi_z = nn.ModuleList(self.phi_z)
        self.mean_z = nn.ModuleList(self.mean_z)
        self.logvar_z = nn.ModuleList(self.logvar_z)
        self.Dec_x = self.get_Dec_x_28()

        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        x = self.Enc_x(x)
        mu_z, logvar_z = self.encode_z(x, env)
        mu_s, logvar_s = self.encode_s(x)
        z = self.reparametrize(mu_z, logvar_z)
        s = self.reparametrize(mu_s, logvar_s)
        z = self.phi_z[env](z)
        s = self.phi_s(s)
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(s)
        if feature == 1:
            return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), torch.cat([mu_z, mu_s], dim=1), \
                torch.cat([logvar_z, logvar_s], dim=1), z, s, zs
        else:
            return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), torch.cat([mu_z, mu_s], dim=1), \
                torch.cat([logvar_z, logvar_s], dim=1), z, s

    def get_pred_y(self, x):
        x = self.Enc_x(x)
        mu_s, logvar_s = self.encode_s(x)
        s = self.reparametrize(mu_s, logvar_s)
        pred_y = self.Dec_y(self.phi_s(s))
        return pred_y

    def get_pred_y_mu(self, x):
        x = self.Enc_x(x)
        mu_s, logvar_s = self.encode_s(x)
        pred_y = self.Dec_y(self.phi_s(mu_s))
        return pred_y

    def get_x_y(self, z, s, env):
        z = self.phi_z[env](z)
        s = self.phi_s(s)
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(s)
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def get_y(self, s):
        return self.Dec_y(self.phi_s(s))

    def get_y_by_zs(self, mu, logvar):
        zs = self.reparametrize(mu, logvar)
        s = self.phi_s(zs[:, self.zs_dim // 2:])
        return self.Dec_y(s)

    def encode_z(self, x, env_idx):
        return self.mean_z[env_idx](x), self.logvar_z[env_idx](x)

    def encode_s(self, x):
        return self.mean_s(x), self.logvar_s(x)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        if self.softmax == 0:
            return nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim / 2), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes)
            )
        else:
            return nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim / 2), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes),
                nn.Softmax(dim=1),
            )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class Generative_model_f_2D_unpooled_env_t_mnist_more_indvidual_small(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 is_sample=0,
                 decoder_type=0,
                 total_env=2,
                 more_layer=0,
                 more_shared=0,
                 args=None,
                 is_cuda=1,
                 softmax=1,
                 ):

        super(
            Generative_model_f_2D_unpooled_env_t_mnist_more_indvidual_small, self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist_more_indvidual_small, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.is_sample = is_sample
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.more_layer = more_layer
        self.more_shared = more_shared
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 64
        self.softmax = softmax

        self.Enc_x = self.get_Enc_x_28()

        self.mean_s = nn.Sequential(
            self.Conv_bn_ReLU(128, self.in_plane),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            self.Fc_bn_ReLU(self.in_plane, self.in_plane),
            nn.Linear(self.in_plane, self.zs_dim // 2)
        )

        self.logvar_s = nn.Sequential(
            self.Conv_bn_ReLU(128, self.in_plane),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            self.Fc_bn_ReLU(self.in_plane, self.in_plane),
            nn.Linear(self.in_plane, self.zs_dim // 2)
        )

        self.phi_s = nn.Sequential(
            # self.Fc_bn_ReLU(1536, 1024),
            nn.Linear(self.zs_dim // 2, self.in_plane),
            nn.ReLU(),
            nn.Linear(self.in_plane, self.zs_dim // 2)
        )

        self.mean_z = []
        self.logvar_z = []
        self.phi_z = []

        for env_idx in range(self.total_env):
            self.mean_z.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Conv_bn_ReLU(128, self.in_plane),
                    nn.AdaptiveAvgPool2d(1),
                    Flatten(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim // 2)
                )
            )
            self.logvar_z.append(
                nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    self.Conv_bn_ReLU(128, self.in_plane),
                    nn.AdaptiveAvgPool2d(1),
                    Flatten(),
                    self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                    nn.Linear(self.in_plane, self.zs_dim // 2)
                )
            )

            self.phi_z.append(
                nn.Sequential(
                    # self.Fc_bn_ReLU(1536, 1024),
                    nn.Linear(self.zs_dim // 2, self.in_plane),
                    nn.ReLU(),
                    nn.Linear(self.in_plane, self.zs_dim // 2)
                )
            )

        self.phi_z = nn.ModuleList(self.phi_z)
        self.mean_z = nn.ModuleList(self.mean_z)
        self.logvar_z = nn.ModuleList(self.logvar_z)
        self.Dec_x = self.get_Dec_x_28()

        self.Dec_y = self.get_Dec_y()
        self.alpha = nn.Parameter(torch.FloatTensor([1.0, 1.0]))

    def forward(self, x, env=0, feature=0, is_train=0, is_debug=0):
        x = self.Enc_x(x)
        mu_z, logvar_z = self.encode_z(x, env)
        mu_s, logvar_s = self.encode_s(x)
        z = self.reparametrize(mu_z, logvar_z)
        s = self.reparametrize(mu_s, logvar_s)
        z = self.phi_z[env](z)
        s = self.phi_s(s)
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(s)
        if feature == 1:
            return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), torch.cat([mu_z, mu_s], dim=1), \
                torch.cat([logvar_z, logvar_s], dim=1), z, s, zs
        else:
            return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), torch.cat([mu_z, mu_s], dim=1), \
                torch.cat([logvar_z, logvar_s], dim=1), z, s

    def get_pred_y(self, x):
        x = self.Enc_x(x)
        mu_s, logvar_s = self.encode_s(x)
        s = self.reparametrize(mu_s, logvar_s)
        pred_y = self.Dec_y(self.phi_s(s))
        return pred_y

    def get_pred_y_mu(self, x):
        x = self.Enc_x(x)
        mu_s, logvar_s = self.encode_s(x)
        pred_y = self.Dec_y(self.phi_s(mu_s))
        return pred_y

    def get_x_y(self, z, s, env):
        z = self.phi_z[env](z)
        s = self.phi_s(s)
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(s)
        return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y

    def get_y(self, s):
        return self.Dec_y(self.phi_s(s))

    def get_y_by_zs(self, mu, logvar):
        zs = self.reparametrize(mu, logvar)
        s = self.phi_s(zs[:, self.zs_dim // 2:])
        return self.Dec_y(s)

    def encode_z(self, x, env_idx):
        return self.mean_z[env_idx](x), self.logvar_z[env_idx](x)

    def encode_s(self, x):
        return self.mean_s(x), self.logvar_s(x)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=64, kernel_size=4, stride=4, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=16,
                               kernel_size=4, stride=4, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        if self.softmax == 0:
            return nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim / 2), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes)
            )
        else:
            return nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim / 2), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes),
                nn.Softmax(dim=1),
            )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class sVAE_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 ):

        super(sVAE_2D, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us

        self.Enc_x = self.get_Enc_x()
        self.Enc_u = self.get_Enc_u()

        self.mean_zs = nn.Sequential(
            self.Fc_bn_ReLU(1536, 128),
            nn.Linear(128, self.zs_dim))
        self.sigma_zs = nn.Sequential(
            self.Fc_bn_ReLU(1536, 128),
            nn.Linear(128, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(512, 128),
            nn.Linear(128, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            self.Fc_bn_ReLU(512, 128),
            nn.Linear(128, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        if feature == 1:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        pred_y = self.Dec_y(mu)
        return pred_y

    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            concat = torch.cat([x, u], dim=1)
        else:
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        concat = u
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
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class sVAE_f_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 decoder_type=0,
                 ):

        super(sVAE_f_2D, self).__init__()
        print('model: sVAE_f_2D, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.relu = nn.ReLU()
        self.decoder_type = decoder_type

        self.Enc_x = self.get_Enc_x()

        if self.is_use_u:
            self.Enc_u = self.get_Enc_u()

            self.mean_zs = nn.Sequential(
                nn.Linear(1536, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(1536, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                nn.Linear(1024, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(1024, self.zs_dim))

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))

        if decoder_type == 0:
            self.Dec_x = self.get_Dec_x()
        else:
            self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        if self.decoder_type == 0:
            if feature == 1:
                return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
            else:
                return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror
        else:
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_piror, z
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        pred_y = self.Dec_y(mu)
        return pred_y

    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            concat = torch.cat([x, u], dim=1)
        else:
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        concat = u
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
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class sVAE_f_mnist_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 decoder_type=0,
                 ):

        super(sVAE_f_mnist_2D, self).__init__()
        print('model: sVAE_f_mnist_2D, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.relu = nn.ReLU()
        self.decoder_type = decoder_type
        self.in_plane = 256
        self.Enc_x = self.get_Enc_x_28()

        if self.is_use_u:
            self.Enc_u = self.get_Enc_u()

            self.mean_zs = nn.Sequential(
                # nn.Linear(1024, 1024),
                # nn.ReLU(),
                self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                nn.Linear(self.in_plane, self.zs_dim)
            )
            self.sigma_zs = nn.Sequential(
                # nn.Linear(1024, 1024),
                # nn.ReLU(),
                self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                nn.Linear(self.in_plane, self.zs_dim)
            )
        else:
            self.mean_zs = nn.Sequential(
                # nn.Linear(1024, 1024),
                # nn.ReLU(),
                self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                nn.Linear(self.in_plane, self.zs_dim)
            )
            self.sigma_zs = nn.Sequential(
                # nn.Linear(1024, 1024),
                # nn.ReLU(),
                self.Fc_bn_ReLU(self.in_plane, self.in_plane),
                nn.Linear(self.in_plane, self.zs_dim)
            )

        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))

        if decoder_type == 0:
            self.Dec_x = self.get_Dec_x()
        else:
            self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        if self.decoder_type == 0:
            if feature == 1:
                return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
            else:
                return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror
        else:
            if feature == 1:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_piror, z
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        pred_y = self.Dec_y(mu)
        return pred_y

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            concat = torch.cat([x, u], dim=1)
        else:
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        concat = u
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
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class sVAE_ff_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 zs_dim=256,
                 num_classes=1,
                 is_use_u=1,
                 is_use_us=0,
                 ):

        super(sVAE_ff_2D, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.zs_dim = zs_dim
        self.is_use_u = is_use_u
        self.is_use_us = is_use_us
        self.relu = nn.ReLU()

        self.Enc_x = self.get_Enc_x()

        if is_use_u == 1:
            self.Enc_u = self.get_Enc_u()

            self.mean_zs = nn.Sequential(
                nn.Linear(1536, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(1536, self.zs_dim))
        else:
            self.mean_zs = nn.Sequential(
                nn.Linear(1024, self.zs_dim))
            self.sigma_zs = nn.Sequential(
                nn.Linear(1024, self.zs_dim))
        # prior
        self.Enc_u_prior = self.get_Enc_u()
        self.mean_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))
        self.sigma_zs_prior = nn.Sequential(
            nn.Linear(512, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x, u, us, feature=0):
        mu, logvar = self.encode(x, u, us)
        mu_prior, logvar_piror = self.encode_prior(u, us)
        z = self.reparametrize(mu, logvar)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        if feature == 1:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror, z
        else:
            return pred_y, rec_x, mu, logvar, mu_prior, logvar_piror

    def get_pred_y(self, x, u, us):
        mu, logvar = self.encode(x, u, us)
        pred_y = self.Dec_y(mu)
        return pred_y

    def encode(self, x, u, us):
        if self.is_use_u == 1:
            x = self.Enc_x(x)
            u = self.Enc_u(u)
            concat = torch.cat([x, u], dim=1)
        else:
            concat = self.Enc_x(x)
        return self.mean_zs(concat), self.sigma_zs(concat)

    def encode_prior(self, u, us):
        u = self.Enc_u_prior(u)
        concat = u
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
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            nn.Linear(int(self.zs_dim), self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_u(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def get_Enc_us(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(self.us_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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


class MM_F_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 ):

        super(MM_F_2D, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )
        self.get_feature_u = nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512)
        )
        if self.is_use_u == 1:
            self.fc = nn.Linear(1536, num_classes)
        else:
            self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, u, us, feature=0):
        x = self.get_feature(x)
        if self.is_use_u == 1:
            u = self.get_feature_u(u)
            return self.fc(torch.cat([x, u], dim=1))
        else:
            return self.fc(x)

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class MM_F_f_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim=256,
                 ):

        super(MM_F_f_2D, self).__init__()
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.relu = nn.ReLU()

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )
        self.get_feature_u = nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256),
            self.Fc_bn_ReLU(256, 512)
        )
        if self.is_use_u == 1:
            self.fc_1 = nn.Linear(1536, self.zs_dim)
            self.fc = nn.Linear(self.zs_dim, num_classes)
        else:
            self.fc_1 = nn.Linear(1024, self.zs_dim)
            self.fc = nn.Linear(self.zs_dim, num_classes)

    def forward(self, x, u, us, feature=0):
        x = self.get_feature(x)
        if self.is_use_u == 1:
            u = self.get_feature_u(u)
            fea = self.relu(self.fc_1(torch.cat([x, u], dim=1)))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)
        else:
            fea = self.relu(self.fc_1(x))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class MM_F_ff_2D(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim=256,
                 ):

        super(MM_F_ff_2D, self).__init__()
        print('MM_F_ff_2D zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.relu = nn.ReLU()

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )

        if self.is_use_u == 1:
            self.get_feature_u = nn.Sequential(
                self.Fc_bn_ReLU(self.u_dim, 128),
                self.Fc_bn_ReLU(128, 256),
                self.Fc_bn_ReLU(256, 512)
            )

            self.fc_1 = nn.Linear(1536, self.zs_dim)
            self.fc = nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes), )
        else:
            self.fc_1 = nn.Linear(1024, self.zs_dim)
            self.fc = nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes),)

    def forward(self, x, u=None, us=None, feature=0):
        x = self.get_feature(x)
        if self.is_use_u == 1:
            u = self.get_feature_u(u)
            fea = self.relu(self.fc_1(torch.cat([x, u], dim=1)))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)
        else:
            fea = self.relu(self.fc_1(x))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class MM_F_ff_2D_NICO(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim=256,
                 ):

        super(MM_F_ff_2D_NICO, self).__init__()
        print('MM_F_ff_2D_NICO zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.relu = nn.ReLU()
        final_channel = 2048

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 128),
            self.Conv_bn_ReLU(128, 256, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )

        if self.is_use_u == 1:
            self.get_feature_u = nn.Sequential(
                self.Fc_bn_ReLU(self.u_dim, 256),
                self.Fc_bn_ReLU(256, 512),
                self.Fc_bn_ReLU(512, 512)
            )

            self.fc_1 = nn.Linear(1536, self.zs_dim)
            self.fc = nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim), 1024),
                self.Fc_bn_ReLU(1024, 2048),
                nn.Linear(2048, self.num_classes), )
        else:
            self.fc_1 = nn.Linear(1024, self.zs_dim)
            self.fc = nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim), 1024),
                self.Fc_bn_ReLU(1024, 2048),
                nn.Linear(2048, self.num_classes), )

    def forward(self, x, u=None, us=None, feature=0):
        x = self.get_feature(x)
        if self.is_use_u == 1:
            u = self.get_feature_u(u)
            fea = self.relu(self.fc_1(torch.cat([x, u], dim=1)))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)
        else:
            fea = self.relu(self.fc_1(x))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class MM_F_ff_2D_mnist(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim=256,
                 ):

        super(MM_F_ff_2D_mnist, self).__init__()
        print('MM_F_ff_2D_mnist zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.relu = nn.ReLU()

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )
        self.get_feature_u = nn.Sequential(
            self.Fc_bn_ReLU(self.u_dim, 128),
            self.Fc_bn_ReLU(128, 256)
        )
        if self.is_use_u == 1:
            self.fc_1 = nn.Linear(512, self.zs_dim)
            self.fc = nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes), )
        else:
            self.fc_1 = nn.Linear(256, self.zs_dim)
            self.fc = nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes), )

    def forward(self, x, u=None, us=None, feature=0):
        x = self.get_feature(x)
        if self.is_use_u == 1:
            u = self.get_feature_u(u)
            fea = self.relu(self.fc_1(torch.cat([x, u], dim=1)))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)
        else:
            fea = self.relu(self.fc_1(x))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class MM_F_ff_2D_mnist_L(nn.Module):
    def __init__(self,
                 in_channel=1,
                 u_dim=2,
                 us_dim=6,
                 num_classes=1,
                 is_use_u=1,
                 zs_dim=256,
                 ):

        super(MM_F_ff_2D_mnist_L, self).__init__()
        print('MM_F_ff_2D_mnist_L zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.is_use_u = is_use_u
        self.u_dim = u_dim
        self.us_dim = us_dim
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.relu = nn.ReLU()

        self.get_feature = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

        if self.is_use_u == 1:
            self.get_feature_u = nn.Sequential(
                self.Fc_bn_ReLU(self.u_dim, 128),
                self.Fc_bn_ReLU(128, 256)
            )

            self.fc_1 = nn.Linear(512, self.zs_dim)
            self.fc = nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes), )
        else:
            self.fc_1 = nn.Linear(256, self.zs_dim)
            self.fc = nn.Sequential(
                self.Fc_bn_ReLU(int(self.zs_dim), 512),
                self.Fc_bn_ReLU(512, 256),
                nn.Linear(256, self.num_classes), )

    def forward(self, x, u=None, us=None, feature=0):
        x = self.get_feature(x)
        if self.is_use_u == 1:
            u = self.get_feature_u(u)
            fea = self.relu(self.fc_1(torch.cat([x, u], dim=1)))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)
        else:
            fea = self.relu(self.fc_1(x))
            if feature:
                return self.fc(fea), fea
            else:
                return self.fc(fea)

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        return layer

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        return layer


class Generative_model_MMD(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 dp=0
                 ):

        super(Generative_model_MMD, self).__init__()
        print('Generative_model_MMD, zs_dim: ', zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.dp = dp
        self.dropout = nn.Dropout(p=self.dp)

        self.Enc_x = self.get_Enc_x()

        self.get_z = nn.Sequential(
            self.Fc_bn_ReLU(1024, 1024),
            nn.Linear(1024, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        return pred_y, rec_x, z

    def get_pred_y(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        return self.Dec_y(z)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(),
            nn.Upsample(6),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=64, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes)
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool3d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool3d(1),
            Flatten(),
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
            nn.Dropout(p=self.dp),
            nn.ReLU())
        return layer


class Generative_model_2D_mnist_normal_MMD(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 dp=0.1,
                 ):
        super(Generative_model_2D_mnist_normal_MMD, self).__init__()
        print('Generative_model_2D_mnist_normal_MMD, zs_dim: ', zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.dp = dp
        self.dropout = nn.Dropout(p=self.dp)

        self.Enc_x = self.get_Enc_x()

        self.get_z = nn.Sequential(
            self.Fc_bn_ReLU(1024, 1024),
            nn.Linear(1024, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), z

    def get_pred_y(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        return self.Dec_y(z)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            nn.ReLU(),
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes)
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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
            nn.Dropout(self.dp),
            nn.ReLU())
        return layer


class Generative_model_2D_mnist_normal_MMD_NICO(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 dp=0.1,
                 ):
        super(Generative_model_2D_mnist_normal_MMD_NICO, self).__init__()
        print('Generative_model_2D_mnist_normal_MMD_NICO, zs_dim: ', zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.dp = dp
        self.dropout = nn.Dropout(p=self.dp)

        self.Enc_x = nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 128),
            self.Conv_bn_ReLU(128, 256, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )

        self.get_z = nn.Sequential(
            self.Fc_bn_ReLU(1024, 1024),
            nn.Linear(1024, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        return pred_y, rec_x, z

    def get_pred_y(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        return self.Dec_y(z)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            self.Conv_bn_ReLU(64, 128, kernel_size=3, stride=2),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 128),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 256),
            self.Conv_bn_ReLU(256, 512),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(512, 512),
            self.Conv_bn_ReLU(512, 1024),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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
            nn.Dropout(self.dp),
            nn.ReLU())
        return layer


class Generative_model_2D_mnist_MMD(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 dp=0.1,
                 ):
        super(Generative_model_2D_mnist_MMD, self).__init__()
        print('Generative_model_2D_mnist_MMD, zs_dim: ', zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.dp = dp
        self.dropout = nn.Dropout(p=self.dp)

        self.Enc_x = self.get_Enc_x()

        self.get_z = nn.Sequential(
            self.Fc_bn_ReLU(512, 512),
            nn.Linear(512, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), z

    def get_pred_y(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        return self.Dec_y(z)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.zs_dim, 1024),
            nn.ReLU(),
            # self.Fc_bn_ReLU(int(self.zs_dim), 512),
            nn.Linear(1024, self.num_classes)
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 512),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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
            nn.Dropout(self.dp),
            nn.ReLU())
        return layer


class Generative_model_2D_mnist_MMD_L(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 dp=0.1,
                 ):
        super(Generative_model_2D_mnist_MMD_L, self).__init__()
        print('Generative_model_2D_mnist_MMD_L, zs_dim: ', zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.dp = dp
        self.dropout = nn.Dropout(p=self.dp)

        self.Enc_x = self.get_Enc_x_28()

        self.get_z = nn.Sequential(
            self.Fc_bn_ReLU(256, 256),
            nn.Linear(256, self.zs_dim))

        self.Dec_x = self.get_Dec_x_28()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), z

    def get_pred_y(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        return self.Dec_y(z)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_x_28(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(2),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=128, kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            self.Fc_bn_ReLU(int(self.zs_dim), 512),
            self.Fc_bn_ReLU(512, 256),
            nn.Linear(256, self.num_classes),
            nn.Softmax(dim=1),
        )

    def get_Enc_x_28(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 32),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(32, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 512),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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
            nn.Dropout(self.dp),
            nn.ReLU())
        return layer


class Generative_model_2D_MMD(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 dp=0.1,
                 ):
        super(Generative_model_2D_MMD, self).__init__()
        print('Generative_model_2D_MMD, zs_dim: ', zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.dp = dp
        self.dropout = nn.Dropout(p=self.dp)

        self.Enc_x = self.get_Enc_x()

        self.get_z = nn.Sequential(
            self.Fc_bn_ReLU(512, 512),
            nn.Linear(512, self.zs_dim))

        self.Dec_x = self.get_Dec_x()
        self.Dec_y = self.get_Dec_y()

    def forward(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        rec_x = self.Dec_x(z)
        pred_y = self.Dec_y(z)
        return pred_y, rec_x, z

    def get_pred_y(self, x):
        x = self.Enc_x(x)
        z = self.get_z(x)
        return self.Dec_y(z)

    def decode_x(self, zs):
        return self.Dec_x(zs)

    def decode_y(self, s):
        return self.Dec_y(s)

    def get_Dec_x(self):
        return nn.Sequential(
            UnFlatten(type='2d'),
            nn.Upsample(16),
            self.TConv_bn_ReLU(
                in_channels=self.zs_dim, out_channels=256, kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=256, out_channels=256,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=256, out_channels=128,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(in_channels=32, out_channels=32,
                              kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def get_Dec_y(self):
        return nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.zs_dim, 1024),
            nn.ReLU(),
            # self.Fc_bn_ReLU(int(self.zs_dim), 512),
            nn.Linear(1024, self.num_classes)
        )

    def get_Enc_x(self):
        return nn.Sequential(
            self.Conv_bn_ReLU(self.in_channel, 64),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(64, 128),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(128, 256),
            nn.MaxPool2d(2),
            self.Conv_bn_ReLU(256, 512),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )

    def Conv_bn_ReLU(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                     bias=True, groups=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=bias, groups=groups),
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
            nn.Dropout(self.dp),
            nn.ReLU())
        return layer


class D_model(nn.Module):
    def __init__(self,
                 zs_dim,
                 hidden_dim,
                 dp=0.1):
        super(D_model, self).__init__()
        self.zs_dim = zs_dim
        self.dp = dp
        self.D = nn.Sequential(
            self.Fc_bn_ReLU(self.zs_dim, hidden_dim),
            # self.Fc_bn_ReLU(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.D(x)

    def Fc_bn_ReLU(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(self.dp),
            nn.ReLU())
        return layer
