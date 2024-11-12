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

# sijie NIPS zhuankan mnist


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
        self.s_dim = int(self.zs_dim - self.z_dim)
        self.is_reparameterize = args.is_reparameterize

        if not self.is_reparameterize:  # prior version
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
            pass
        else:  # reparameterize version
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

        # self.mean_z = []
        # self.logvar_z = []
        # self.mean_s = []
        # self.logvar_s = []
        # self.shared_s = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
        # for env_idx in range(self.total_env):
        #     self.mean_z.append(
        #         nn.Sequential(
        #             self.Fc_bn_ReLU(self.in_plane, self.in_plane),
        #             nn.Linear(self.in_plane, self.z_dim)
        #         )
        #     )
        #     self.logvar_z.append(
        #         nn.Sequential(
        #             self.Fc_bn_ReLU(self.in_plane, self.in_plane),
        #             nn.Linear(self.in_plane, self.z_dim)
        #         )
        #     )
        #     self.mean_s.append(
        #         nn.Sequential(
        #             self.shared_s,
        #             nn.Linear(self.in_plane, self.s_dim)
        #         )
        #     )
        #     self.logvar_s.append(
        #         nn.Sequential(
        #             self.shared_s,
        #             nn.Linear(self.in_plane, self.s_dim)
        #         )
        #     )

        # self.mean_z = nn.ModuleList(self.mean_z)
        # self.logvar_z = nn.ModuleList(self.logvar_z)
        # self.mean_s = nn.ModuleList(self.mean_s)
        # self.logvar_s = nn.ModuleList(self.logvar_s)

        # # prior
        # self.Enc_u_prior = self.get_Enc_u()
        # self.mean_zs_prior = nn.Sequential(
        #     nn.Linear(32, self.zs_dim))
        # self.logvar_zs_prior = nn.Sequential(
        #     nn.Linear(32, self.zs_dim))

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
                        if env_idx == 0:
                            mu_0 = mu.clone()
                        else:
                            mu_1 = mu.clone()
                        # pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        zs = self.reparametrize(mu, logvar)
                        if not self.is_reparameterize:  # prior version
                            z = zs[:, :self.z_dim]
                            s = zs[:, self.z_dim:]
                            pass
                        else:  # reparameterize version
                            z = self.phi_z[env_idx](zs[:, :self.zs_dim // 2])
                            s = self.phi_s[env_idx](zs[:, self.zs_dim // 2:])
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:  # sijiez_init, s_init = None, None
                            z_init, s_init = z, s
                            if self.args.mse_loss:  # sijie mse_loss=1
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x).view(-1,
                                                                       3 * self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x).view(-1,
                                                                                   3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:  # sijie mse_loss=1
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x).view(-1,
                                                                   3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x).view(-1,
                                                                               3 * self.args.image_size ** 2),
                                                                  reduction='none').mean(1)
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]

            z, s = z_init, s_init
            z_init_save, s_init_save = z_init.clone(), s_init.clone()
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
                                     (raw_x).view(-1,
                                                  3 * self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x).view(-1,
                                                              3 * self.args.image_size ** 2),
                                                 reduction='none')
                loss = BCE.mean(1)  # + args.gamma2 * cls_loss
                loss = loss.mean()
                if i % 50 == 0:
                    print("test epoch: {}, rec loss: {}".format(i, loss))
                loss.backward()
                optimizer.step()
                # print(i, s)
            pred_y = self.get_y(s)
            if is_debug:
                # return pred_y_init, pred_y
                return pred_y_init, pred_y, z_init_save, s_init_save, z, s, mu_0, mu_1
            else:
                return pred_y
        elif is_train == 2:
            mu, logvar = self.encode(x, 0)
            zs = self.reparametrize(mu, logvar)
            s = zs[:, self.z_dim:]
            return self.Dec_y(s)
        else:  # sijie train is_train=1
            mu, logvar = self.encode(x, env)
            mu_prior, logvar_prior = 0, 0
            zs = self.reparametrize(mu, logvar)
            if not self.is_reparameterize:
                mu_prior, logvar_prior = self.encode_prior(x, env)
                z = zs[:, :self.z_dim]
                s = zs[:, self.z_dim:]
            else:
                z = self.phi_z[env](zs[:, :self.zs_dim // 2])
                s = self.phi_s[env](zs[:, self.zs_dim // 2:])
            # z = zs[:, :self.z_dim]
            # s = zs[:, self.z_dim:]
            zs = torch.cat([z, s], dim=1)
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(s)
            if feature == 1:  # sijie feature=1
                # print("rec_x", rec_x.size())
                # print("rec_x[:, :, 2:30, 2:30]",
                #       rec_x[:, :, 2:30, 2:30].size())
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s, zs
            else:
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        if not self.is_reparameterize:  # prior version
            # mu_prior, logvar_prior = self.encode_prior(x, env)
            z = zs[:, :self.z_dim]
            s = zs[:, self.z_dim:]
        else:  # reparameterize version
            z = self.phi_z[env](zs[:, :self.zs_dim // 2])  # 正态逆变换
            s = self.phi_s[env](zs[:, self.zs_dim // 2:])  # 正态逆变换
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
        if not self.is_reparameterize:  # prior version
            # mu_prior, logvar_prior = self.encode_prior(x, env)
            # z = zs[:, :self.z_dim]
            s = zs[:, self.z_dim:]
        else:  # reparameterize version
            # z = self.phi_z[env](zs[:, :self.zs_dim // 2])  # 正态逆变换
            s = self.phi_s[env](zs[:, self.zs_dim // 2:])  # 正态逆变换
        # s = zs[:, self.z_dim:]
        return self.Dec_y(s)

    def encode(self, x, env_idx):
        if not self.is_reparameterize:  # prior version
            return torch.cat([self.mean_z[env_idx](x), self.mean_s[env_idx](x)], dim=1), \
                torch.cat([self.logvar_z[env_idx](x),
                           self.logvar_s[env_idx](x)], dim=1)
        else:  # reparameterize
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
            self.Conv_bn_ReLU(128, 128),
            self.TConv_bn_ReLU(in_channels=128, out_channels=64,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(64, 64),
            self.TConv_bn_ReLU(in_channels=64, out_channels=32,
                               kernel_size=2, stride=2, padding=0),
            self.Conv_bn_ReLU(32, 32),
            self.TConv_bn_ReLU(in_channels=32, out_channels=16,
                               kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=self.in_channel,
                      kernel_size=3, stride=1, padding=1),
            nn.Tanh()
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


# sijie nips 2021 zhuankan NICO


class Generative_model_f_2D_unpooled_env_t_mnist_prior_share_NICO(nn.Module):
    def __init__(self,
                 in_channel=1,
                 zs_dim=256,
                 num_classes=1,
                 decoder_type=0,
                 total_env=2,
                 args=None,
                 is_cuda=1
                 ):

        super(Generative_model_f_2D_unpooled_env_t_mnist_prior_share_NICO,
              self).__init__()
        print('model: Generative_model_f_2D_unpooled_env_t_mnist_prior_share_NICO, zs_dim: %d' % zs_dim)
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.zs_dim = zs_dim
        self.decoder_type = decoder_type
        self.total_env = total_env
        self.args = args
        self.is_cuda = is_cuda
        self.in_plane = 256
        self.z_dim = int(round(zs_dim * args.z_ratio))
        self.Enc_x = self.get_Enc_x()
        self.u_dim = total_env
        print('z_dim is ', self.z_dim)
        self.s_dim = int(self.zs_dim - self.z_dim)

        self.is_reparameterize = args.is_reparameterize

        if not self.is_reparameterize:  # prior version
            self.mean_z = []
            self.logvar_z = []
            self.mean_s = []
            self.logvar_s = []
            self.shared_s = self.Fc_bn_ReLU(1024, self.in_plane)
            for env_idx in range(self.total_env):
                self.mean_z.append(
                    nn.Sequential(
                        self.Fc_bn_ReLU(1024, self.in_plane),
                        nn.Linear(self.in_plane, self.z_dim)
                    )
                )
                self.logvar_z.append(
                    nn.Sequential(
                        self.Fc_bn_ReLU(1024, self.in_plane),
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
            pass
        else:  # reparameterize version
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

        # self.mean_z = []
        # self.logvar_z = []
        # self.mean_s = []
        # self.logvar_s = []
        # self.shared_s = self.Fc_bn_ReLU(self.in_plane, self.in_plane)
        # for env_idx in range(self.total_env):
        #     self.mean_z.append(
        #         nn.Sequential(
        #             self.Fc_bn_ReLU(self.in_plane, self.in_plane),
        #             nn.Linear(self.in_plane, self.z_dim)
        #         )
        #     )
        #     self.logvar_z.append(
        #         nn.Sequential(
        #             self.Fc_bn_ReLU(self.in_plane, self.in_plane),
        #             nn.Linear(self.in_plane, self.z_dim)
        #         )
        #     )
        #     self.mean_s.append(
        #         nn.Sequential(
        #             self.shared_s,
        #             nn.Linear(self.in_plane, self.s_dim)
        #         )
        #     )
        #     self.logvar_s.append(
        #         nn.Sequential(
        #             self.shared_s,
        #             nn.Linear(self.in_plane, self.s_dim)
        #         )
        #     )

        # self.mean_z = nn.ModuleList(self.mean_z)
        # self.logvar_z = nn.ModuleList(self.logvar_z)
        # self.mean_s = nn.ModuleList(self.mean_s)
        # self.logvar_s = nn.ModuleList(self.logvar_s)

        # # prior
        # self.Enc_u_prior = self.get_Enc_u()
        # self.mean_zs_prior = nn.Sequential(
        #     nn.Linear(32, self.zs_dim))
        # self.logvar_zs_prior = nn.Sequential(
        #     nn.Linear(32, self.zs_dim))

        self.Dec_x = self.get_Dec_x_256()
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
                        if not self.is_reparameterize:  # reparameterize version
                            z = zs[:, :self.z_dim]
                            s = zs[:, self.z_dim:]
                            pass
                        else:  # prior version
                            z = self.phi_z[env_idx](zs[:, :self.zs_dim // 2])
                            s = self.phi_s[env_idx](zs[:, self.zs_dim // 2:])
                        # z = zs[:, :self.z_dim]
                        # s = zs[:, self.z_dim:]
                        zs = torch.cat([z, s], dim=1)
                        recon_x = self.Dec_x(zs)
                        # recon_x = recon_x[:, :, 2:30, 2:30].contiguous()

                        if z_init is None:  # sijiez_init, s_init = None, None
                            z_init, s_init = z, s
                            if self.args.mse_loss:  # sijie mse_loss=1
                                min_rec_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                          (raw_x).view(-1,
                                                                       3 * self.args.image_size ** 2),
                                                          reduction='none').mean(1)
                            else:
                                min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                      (raw_x).view(-1,
                                                                                   3 * self.args.image_size ** 2),
                                                                      reduction='none').mean(1)
                        else:
                            if self.args.mse_loss:
                                new_loss = F.mse_loss(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                      (raw_x).view(-1,
                                                                   3 * self.args.image_size ** 2),
                                                      reduction='none').mean(1)
                            else:
                                new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                                  (raw_x).view(-1,
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
                                     (raw_x).view(-1,
                                                  3 * self.args.image_size ** 2),
                                     reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * self.args.image_size ** 2),
                                                 (raw_x).view(-1,
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
            mu_prior, logvar_prior = 0, 0
            zs = self.reparametrize(mu, logvar)
            if not self.is_reparameterize:  # prior version
                mu_prior, logvar_prior = self.encode_prior(x, env)
                z = zs[:, :self.z_dim]
                s = zs[:, self.z_dim:]
            else:  # parameterize
                z = self.phi_z[env](zs[:, :self.zs_dim // 2])
                s = self.phi_s[env](zs[:, self.zs_dim // 2:])
            zs = torch.cat([z, s], dim=1)
            rec_x = self.Dec_x(zs)
            pred_y = self.Dec_y(s)
            if feature == 1:  # sijie feature=1
                return pred_y, rec_x, mu, logvar, mu_prior, logvar_prior, z, s, zs
            else:  # ci fen zhi buyong
                return pred_y, rec_x[:, :, 2:30, 2:30].contiguous(), mu, logvar, mu_prior, logvar_prior, z, s

    def get_pred_y(self, x, env):
        x = self.Enc_x(x)
        mu, logvar = self.encode(x, env)
        zs = self.reparametrize(mu, logvar)
        if not self.is_reparameterize:  # prior version
            # mu_prior, logvar_prior = self.encode_prior(x, env)
            z = zs[:, :self.z_dim]
            s = zs[:, self.z_dim:]
        else:  # reparameterize version
            z = self.phi_z[env](zs[:, :self.zs_dim // 2])  # 正态逆变换
            s = self.phi_s[env](zs[:, self.zs_dim // 2:])  # 正态逆变换
        zs = torch.cat([z, s], dim=1)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        return pred_y

    def get_x_y(self, z, s):
        zs = torch.cat([z, s], dim=1)
        rec_x = self.Dec_x(zs)
        pred_y = self.Dec_y(zs[:, self.z_dim:])
        # return rec_x[:, :, 2:30, 2:30].contiguous(), pred_y
        return rec_x, pred_y

    def get_y(self, s):
        return self.Dec_y(s)

    def get_y_by_zs(self, mu, logvar, env):
        zs = self.reparametrize(mu, logvar)
        if not self.is_reparameterize:  # prior version
            # mu_prior, logvar_prior = self.encode_prior(x, env)
            # z = zs[:, :self.z_dim]
            s = zs[:, self.z_dim:]
        else:  # reparameterize version
            # z = self.phi_z[env](zs[:, :self.zs_dim // 2])  # 正态逆变换
            s = self.phi_s[env](zs[:, self.zs_dim // 2:])  # 正态逆变换
        # s = zs[:, self.z_dim:]
        return self.Dec_y(s)

    def encode(self, x, env_idx):
        if not self.is_reparameterize:  # prior version
            # print(x.size())
            # print(self.mean_z[env_idx](x).size(),
            #       self.mean_s[env_idx](x).size())
            # print(self.logvar_z[env_idx](x).size(),
            #       self.logvar_s[env_idx](x).size())
            return torch.cat([self.mean_z[env_idx](x), self.mean_s[env_idx](x)], dim=1), \
                torch.cat([self.logvar_z[env_idx](x),
                           self.logvar_s[env_idx](x)], dim=1)
        else:  # reparameterize
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

    def get_Dec_x_256(self):
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
            nn.ReLU())
        return layer
