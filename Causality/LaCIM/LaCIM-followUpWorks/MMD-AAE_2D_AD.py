# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils_baseline import *
from torchvision import transforms
from models_baseline import *
import torch.utils.data as data
import xlrd
import torch.nn.functional as F
import sys


def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1. / (n_1 * (n_1 - 1))
        self.a11 = 1. / (n_2 * (n_2 - 1))
        self.a01 = - 1. / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        # print(k_1.size(), k_2.size(), k_12.size())
        # print(k_1, k_2, k_12)
        # print(torch.trace(k_1))
        # print(torch.trace(k_2))
        return k_1.mean() + k_2.mean() - 2 * k_12.mean()

        # mmd = (2 * self.a01 * k_12.sum() +
        #        self.a00 * (k_1.sum() - torch.trace(k_1)) +
        #        self.a11 * (k_2.sum() - torch.trace(k_2)))
        # if ret_matrix:
        #     return mmd, kernels
        # else:
        #     return mmd

    # def pval(self, distances, n_permutations=1000):
    #     r"""Compute a p-value using a permutation test.
    #     Arguments
    #     ---------
    #     matrix: :class:`torch:torch.autograd.Variable`
    #         The matrix computed using :py:meth:`~.MMDStatistic.__call__`.
    #     n_permutations: int
    #         The number of random draws from the permutation null.
    #     Returns
    #     -------
    #     float
    #         The estimated p-value."""
    #     if isinstance(distances, Variable):
    #         distances = distances.data
    #     return permutation_test_mat(distances.cpu().numpy(),
    #                                 self.n_1, self.n_2,
    #                                 n_permutations,
    #                                 a00=self.a00, a11=self.a11, a01=self.a01)


def train(epoch,
          model,
          D_net,
          optimizer,
          optimizer_D,
          dataloader,
          args):
    all_zs = np.zeros((len(dataloader.dataset), args.zs_dim))
    RECON_loss = AverageMeter()
    KLD_loss = AverageMeter()
    classify_loss = AverageMeter()
    MMD_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    EPS = 1e-15
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda(), env.cuda(), u.cuda()

        model.zero_grad()
        D_net.zero_grad()
        model.eval()
        D_net.train()
        if args.gamma > 0:
            z_real_gauss = torch.randn(x.size(0), args.zs_dim).cuda()
            D_real_gauss = D_net(z_real_gauss)

            z_fake_gauss = model.get_z(model.Enc_x(x))
            D_fake_gauss = D_net(z_fake_gauss)

            kld_loss = torch.mean(
                torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))

            loss = -1 * args.gamma * kld_loss
            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

        # the second step
        model.zero_grad()
        D_net.zero_grad()
        model.train()
        D_net.eval()
        pred_y, rec_x, z = model(x)
        if args.dataset == 'AD':
            recon_loss = F.binary_cross_entropy(rec_x.view(-1, 1 * args.image_size ** 3) + EPS,
                                                (x * 0.5 + 0.5).view(-1, 1 * args.image_size ** 3) + EPS)
        else:
            recon_loss = F.binary_cross_entropy(rec_x.view(-1, 3 * args.image_size ** 2) + EPS,
                                                (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2) + EPS)

        z_real_gauss = torch.randn(x.size(0), args.zs_dim).cuda()
        D_real_gauss = D_net(z_real_gauss)

        z_fake_gauss = model.get_z(model.Enc_x(x))
        D_fake_gauss = D_net(z_fake_gauss)

        kld_loss = torch.mean(torch.log(D_real_gauss + EPS) +
                              torch.log(1 - D_fake_gauss + EPS))
        # print('pred_y', pred_y, 'target', target )
        cls_loss = F.cross_entropy(pred_y, target)

        mmd_loss = torch.FloatTensor([0.0]).cuda()
        valid_counter = 0
        for s_i in range(args.env_num):
            for s_j in range(args.env_num):
                if torch.sum(env == s_i) <= 0 or torch.sum(env == s_j) <= 0:
                    continue
                mmd_stat = MMDStatistic(
                    torch.sum(env == s_i), torch.sum(env == s_j))
                mmd_loss = torch.add(mmd_loss, mmd_stat(
                    z[env == s_i, :], z[env == s_j, :], [1, 5, 10]))
                valid_counter = valid_counter + 1
        mmd_loss = mmd_loss / valid_counter

        loss = cls_loss + args.alpha * recon_loss + \
            args.beta * mmd_loss + args.gamma * kld_loss
        # print('cls_loss', cls_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('kld loss after model optimizer, ', kld_loss.item())

        RECON_loss.update(recon_loss.item(), x.size(0))
        KLD_loss.update(kld_loss.item(), x.size(0))
        classify_loss.update(cls_loss.item(), x.size(0))
        MMD_loss.update(mmd_loss.item(), x.size(0))
        all_loss.update(loss.item(), x.size(0))

        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(),
                        target.detach().cpu().numpy()), x.size(0))

        if batch_idx % 10 == 0:
            print(
                'epoch [{}/{}], batch: {}, rec_loss:{:.4f}, kld_loss:{:.4f} cls_loss:{:.4f}, mmd_loss:{:.4f}, overall_loss:{:.4f},acc:{:.4f}'
                .format(epoch,
                        args.epochs,
                        batch_idx,
                        RECON_loss.avg * args.alpha,
                        KLD_loss.avg * args.gamma,
                        classify_loss.avg,
                        MMD_loss.avg * args.beta,
                        all_loss.avg,
                        accuracy.avg * 100))

    if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' or \
            args.model == 'VAE_f' or args.model == 'sVAE_f':
        all_zs = all_zs[:batch_begin]
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch,
          args.epochs, all_loss.avg))

    return all_zs, accuracy.avg


def evaluate(model,
             dataloader,
             args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    # print('in test code', dataloader.dataset.__len__())
    # print(dataloader)
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        x, target, env, u = x.cuda(), target.cuda(), env.cuda(), u.cuda()

        pred_y = model.get_pred_y(x)
        # print(x, pred_y)
        pred[batch_begin:batch_begin + x.size(0), :] = pred_y.detach().cpu()
        batch_begin = batch_begin + x.size(0)
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(),
                        target.detach().cpu().numpy()), x.size(0))

    return pred, accuracy.avg


def evaluate_only(model,
                  dataloader,
                  args):
    model.eval()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    label = np.zeros((dataloader.dataset.__len__(), ))
    batch_begin = 0
    # print('in test code', dataloader.dataset.__len__())
    # print(dataloader)
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        x, target, env, u = x.cuda(), target.cuda(), env.cuda(), u.cuda()

        pred_y = model.get_pred_y(x)
        # print(x, pred_y)
        pred[batch_begin:batch_begin +
             x.size(0), :] = pred_y.detach().cpu().numpy()
        label[batch_begin:batch_begin +
              x.size(0), ] = target.detach().cpu().numpy()
        batch_begin = batch_begin + x.size(0)
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(),
                        target.detach().cpu().numpy()), x.size(0))

    return pred, label, accuracy.avg


# sijie nips zhuankan on use ADNI


class get_dataset(data.Dataset):
    def __init__(self,
                 root=None,
                 fold='train',
                 args=None,
                 aug=1):
        self.root = root
        self.args = args
        self.image_path_list = []
        self.u = []
        self.us = []
        self.train_u = []
        self.train_us = []
        self.y = []
        self.env = []
        self.aug = aug
        fold_map = {'train': 0, 'val': 1, 'test': 2}
        Label_map = {'AD': 2, 'MCI': 1, 'NC': 0}
        if self.args.if_unsp_cluster:  # sijie to use unsupervied cluster datset
            # if self.args.env_num == 2:
            print("load unsupervised cluster ADNI dataset")
            workbook = xlrd.open_workbook(
                r"../Dataset_E/E%s_%s_AD_%s.xls" % (self.args.env_num, fold, self.args.dataset_type[-3:]))
            sheet = workbook.sheet_by_index(0)
            for rows in range(1, sheet.nrows):
                # if sheet.row_values(rows)[4] == fold_map[fold]:
                self.image_path_list.append(sheet.row_values(rows)[0])
                self.y.append(sheet.row_values(rows)[1])
                self.env.append(sheet.row_values(rows)[self.args.env_num])
                self.u.append(float(sheet.row_values(rows)[1]))
                self.us.append(float(sheet.row_values(rows)[1]))
                self.train_u.append(sheet.row_values(rows)[1:3])
                self.train_us.append(float(sheet.row_values(rows)[1]))
            pass
            # elif self.args.env_num == 3:
            # pass
        # se:

        elif args.dataset == 'AD':  # Nips2021 dataset
            # if args.dataset == 'AD':
            if self.root is None:
                self.root = '/home/botong/'

            if args.dataset_type == 'gene_1':
                workbook = xlrd.open_workbook(os.path.join(
                    self.root, "Dataset/ADNI_dataset_APOE4.xlsx"))
                sheet = workbook.sheet_by_index(0)
                for rows in range(1, sheet.nrows):
                    if sheet.row_values(rows)[6] == fold_map['train']:
                        self.train_u.append(sheet.row_values(rows)[8:12])
                        self.train_us.append(float(sheet.row_values(rows)[12]))

                    if sheet.row_values(rows)[6] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(
                            self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        self.us.append(float(sheet.row_values(rows)[12]))

            elif 'gene_5' in args.dataset_type:
                if args.fold == 1:  # sijie use this dataset
                    workbook = xlrd.open_workbook(os.path.join(
                        self.root, "Dataset/ADNI_dataset_APOE4.xlsx"))
                    print('load data excel 1')
                elif args.fold == 2:
                    workbook = xlrd.open_workbook(os.path.join(
                        self.root, "Dataset/ADNI_dataset_APOE4_2.xlsx"))
                    print('load data excel 2')
                elif args.fold == 3:
                    workbook = xlrd.open_workbook(os.path.join(
                        self.root, "Dataset/ADNI_dataset_APOE4_3.xlsx"))
                    print('load data excel 3')
                if 'SEX' in args.dataset_type:
                    sheet = workbook.sheet_by_index(10)
                elif 'AV45' in args.dataset_type:
                    sheet = workbook.sheet_by_index(12)
                elif 'ABE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(13)
                elif 'TAU' in args.dataset_type:
                    sheet = workbook.sheet_by_index(14)  # sijie zhuankan
                elif 'APOE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(15)
                    print('workbook.sheet_by_index(15)')
                elif 'EDU' in args.dataset_type:
                    sheet = workbook.sheet_by_index(8)
                elif 'AGE2' in args.dataset_type:
                    sheet = workbook.sheet_by_index(11)
                elif 'AGE' in args.dataset_type:
                    sheet = workbook.sheet_by_index(6)  # sijie zhuankan
                elif 're' in args.dataset_type:
                    sheet = workbook.sheet_by_index(4)
                else:
                    sheet = workbook.sheet_by_index(2)
                for rows in range(1, sheet.nrows):
                    # if sheet.row_values(rows)[6] == fold_map['train']:
                    #     self.train_u.append(sheet.row_values(rows)[8:12])
                    #     if 'm' in args.dataset_type:
                    #         self.train_us.append(sheet.row_values(rows)[12:21])
                    #     else:
                    #         self.train_us.append(sheet.row_values(rows)[12:17])
                    # and int(sheet.row_values(rows)[21]) == select_env:
                    if sheet.row_values(rows)[6] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(
                            self.root, sheet.row_values(rows)[7]))
                        self.y.append(Label_map[sheet.row_values(rows)[2]])
                        self.u.append(sheet.row_values(rows)[8:12])
                        if 'm' in args.dataset_type:
                            self.us.append(sheet.row_values(rows)[12:21])
                        else:
                            self.us.append(sheet.row_values(rows)[12:17])
                        self.env.append(int(sheet.row_values(rows)[21]))

        self.u = np.array(self.u).astype('float32')
        self.us = np.array(self.us).astype('float32')
        self.train_u = np.array(self.train_u).astype('float32')
        self.train_us = np.array(self.train_us).astype('float32')
        if args.data_process == 'fill':
            for ss in range(self.u.shape[1]):
                self.u[self.u[:, ss] < 0,
                       ss] = self.train_u[self.train_u[:, ss] >= 0, ss].mean()
            for ss in range(self.us.shape[1]):
                self.us[self.us[:, ss] < 0,
                        ss] = self.train_us[self.train_us[:, ss] >= 0, ss].mean()
        elif args.data_process == 'fill_std':
            for ss in range(self.u.shape[1]):
                self.u[self.u[:, ss] < 0,
                       ss] = self.train_u[self.train_u[:, ss] >= 0, ss].mean()
                self.train_u[self.train_u[:, ss] < 0,
                             ss] = self.train_u[self.train_u[:, ss] >= 0, ss].mean()
                self.u[:, ss] = (self.u[:, ss] - self.train_u[:,
                                 ss].mean()) / self.train_u[:, ss].std()
            for ss in range(self.us.shape[1]):
                self.us[self.us[:, ss] < 0,
                        ss] = self.train_us[self.train_us[:, ss] >= 0, ss].mean()
                self.train_us[self.train_us[:, ss] < 0,
                              ss] = self.train_us[self.train_us[:, ss] >= 0, ss].mean()
                self.us[:, ss] = (
                    self.us[:, ss] - self.train_us[:, ss].mean()) / self.train_us[:, ss].std()

    def __getitem__(self, index):
        x = load_img_path(self.image_path_list[index]) / 255.0
        if self.aug == 1:
            x = self._img_aug(x).reshape(self.args.in_channel,
                                         self.args.crop_size,
                                         self.args.crop_size,
                                         self.args.crop_size, )
        else:
            crop_x = int(x.shape[0] / 2 - self.args.crop_size / 2)
            crop_y = int(x.shape[1] / 2 - self.args.crop_size / 2)
            crop_z = int(x.shape[2] / 2 - self.args.crop_size / 2)
            x = x[crop_x: crop_x + self.args.crop_size, crop_y: crop_y + self.args.crop_size,
                  crop_z: crop_z + self.args.crop_size]
            x = x.reshape(self.args.in_channel,
                          self.args.crop_size,
                          self.args.crop_size,
                          self.args.crop_size, )
        return torch.from_numpy(x.astype('float32')), \
            torch.from_numpy(np.array(self.y[index]).astype('int')), \
            torch.from_numpy(np.array(self.env[index]).astype('int')),\
            torch.from_numpy(np.array(self.u[index]).astype('float32')),
        # torch.from_numpy(np.array(self.us[index]).astype('float32')), \

    def __len__(self):
        return len(self.image_path_list)

    def _img_aug(self, x):
        if self.args.shift > 0:
            shift_x = (np.random.choice(2) * 2 - 1) * \
                np.random.choice(int(round(self.args.shift)))
            shift_y = (np.random.choice(2) * 2 - 1) * \
                np.random.choice(int(round(self.args.shift)))
            shift_z = (np.random.choice(2) * 2 - 1) * \
                np.random.choice(int(round(self.args.shift)))
        else:
            shift_x, shift_y, shift_z = 0, 0, 0

        crop_x = int(x.shape[0] / 2 - self.args.crop_size / 2 + shift_x)
        crop_y = int(x.shape[1] / 2 - self.args.crop_size / 2 + shift_y)
        crop_z = int(x.shape[2] / 2 - self.args.crop_size / 2 + shift_z)

        x_aug = x[crop_x: crop_x + self.args.crop_size, crop_y: crop_y + self.args.crop_size,
                  crop_z: crop_z + self.args.crop_size]

        transpose_type = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]

        if self.args.transpose:
            trans_idx = np.random.choice(3)
            if trans_idx != 0:
                x_aug = x_aug.transpose(transpose_type[trans_idx])

        if self.args.flip:
            flip_x = np.random.choice(2) * 2 - 1
            flip_y = np.random.choice(2) * 2 - 1
            flip_z = np.random.choice(2) * 2 - 1
            x_aug = x_aug[::flip_x, ::flip_y, ::flip_z]

        return x_aug

# sijie nips zhuankan on use CMNIST & NICO


class get_dataset_2D_env(data.Dataset):
    def __init__(self,
                 root=None,
                 fold='train',
                 args=None,
                 transform=None):
        self.root = root
        self.args = args
        self.image_path_list = []
        self.u = []
        self.us = []
        self.train_u = []
        self.train_us = []
        self.y = []
        self.env = []
        self.transform = transform
        fold_map = {'train': 0, 'val': 1, 'test': 2}
        if args.dataset == 'NICO':
            # todo xiugai unsp cluster results dataset(ADNI CMNIST)
            if self.args.if_unsp_cluster:  # sijie to use unsupervied cluster datset
                # if self.args.env_num == 2:
                workbook = xlrd.open_workbook(
                    r"../Dataset_E/E%s_%s_NICO.xls" % (self.args.env_num, fold))
                sheet = workbook.sheet_by_index(0)
                for rows in range(1, sheet.nrows):
                    # if sheet.row_values(rows)[4] == fold_map[fold]:
                    self.image_path_list.append(sheet.row_values(rows)[0])
                    self.y.append(sheet.row_values(rows)[1])
                    self.env.append(sheet.row_values(rows)[2])
                    self.u.append(sheet.row_values(rows)[2])
                pass

            else:  # Nips2021 dataset
                if self.root is None:
                    self.root = '/home/botong/Dataset/'

                workbook = xlrd.open_workbook(
                    r"%sNICO_dataset.xls" % self.root)
                if args.dataset_type == 'NICO_0':
                    sheet = workbook.sheet_by_index(0)
                elif args.dataset_type == 'NICO_1':
                    sheet = workbook.sheet_by_index(1)
                elif args.dataset_type == 'NICO_2':
                    sheet = workbook.sheet_by_index(2)
                elif args.dataset_type == 'NICO_3':
                    sheet = workbook.sheet_by_index(3)
                elif args.dataset_type == 'NICO_4':
                    sheet = workbook.sheet_by_index(4)
                elif args.dataset_type == 'NICO_5':
                    sheet = workbook.sheet_by_index(5)
                elif args.dataset_type == 'NICO_6':
                    sheet = workbook.sheet_by_index(6)
                elif args.dataset_type == 'NICO_7':
                    sheet = workbook.sheet_by_index(7)
                elif args.dataset_type == 'NICO_8':
                    sheet = workbook.sheet_by_index(8)
                elif args.dataset_type == 'NICO_9':
                    sheet = workbook.sheet_by_index(9)
                elif args.dataset_type == 'NICO_10':
                    sheet = workbook.sheet_by_index(10)
                elif args.dataset_type == 'NICO_11':
                    sheet = workbook.sheet_by_index(11)
                elif args.dataset_type == 'NICO_12':
                    sheet = workbook.sheet_by_index(12)
                else:
                    sheet = workbook.sheet_by_index(13)
                for rows in range(1, sheet.nrows):
                    if sheet.row_values(rows)[4] == fold_map[fold]:
                        self.image_path_list.append(os.path.join(
                            self.root, sheet.row_values(rows)[0]))
                        self.y.append(sheet.row_values(rows)[1])
                        self.env.append(sheet.row_values(rows)[3])
                        self.u.append(sheet.row_values(rows)[3])

        elif args.dataset == 'mnist_2':
            if self.args.if_unsp_cluster:  # sijie to use unsupervied cluster datset
                # if self.args.env_num == 2:
                workbook = xlrd.open_workbook(
                    r"../Dataset_E/E%s_%s_CMNIST.xls" % (self.args.env_num, fold))
                sheet = workbook.sheet_by_index(0)
                for rows in range(1, sheet.nrows):
                    # if sheet.row_values(rows)[4] == fold_map[fold]:
                    self.image_path_list.append(sheet.row_values(rows)[0])
                    self.y.append(sheet.row_values(rows)[1])
                    self.env.append(sheet.row_values(rows)[self.args.env_num])
                    self.u.append(float(sheet.row_values(rows)[0][-10:-6]))
                pass
                # elif self.args.env_num == 3:
                # pass
            else:  # Nips2021 dataset
                if self.root is None:
                    self.root = '../data/colored_MNIST_0.02_env_2_0_c_2/%s/' % fold
                else:
                    self.root = self.root + '%s/' % fold

                all_classes = os.listdir(self.root)
                for one_class in all_classes:
                    for filename in os.listdir(os.path.join(self.root, one_class)):
                        self.u.append(float(filename[-10:-6]))
                        self.env.append(int(filename[-5:-4]))
                        self.image_path_list.append(
                            os.path.join(self.root, one_class, filename))
                        if int(one_class) <= 4:
                            self.y.append(0)
                        else:
                            self.y.append(1)

        print(self.root)

    def __getitem__(self, index):
        # print(self.image_path_list[index])
        with open(self.image_path_list[index], 'rb') as f:
            img_1 = Image.open(f)
            if '225. cat_mm8_2-min.png' in self.image_path_list[index]:
                img_1 = np.asarray(img_1.convert('RGBA'))[:, :, :3]
                img_1 = Image.fromarray(img_1.astype('uint8'))
            else:
                img_1 = Image.fromarray(np.asarray(
                    img_1.convert('RGB')).astype('uint8'))
        if self.transform is not None:
            img_1 = self.transform(img_1)
        return img_1, \
            torch.from_numpy(np.array(self.y[index]).astype('int')), \
            torch.from_numpy(np.array(self.env[index]).astype('int')), \
            torch.from_numpy(np.array(self.u[index]).astype(
                'float32').reshape((1)))

    def __len__(self):
        return len(self.image_path_list)


def main():
    args = get_opt()
    args = make_dirs(args)
    logger = get_logger(args)
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    if args.dataset == 'AD':
        train_loader = DataLoaderX(get_dataset(root=args.root, args=args, fold='train', aug=args.aug),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=True,
                                   drop_last=True)
        test_loader = DataLoaderX(get_dataset(root=args.root, args=args, fold='test', aug=0),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True, )
        val_loader = None

        model = Generative_model_MMD(
            in_channel=args.in_channel,
            zs_dim=args.zs_dim,
            num_classes=args.num_classes,
        ).cuda()
    elif args.dataset == 'NICO':
        train_loader = DataLoaderX(get_dataset_2D_env(root=args.root, args=args, fold='train',
                                                      transform=transforms.Compose([
                                                          transforms.RandomResizedCrop(
                                                              (256, 256)),
                                                          transforms.RandomHorizontalFlip(
                                                              p=0.5),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(
                                                              (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                      ])),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=2,
                                   pin_memory=True,
                                   drop_last=True)
        test_loader = DataLoaderX(get_dataset_2D_env(root=args.root, args=args, fold='test',
                                                     transform=transforms.Compose([
                                                         transforms.Resize(
                                                             (256, 256)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                     ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True)
        val_loader = None
        if 'NICO' in args.dataset:
            model = Generative_model_2D_mnist_normal_MMD_NICO(
                in_channel=args.in_channel,
                zs_dim=args.zs_dim,
                num_classes=args.num_classes,
                dp=args.dp,
            ).cuda()

    elif 'mnist' in args.dataset:
        train_loader = DataLoader(get_dataset_2D_env(root=args.root, args=args, fold='train',
                                                     transform=transforms.Compose([
                                                         transforms.RandomHorizontalFlip(
                                                             p=0.5),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                     ])),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=True)
        test_loader = DataLoader(get_dataset_2D_env(root=args.root, args=args, fold='test',
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                    ])),
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)
        val_loader = None
        if 'NICO' in args.dataset:
            model = Generative_model_2D_mnist_normal_MMD_NICO(
                in_channel=args.in_channel,
                zs_dim=args.zs_dim,
                num_classes=args.num_classes,
                dp=args.dp,
            ).cuda()
        else:
            if args.smaller_net:
                model = Generative_model_2D_mnist_MMD_L(
                    in_channel=args.in_channel,
                    zs_dim=args.zs_dim,
                    num_classes=args.num_classes,
                    dp=args.dp,
                ).cuda()
                # model = Generative_model_2D_mnist_MMD(
                #     in_channel=args.in_channel,
                #     zs_dim=args.zs_dim,
                #     num_classes=args.num_classes,
                #     dp=args.dp,
                # ).cuda()
            else:
                model = Generative_model_2D_mnist_normal_MMD(
                    in_channel=args.in_channel,
                    zs_dim=args.zs_dim,
                    num_classes=args.num_classes,
                    dp=args.dp,
                ).cuda()

    D_net = D_model(args.zs_dim, 1024, dp=args.dp).cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params = pytorch_total_params + \
        sum(p.numel() for p in D_net.parameters())
    print('model params: ', pytorch_total_params,
          '%0.4f M' % (pytorch_total_params / 1e6))

    if args.inference_only and args.load_path != '':
        model.load_state_dict(torch.load(os.path.join(
            args.load_path, 'best_acc.pth.tar'))['state_dict'])
        test_loader_NICO = DataLoaderX(get_dataset_NICO_inter(args=args,
                                                              transform=transforms.Compose([
                                                                  transforms.Resize(
                                                                      (256, 256)),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                                       (0.5, 0.5, 0.5)),

                                                              ])),
                                       batch_size=args.test_batch_size,
                                       shuffle=False,
                                       num_workers=1,
                                       pin_memory=True)
        pred_test, label_test, test_acc = evaluate_only(
            model, test_loader_NICO, args)
        save_pred_label_as_xlsx(
            args.model_save_dir, 'pred.xls', pred_test, label_test, test_loader_NICO, args)
        logger.info('model save path: %s' % args.model_save_dir)
        print('test_acc', test_acc)
        exit(123)

    # for reconstruction
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr, weight_decay=args.reg)
    optimizer_D = optim.Adam(
        D_net.parameters(),
        lr=args.lr2, weight_decay=args.reg)

    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr,
                             args.lr_decay, args.lr_controler)
        adjust_learning_rate(optimizer_D, epoch, args.lr2,
                             args.lr_decay, args.lr_controler)
        _, _ = train(
            epoch,
            model,
            D_net,
            optimizer,
            optimizer_D,
            train_loader,
            args)
        if val_loader is not None:
            pred_val, val_acc = evaluate(model, val_loader, args)
        else:
            pred_val = None
            val_acc = -1
        pred_test, test_acc = evaluate(model, test_loader, args)
        if test_acc >= best_acc:
            best_acc = copy.deepcopy(test_acc)
            best_acc_ep = copy.deepcopy(epoch)
            is_best = 1
        else:
            is_best = 0
        other_info = {
            'pred_val': pred_val,
            'pred_test': pred_test,
        }
        checkpoint(epoch, args.model_save_dir, model,
                   is_best, other_info, logger)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d, val_acc: %0.4f'
                    % (epoch, test_acc, best_acc, best_acc_ep, val_acc))
    logger.info('model save path: %s' % args.model_save_dir)
    xlsx_name = '%s_MMD_AAE_u_%d_fold_%d.xls' % \
                (os.path.basename(sys.argv[0][:-3]), args.is_use_u,
                 args.fold)
    save_results_as_xlsx('./results/', xlsx_name, best_acc,
                         best_acc_ep, auc=None, args=args)
    logger.info('xls save path: %s' % xlsx_name)
    logger.info('*' * 50)
    logger.info('*' * 50)


if __name__ == '__main__':
    main()
