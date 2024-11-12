# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:26:49 2020

@author: xinsun
"""
import argparse
import matplotlib.pyplot as plt
import os
import pickle
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from metrics import mean_corr_coef
# from generate_discrete_c import SyntheticDataset
# from generate_discrete_c import save_data
from generate_continuous_c import SyntheticDataset
from generate_continuous_c import save_data
from causal_ivae import Causal_iVAE
from causal_ivae import Causal_VAE
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='PyTorch')
# Model parameters
parser.add_argument('--cuda', type=str, default='cpu')
parser.add_argument('--num_data', type=int, default=3000)
parser.add_argument('--num_cluster', type=int, default=5)
parser.add_argument('--if_plot', type=int, default=0)

args = parser.parse_args()

device = args.cuda
if_plot = args.if_plot
dim_z = 2
dim_s = 2
dim_x = 4
dim_y = 2
dim_c = 2
num_data = args.num_data
n_per_clus = 1000
num_cluster = args.num_cluster
n_layers = 3
activation = 'lrelu'
color = [["purple"], ["royalblue"], [
    "lightseagreen"], ["lightgreen"], ["yellow"]]

# IVAE

# num_data = 5
num_trial = 20
z_perf_ivae_discrete_5 = np.zeros([num_data, num_trial])
s_perf_ivae_discrete_5 = np.zeros([num_data, num_trial])
z_ivae_list_data_discrete_5 = []
s_ivae_list_data_discrete_5 = []
hist_ivae_perf_z_data_discrete_5 = []
hist_ivae_perf_s_data_discrete_5 = []

z_perf_vae = np.zeros([num_data, num_trial])
s_perf_vae = np.zeros([num_data, num_trial])
z_vae_list_data = []
s_vae_list_data = []
hist_vae_perf_z_data = []
hist_vae_perf_s_data = []

i = 0
print('train 5 times')
while i < 5:
    hist_ivae_perf_z_discrete_5 = []
    hist_ivae_perf_s_discrete_5 = []
    dim_c = 2
    path = save_data(num_cluster, n_per_clus, dim_c, dim_z, dim_s, dim_x, dim_y, n_layers, activation, slope=0.5,
                     dtype=np.float32, seed=i)
    path += '.npz'
    print(path)
    dset = SyntheticDataset(path, device)
    N = dset.len
    train_loader = DataLoader(dset, shuffle=True, batch_size=512)
    x_gt = dset.x
    y_gt = dset.y
    c_gt = dset.c
    z_gt = dset.z
    s_gt = dset.s
    label_gt = dset.label
    # print(c_gt[:10])
    # print(label_gt[2080:2100])
    if if_plot:
        fig = plt.figure()
        # fig.patch.set_facecolor('FloralWhite')
        ax = plt.axes()
        # ax.set_facecolor("FloralWhite")
        plt.scatter(s_gt[label_gt.argmax(dim=1) == 0, 1],
                    s_gt[label_gt.argmax(dim=1) == 0, 0], c=color[0])
        plt.scatter(s_gt[label_gt.argmax(dim=1) == 1, 1],
                    s_gt[label_gt.argmax(dim=1) == 1, 0], c=color[1])
        plt.scatter(s_gt[label_gt.argmax(dim=1) == 2, 1],
                    s_gt[label_gt.argmax(dim=1) == 2, 0], c=color[2])
        plt.scatter(s_gt[label_gt.argmax(dim=1) == 3, 1],
                    s_gt[label_gt.argmax(dim=1) == 3, 0], c=color[3])
        plt.scatter(s_gt[label_gt.argmax(dim=1) == 4, 1],
                    s_gt[label_gt.argmax(dim=1) == 4, 0], c=color[4])
        # plt.show()
        plt.savefig('./CB_simu_img/pic_lacim_{}_i_{}.png'.format("gt_s", i))
        print("*** Finish ploting lacim_gt_s _i_{} !!! ***".format(i))

    if if_plot:
        fig = plt.figure()
        # fig.patch.set_facecolor('FloralWhite')
        ax = plt.axes()
        # ax.set_facecolor("FloralWhite")
        plt.scatter(z_gt[label_gt.argmax(dim=1) == 0, 1],
                    z_gt[label_gt.argmax(dim=1) == 0, 0], c=color[0])
        plt.scatter(z_gt[label_gt.argmax(dim=1) == 1, 1],
                    z_gt[label_gt.argmax(dim=1) == 1, 0], c=color[1])
        plt.scatter(z_gt[label_gt.argmax(dim=1) == 2, 1],
                    z_gt[label_gt.argmax(dim=1) == 2, 0], c=color[2])
        plt.scatter(z_gt[label_gt.argmax(dim=1) == 3, 1],
                    z_gt[label_gt.argmax(dim=1) == 3, 0], c=color[3])
        plt.scatter(z_gt[label_gt.argmax(dim=1) == 4, 1],
                    z_gt[label_gt.argmax(dim=1) == 4, 0], c=color[4])
        # plt.show()
        plt.savefig('./CB_simu_img/pic_lacim_{}_i_{}.png'.format("gt_z", i))
        print("*** Finish ploting lacim_gt_z _i_{} !!! ***".format(i))

    dim_c = label_gt.shape[1]
    # lacim
    j = 0
    while j < 10:
        model_ivae = Causal_iVAE(dim_z, dim_s, dim_x, dim_y, dim_c)
        optimizer_ivae = optim.Adam(model_ivae.parameters(), lr=0.5 * 1e-3)
        scheduler_ivae = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ivae, factor=0.1, patience=10, verbose=True)
        hist_ivae_perf_z = []
        hist_ivae_perf_s = []
        z_ivae_list = []
        s_ivae_list = []
        start_time = time.time()
        for k in range(2000):
            if k % 400 == 0 and k > 0:
                print(i, j, k, 'using: ', time.time() - start_time)
                start_time = time.time()
            x, y, s, z, _, c = next(iter(train_loader))
            optimizer_ivae.zero_grad()
            elbo, elbo_1, elbo_2, elbo_5, z, s, decoder_x_params, decoder_y_params, prior_params = model_ivae.elbo(x, y,
                                                                                                                   c)
            elbo.backward()
            optimizer_ivae.step()
            if torch.isnan(elbo) or torch.isinf(elbo):
                break

            _, _, _, _, z_ivae, s_ivae, _, _, _ = model_ivae.elbo(
                x_gt, y_gt, label_gt)
            if torch.sum(torch.isnan(z_ivae)) == 0 and torch.sum(torch.isnan(s_ivae)) == 0:
                perf_z = mean_corr_coef(z_gt.numpy(), z_ivae.detach().numpy())
                perf_s = mean_corr_coef(s_gt.numpy(), s_ivae.detach().numpy())
                hist_ivae_perf_z.append(perf_z)
                hist_ivae_perf_s.append(perf_s)

        if torch.isnan(elbo) or torch.isinf(elbo):
            continue

        label = dset.label
        _, _, _, _, z_ivae, s_ivae, _, _, _ = model_ivae.elbo(
            x_gt, y_gt, label_gt)
        s_ivae_plot = s_ivae.detach().numpy()
        z_ivae_plot = z_ivae.detach().numpy()

        # print("s_ivae_plot", s_ivae_plot[:5])
        if if_plot:
            fig = plt.figure()
            # fig.patch.set_facecolor('FloralWhite')
            ax = plt.axes()
            # ax.set_facecolor("FloralWhite")
            plt.scatter(s_ivae_plot[label_gt.argmax(dim=1) == 0, 1],
                        s_ivae_plot[label_gt.argmax(dim=1) == 0, 0], c=color[0])
            plt.scatter(s_ivae_plot[label_gt.argmax(dim=1) == 1, 1],
                        s_ivae_plot[label_gt.argmax(dim=1) == 1, 0], c=color[1])
            plt.scatter(s_ivae_plot[label_gt.argmax(dim=1) == 2, 1],
                        s_ivae_plot[label_gt.argmax(dim=1) == 2, 0], c=color[2])
            plt.scatter(s_ivae_plot[label_gt.argmax(dim=1) == 3, 1],
                        s_ivae_plot[label_gt.argmax(dim=1) == 3, 0], c=color[3])
            plt.scatter(s_ivae_plot[label_gt.argmax(dim=1) == 4, 1],
                        s_ivae_plot[label_gt.argmax(dim=1) == 4, 0], c=color[4])
            # plt.show()
            plt.savefig(
                './CB_simu_img/pic_lacim_{}_i_{}_j_{}.png'.format("s_ivae", i, j))
            print("*** Finish ploting lacim_s_ivae _i_{}_j_{} !!! ***".format(i, j))
        if if_plot:
            fig = plt.figure()
            # fig.patch.set_facecolor('FloralWhite')
            ax = plt.axes()
            # ax.set_facecolor("FloralWhite")
            plt.scatter(z_ivae_plot[label_gt.argmax(dim=1) == 0, 1],
                        z_ivae_plot[label_gt.argmax(dim=1) == 0, 0], c=color[0])
            plt.scatter(z_ivae_plot[label_gt.argmax(dim=1) == 1, 1],
                        z_ivae_plot[label_gt.argmax(dim=1) == 1, 0], c=color[1])
            plt.scatter(z_ivae_plot[label_gt.argmax(dim=1) == 2, 1],
                        z_ivae_plot[label_gt.argmax(dim=1) == 2, 0], c=color[2])
            plt.scatter(z_ivae_plot[label_gt.argmax(dim=1) == 3, 1],
                        z_ivae_plot[label_gt.argmax(dim=1) == 3, 0], c=color[3])
            plt.scatter(z_ivae_plot[label_gt.argmax(dim=1) == 4, 1],
                        z_ivae_plot[label_gt.argmax(dim=1) == 4, 0], c=color[4])
            # plt.show()
            plt.savefig(
                './CB_simu_img/pic_lacim_{}_i_{}_j_{}.png'.format("z_ivae", i, j))
            print("*** Finish ploting lacim_z_ivae _i_{}_j_{} !!! ***".format(i, j))
        perf_ivae_z = mean_corr_coef(z_gt.numpy(), z_ivae.detach().numpy())
        perf_ivae_s = mean_corr_coef(s_gt.numpy(), s_ivae.detach().numpy())
        z_ivae_list.append(z_ivae)
        s_ivae_list.append(s_ivae)
        z_perf_ivae_discrete_5[i, j] = perf_ivae_z
        s_perf_ivae_discrete_5[i, j] = perf_ivae_s
        hist_ivae_perf_z_discrete_5.append(hist_ivae_perf_z)
        hist_ivae_perf_s_discrete_5.append(hist_ivae_perf_s)
        j += 1
        print('This is the (%s,%s)-th experiment, the z_perf is %0.5f, the s_perf is %0.5f' % (
            i, j, perf_ivae_z, perf_ivae_s))

        if not os.path.exists(path[:-6] + '_%s' % (os.path.basename(__file__)[:-3]) + '/'):
            os.makedirs(path[:-6] + '_%s' %
                        (os.path.basename(__file__)[:-3]) + '/')
        with open(path[:-6] + '_%s' % (os.path.basename(__file__)[:-3]) + '/check.pkl', 'wb') as f:
            dicts = {
                'i': i, 'j': j, 'k': k,
                'z_perf_ivae_discrete_5': z_perf_ivae_discrete_5,
                's_perf_ivae_discrete_5': s_perf_ivae_discrete_5,
                'z_ivae_list_data_discrete_5': z_ivae_list_data_discrete_5,
                's_ivae_list_data_discrete_5': s_ivae_list_data_discrete_5,
                'hist_ivae_perf_z_data_discrete_5': hist_ivae_perf_z_data_discrete_5,
                'hist_ivae_perf_s_data_discrete_5': hist_ivae_perf_s_data_discrete_5
            }
            pickle.dump(dicts, f)

    z_ivae_list_data_discrete_5.append(z_ivae_list)
    s_ivae_list_data_discrete_5.append(s_ivae_list)
    hist_ivae_perf_z_data_discrete_5.append(hist_ivae_perf_z_discrete_5)
    hist_ivae_perf_s_data_discrete_5.append(hist_ivae_perf_s_discrete_5)
    i += 1

    with open(path[:-6] + '_%s' % (os.path.basename(__file__)[:-3]) + '/final.pkl', 'wb') as f:
        dicts = {
            'i': i, 'j': j, 'k': k,
            'z_perf_ivae_discrete_5': z_perf_ivae_discrete_5,
            's_perf_ivae_discrete_5': s_perf_ivae_discrete_5,
            'z_ivae_list_data_discrete_5': z_ivae_list_data_discrete_5,
            's_ivae_list_data_discrete_5': s_ivae_list_data_discrete_5,
            'hist_ivae_perf_z_data_discrete_5': hist_ivae_perf_z_data_discrete_5,
            'hist_ivae_perf_s_data_discrete_5': hist_ivae_perf_s_data_discrete_5
        }
        pickle.dump(dicts, f)

    #  pool lacim
    j = 0
    while j < 10:
        model_vae = Causal_VAE(dim_z, dim_s, dim_x, dim_y, device=device)
        optimizer_vae = optim.Adam(model_vae.parameters(), lr=0.5*1e-3)
        scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_vae, factor=0.1, patience=10, verbose=True)
        hist_vae_perf_z = []
        hist_vae_perf_s = []
        z_vae_list = []
        s_vae_list = []
        for k in range(2000):
            x, y, s, z, c, _ = next(iter(train_loader))
            if k % 500 == 0 and k > 0:
                print('k', k, 'j', j, 'i', i)
            x, y, s, z, c = x.to(device), y.to(device), s.to(
                device), z.to(device), c.to(device)
            optimizer_vae.zero_grad()
            elbo, elbo_1, elbo_2, elbo_3, z, s, decoder_x_params, decoder_y_params, prior_params = model_vae.elbo(
                x, y)
            elbo.backward()
            optimizer_vae.step()
            if torch.isnan(elbo) or torch.isinf(elbo):
                break

            _, _, _, _, z_vae, s_vae, _, _, _ = model_vae.elbo(x_gt, y_gt)
            if torch.sum(torch.isnan(z_vae)) == 0 and torch.sum(torch.isnan(s_vae)) == 0:
                perf_z = mean_corr_coef(
                    z_gt.cpu().numpy(), z_vae.detach().cpu().numpy())
                perf_s = mean_corr_coef(
                    s_gt.cpu().numpy(), s_vae.detach().cpu().numpy())
                hist_vae_perf_z.append(perf_z)
                hist_vae_perf_s.append(perf_s)

        if torch.isnan(elbo) or torch.isinf(elbo):
            continue

        label = dset.label
        _, _, _, _, z_vae, s_vae, _, _, _ = model_vae.elbo(x_gt, y_gt)
        s_vae_plot = s_vae.detach().numpy()
        z_vae_plot = z_vae.detach().numpy()

        if if_plot:
            fig = plt.figure()
            # fig.patch.set_facecolor('FloralWhite')
            ax = plt.axes()
            # ax.set_facecolor("FloralWhite")
            plt.scatter(s_vae_plot[label_gt.argmax(dim=1) == 0, 1],
                        s_vae_plot[label_gt.argmax(dim=1) == 0, 0], c=color[0])
            plt.scatter(s_vae_plot[label_gt.argmax(dim=1) == 1, 1],
                        s_vae_plot[label_gt.argmax(dim=1) == 1, 0], c=color[1])
            plt.scatter(s_vae_plot[label_gt.argmax(dim=1) == 2, 1],
                        s_vae_plot[label_gt.argmax(dim=1) == 2, 0], c=color[2])
            plt.scatter(s_vae_plot[label_gt.argmax(dim=1) == 3, 1],
                        s_vae_plot[label_gt.argmax(dim=1) == 3, 0], c=color[3])
            plt.scatter(s_vae_plot[label_gt.argmax(dim=1) == 4, 1],
                        s_vae_plot[label_gt.argmax(dim=1) == 4, 0], c=color[4])
            # plt.show()
            plt.savefig(
                './CB_simu_img/pic_poollacim_{}_i_{}_j_{}.png'.format("s_vae", i, j))
            print("*** Finish ploting poollacim_s_vae _i_{}_j_{} !!! ***".format(i, j))

        if if_plot:
            fig = plt.figure()
            # fig.patch.set_facecolor('FloralWhite')
            ax = plt.axes()
            # ax.set_facecolor("FloralWhite")
            plt.scatter(z_vae_plot[label_gt.argmax(dim=1) == 0, 1],
                        z_vae_plot[label_gt.argmax(dim=1) == 0, 0], c=color[0])
            plt.scatter(z_vae_plot[label_gt.argmax(dim=1) == 1, 1],
                        z_vae_plot[label_gt.argmax(dim=1) == 1, 0], c=color[1])
            plt.scatter(z_vae_plot[label_gt.argmax(dim=1) == 2, 1],
                        z_vae_plot[label_gt.argmax(dim=1) == 2, 0], c=color[2])
            plt.scatter(z_vae_plot[label_gt.argmax(dim=1) == 3, 1],
                        z_vae_plot[label_gt.argmax(dim=1) == 3, 0], c=color[3])
            plt.scatter(z_vae_plot[label_gt.argmax(dim=1) == 4, 1],
                        z_vae_plot[label_gt.argmax(dim=1) == 4, 0], c=color[4])
            # plt.show()
            plt.savefig(
                './CB_simu_img/pic_poollacim_{}_i_{}_j_{}.png'.format("z_vae", i, j))
            print("*** Finish ploting poollacim_z_vae _i_{}_j_{} !!! ***".format(i, j))
        perf_vae_z = mean_corr_coef(
            z_gt.cpu().numpy(), z_vae.detach().cpu().numpy())
        perf_vae_s = mean_corr_coef(
            s_gt.cpu().numpy(), s_vae.detach().cpu().numpy())
        z_vae_list.append(z_vae)
        s_vae_list.append(s_vae)
        z_perf_vae[i, j] = perf_vae_z
        s_perf_vae[i, j] = perf_vae_s
        j += 1
        print('This is the (%s,%s)-th experiment, the z_perf is %0.5f, the s_perf is %0.5f' %
              (i, j, perf_vae_z, perf_vae_s))

        if not os.path.exists(path[:-6]+'_%s' % (os.path.basename(__file__)[:-3]) + '/'):
            os.makedirs(path[:-6]+'_%s' %
                        (os.path.basename(__file__)[:-3]) + '/')
        with open(path[:-6]+'_%s' % (os.path.basename(__file__)[:-3]) + '/poollacim_check.pkl', 'wb') as f:
            dicts = {
                'i': i, 'j': j, 'k': k,
                'z_perf_vae': z_perf_vae,
                's_perf_vae': s_perf_vae,
                'z_vae_list_data': z_vae_list_data,
                's_vae_list_data': s_vae_list_data,
                'hist_vae_perf_z_data': hist_vae_perf_z_data,
                'hist_vae_perf_s_data': hist_vae_perf_s_data
            }
            pickle.dump(dicts, f)

    z_vae_list_data.append(z_vae_list)
    s_vae_list_data.append(s_vae_list)
    hist_vae_perf_z_data.append(hist_vae_perf_z)
    hist_vae_perf_s_data.append(hist_vae_perf_s)
    i += 1

    with open(path[:-6]+'_%s' % (os.path.basename(__file__)[:-3]) + '/poollacim_final.pkl', 'wb') as f:
        dicts = {
            'i': i, 'j': j, 'k': k,
            'z_perf_vae': z_perf_vae,
            's_perf_vae': s_perf_vae,
            'z_vae_list_data': z_vae_list_data,
            's_vae_list_data': s_vae_list_data,
            'hist_vae_perf_z_data': hist_vae_perf_z_data,
            'hist_vae_perf_s_data': hist_vae_perf_s_data
        }
        pickle.dump(dicts, f)
