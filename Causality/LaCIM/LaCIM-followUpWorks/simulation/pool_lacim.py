# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:38:36 2020

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
#from generate_discrete_c import SyntheticDataset
#from generate_discrete_c import save_data
from generate_contimuous_slect import SyntheticDataset
from generate_contimuous_slect import save_data
from causal_ivae import Causal_VAE
import matplotlib
matplotlib.use('Agg')


dim_z = 2
dim_s = 2
dim_x = 4
dim_y = 2
dim_c = 2
n_per_clus = 1000
num_cluster = 7
n_layers = 3
activation = 'lrelu'
device = 'cpu'
color = [["purple"], ["royalblue"], [
    "lightseagreen"], ["lightgreen"], ["yellow"]]

if_plot = 1

num_data = 2000
num_trial = 20
z_perf_vae = np.zeros([num_data, num_trial])
s_perf_vae = np.zeros([num_data, num_trial])
z_vae_list_data = []
s_vae_list_data = []
hist_vae_perf_z_data = []
hist_vae_perf_s_data = []
i = 0
while i < 5:
    path = save_data(num_data, num_cluster, n_per_clus, dim_c, dim_z, dim_s, dim_x,
                     dim_y, n_layers, activation, slope=0.5, dtype=np.float32, seed=i)
    path += '.npz'
    dset = SyntheticDataset(path, device)
    N = dset.len
    train_loader = DataLoader(dset, shuffle=True, batch_size=512)
    x_gt = dset.x
    y_gt = dset.y
    c_gt = dset.c
    z_gt = dset.z
    s_gt = dset.s
    if if_plot:
        fig = plt.figure()
        # fig.patch.set_facecolor('FloralWhite')
        ax = plt.axes()
        # ax.set_facecolor("FloralWhite")
        plt.scatter(s_gt[c_gt.argmax(dim=1) == 0, 1],
                    s_gt[c_gt.argmax(dim=1) == 0, 0], c=color[0])
        plt.scatter(s_gt[c_gt.argmax(dim=1) == 1, 1],
                    s_gt[c_gt.argmax(dim=1) == 1, 0], c=color[1])
        plt.scatter(s_gt[c_gt.argmax(dim=1) == 2, 1],
                    s_gt[c_gt.argmax(dim=1) == 2, 0], c=color[2])
        plt.scatter(s_gt[c_gt.argmax(dim=1) == 3, 1],
                    s_gt[c_gt.argmax(dim=1) == 3, 0], c=color[3])
        plt.scatter(s_gt[c_gt.argmax(dim=1) == 4, 1],
                    s_gt[c_gt.argmax(dim=1) == 4, 0], c=color[4])
        # plt.show()
        plt.savefig('./simu_img/pic_pool-lacim_{}_i_{}.png'.format("gt_s", i))
        print("*** Finish ploting pool-lacim_gt_s _i_{} !!! ***".format(i))
    j = 0
    while j < 20:
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

        if if_plot:
            fig = plt.figure()
            # fig.patch.set_facecolor('FloralWhite')
            ax = plt.axes()
            # ax.set_facecolor("FloralWhite")
            plt.scatter(s_vae_plot[c_gt.argmax(dim=1) == 0, 1],
                        s_vae_plot[c_gt.argmax(dim=1) == 0, 0], c=color[0])
            plt.scatter(s_vae_plot[c_gt.argmax(dim=1) == 1, 1],
                        s_vae_plot[c_gt.argmax(dim=1) == 1, 0], c=color[1])
            plt.scatter(s_vae_plot[c_gt.argmax(dim=1) == 2, 1],
                        s_vae_plot[c_gt.argmax(dim=1) == 2, 0], c=color[2])
            plt.scatter(s_vae_plot[c_gt.argmax(dim=1) == 3, 1],
                        s_vae_plot[c_gt.argmax(dim=1) == 3, 0], c=color[3])
            plt.scatter(s_vae_plot[c_gt.argmax(dim=1) == 4, 1],
                        s_vae_plot[c_gt.argmax(dim=1) == 4, 0], c=color[4])
            # plt.show()
            plt.savefig(
                './simu_img/pic_poollacim_{}_i_{}_j_{}.png'.format("s_vae", i, j))
            print("*** Finish ploting poollacim_s_vae _i_{}_j_{} !!! ***".format(i, j))
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
        with open(path[:-6]+'_%s' % (os.path.basename(__file__)[:-3]) + '/check.pkl', 'wb') as f:
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

    with open(path[:-6]+'_%s' % (os.path.basename(__file__)[:-3]) + '/final.pkl', 'wb') as f:
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
