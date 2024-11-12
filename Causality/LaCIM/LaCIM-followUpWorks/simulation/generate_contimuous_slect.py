# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 10:44:00 2020

@author: xinsun
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch import distributions as dist
from torch.utils.data import Dataset

from sklearn.cluster import KMeans
# KMeans_model = KMeans(n_clusters=8,random_state = 1)
# KMeans_model.fit(x_normalization)


def to_one_hot(x, m=None):
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
    return xoh


def lrelu(x, neg_slope):
    """
    Leaky ReLU activation function
    @param x: input array
    @param neg_slope: slope for negative values
    @return:
        out: output rectified array
    """

    def _lrelu_1d(_x, _neg_slope):
        """
        one dimensional implementation of leaky ReLU
        """
        if _x > 0:
            return _x
        else:
            return _x * _neg_slope

    leaky1d = np.vectorize(_lrelu_1d)
    assert neg_slope > 0  # must be positive
    return leaky1d(x, neg_slope)


def sigmoid(x):
    """
    Sigmoid activation function
    @param x: input array
    @return:
        out: output array
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x * (x >= 0)


def generate_mixing_matrix(d_sources: int, d_data=None, dtype=np.float32):
    """
    Generate square linear mixing matrix
    @param d_sources: dimension of the latent sources
    @param d_data: dimension of the mixed data
    @param cond_threshold: higher bound on the condition number of the matrix to ensure well-conditioned problem
    @param dtype: data type for data
    @return:
        A: mixing matrix
    @rtype: np.ndarray
    """
    if d_data is None:
        d_data = d_sources

    A = (np.linalg.qr(np.random.uniform(-1, 1, (d_sources, d_data)))
         [0]).astype(dtype)
    return A


def generate_select(num_cluster, n_per_clus, dim_c: int, dim_z: int, K_mu=None, seed=None, dtype=np.float32):

    if seed is None:
        seed = 5
    np.random.seed(seed)
    torch.manual_seed(seed)

    if K_mu is None:
        K_mu = np.zeros((num_cluster, dim_c))
        for k in range(num_cluster):
            K_mu[k, :] = (np.random.randn(dim_c)+5*k)*2

    # Distribution
    dist_clus = []
    for k in range(np.shape(K_mu)[0]):
        mu = K_mu[k, :]
        dist_clus.append(dist.normal.Normal(
            torch.Tensor(mu), torch.ones(dim_c)))

    # Generate c and label
    #sigma = [[1,0],[0,1]]
    for i in range(num_cluster):
        if i == 0:
            label = np.zeros(n_per_clus)
            # c = np.random.multivariate_normal(K_mu[i,:],sigma,n_per_clus) # every sample has a c
        else:
            label = np.r_[label, i*np.ones(n_per_clus,)]
            # c = np.r_[c,np.random.multivariate_normal(K_mu[i,:],sigma,n_per_clus)] # every sample has a c
    #A = generate_mixing_matrix(dim_c, dim_z, dtype=np.float32)
    #c_clus = np.matmul(K_mu,np.linalg.inv(A))
    c = np.matmul(to_one_hot(label, num_cluster)[0], K_mu).astype(dtype)
    return c, label, K_mu


def generate_zs_selectBias(n, num_cluster, dim_c, dim_z, dim_s, seed=None):

    n = n
    dim_c = dim_c

    if seed is None:
        seed = 5

    #c_clus = np.unique(c,axis=0)
    # label_c = np.array(to_one_hot(np.array(range(K_mu.shape[0]))))[0]
    np.random.seed(seed)
    torch.manual_seed(seed)

    #As_mu = generate_mixing_matrix(dim_c, dim_s, dtype=np.float32)
    #Az_mu = generate_mixing_matrix(dim_c, dim_z, dtype=np.float32)
    Az_sigma = generate_mixing_matrix(dim_z, dim_z, dtype=np.float32)
    As_sigma = generate_mixing_matrix(dim_z, dim_s, dtype=np.float32)

    # #sijie used before
    # K_mu = np.zeros((dim_z))
    # K_mu[:] = (np.random.randn(dim_z))*2

    # sijie modify
    K_mu = np.zeros((num_cluster, dim_c))
    for k in range(num_cluster):
        K_mu[k, :] = (np.random.randn(dim_c)+5*k)*2
    n_per_clus = int(n/K_mu.shape[0])

    mu_z = K_mu
    log_sigma_z = np.matmul(K_mu, Az_sigma)
    mu_s = K_mu
    log_sigma_s = np.matmul(K_mu, As_sigma)

    sigma_z = np.exp(log_sigma_z/np.max(np.abs(log_sigma_z))/2)
    sigma_s = np.exp(log_sigma_s/np.max(np.abs(log_sigma_s))/2)
    _dist_z = dist.normal.Normal(torch.Tensor(mu_z), torch.Tensor(sigma_z))
    _dist_s = dist.normal.Normal(torch.Tensor(mu_s), torch.Tensor(sigma_s))
    z = _dist_z.sample((n,))
    s = _dist_s.sample((n,))

    return z, s, mu_z, sigma_z, mu_s, sigma_s


def generate_zs(n, num_cluster, dim_c, dim_z, dim_s, seed=None):

    n = n
    dim_c = dim_c

    if seed is None:
        seed = 5

    #c_clus = np.unique(c,axis=0)
    # label_c = np.array(to_one_hot(np.array(range(K_mu.shape[0]))))[0]
    np.random.seed(seed)
    torch.manual_seed(seed)

    #As_mu = generate_mixing_matrix(dim_c, dim_s, dtype=np.float32)
    #Az_mu = generate_mixing_matrix(dim_c, dim_z, dtype=np.float32)
    Az_sigma = generate_mixing_matrix(dim_c, dim_z, dtype=np.float32)
    As_sigma = generate_mixing_matrix(dim_c, dim_s, dtype=np.float32)

    # #sijie used before
    # K_mu = np.zeros((dim_z))
    # K_mu[:] = (np.random.randn(dim_z))*2

    # sijie modify
    K_mu = np.zeros((num_cluster, dim_c))
    for k in range(num_cluster):
        K_mu[k, :] = (np.random.randn(dim_c)+5*k)*2
    n_per_clus = int(n/K_mu.shape[0])

    # K_mu_z = np.zeros((num_cluster, dim_c))
    # for k in range(num_cluster):
    #     K_mu_z[k, :] = (np.random.randn(dim_c)+1)*2
    # sijie 0614 xiugai butong de kmu_z
    mu_z = K_mu
    log_sigma_z = np.matmul(K_mu, Az_sigma)
    mu_s = K_mu
    log_sigma_s = np.matmul(K_mu, As_sigma)

    sigma_z = np.exp(log_sigma_z/np.max(np.abs(log_sigma_z))/2)
    sigma_s = np.exp(log_sigma_s/np.max(np.abs(log_sigma_s))/2)
    # _dist_z = dist.normal.Normal(torch.Tensor(mu_z), torch.Tensor(sigma_z))
    # _dist_s = dist.normal.Normal(torch.Tensor(mu_s), torch.Tensor(sigma_s))
    # z = _dist_z.sample((n,))
    # s = _dist_s.sample((n,))

    for k in range(K_mu.shape[0]):
        mu_z_k = mu_z[k, :]
        sigma_z_k = sigma_z[k, :]
        mu_s_k = mu_s[k, :]
        sigma_s_k = sigma_s[k, :]
        _dist_z = dist.normal.Normal(
            torch.Tensor(mu_z_k), torch.Tensor(sigma_z_k))
        _dist_s = dist.normal.Normal(
            torch.Tensor(mu_s_k), torch.Tensor(sigma_s_k))
        if k == 0:
            z = _dist_z.sample((n_per_clus,))
            s = _dist_s.sample((n_per_clus,))
        else:
            z = np.r_[z, _dist_z.sample((n_per_clus,))]
            s = np.r_[s, _dist_s.sample((n_per_clus,))]

    return z, s, mu_z, sigma_z, mu_s, sigma_s


def generate_c(num_cluster, s, z, seed=None, dtype=np.float32):

    # latent = np.c_[z,_dist_z.sample((n_per_clus,))]
    # print(s, s.size(), z, z.size())
    # latent = torch.cat([s, z], dim=1)
    latent = np.c_[z, s]
    KMeans_model = KMeans(n_clusters=num_cluster, random_state=1)
    KMeans_model.fit(latent)
    label = KMeans_model.labels_
    if seed is None:
        seed = 5
    np.random.seed(seed)
    torch.manual_seed(seed)

    # if K_mu is None:
    #     K_mu = np.zeros((num_cluster,dim_c))
    #     for k in range(num_cluster):
    #         K_mu[k,:] = (np.random.randn(dim_c)+5*k)*2

    # # Distribution
    # dist_clus = []
    # for k in range(np.shape(K_mu)[0]): # K_mu jiushi d^e
    #     mu = K_mu[k,:]
    #     dist_clus.append(dist.normal.Normal(torch.Tensor(mu), torch.ones(dim_c)))

    # Generate c and label
    #sigma = [[1,0],[0,1]]
    # for i in range(num_cluster):
    #     if i == 0:
    #         label = np.zeros(n_per_clus)
    #         #c = np.random.multivariate_normal(K_mu[i,:],sigma,n_per_clus) # every sample has a c
    #     else:
    #         label = np.r_[label,i*np.ones(n_per_clus,)]
    #         #c = np.r_[c,np.random.multivariate_normal(K_mu[i,:],sigma,n_per_clus)] # every sample has a c
    # #A = generate_mixing_matrix(dim_c, dim_z, dtype=np.float32)
    # #c_clus = np.matmul(K_mu,np.linalg.inv(A))
    c = to_one_hot(label, num_cluster)

    # c = label
    return c, label  # TODO C label haixuyao tiaoshi


def generate_x(z, s, dim_x, n_layers, activation, slope=None, dtype=np.float32, seed=None):
    zs = np.c_[z, s]
    [n, dim_zs] = np.shape(zs)
    if seed is None:
        seed = 5

    np.random.seed(seed)
    torch.manual_seed(seed)

    if activation == 'lrelu':
        def act_f(x): return lrelu(x, slope).astype(dtype)
    elif activation == 'sigmoid':
        act_f = sigmoid
    elif activation == 'xtanh':
        def act_f(x): return np.tanh(x) + slope * x
    elif activation == 'relu':
        def act_f(x): return relu(x)
    elif activation == 'none':
        def act_f(x): return x
    else:
        raise ValueError('incorrect non linearity: {}'.format(activation))

    A = generate_mixing_matrix(dim_zs, dim_x, dtype=np.float32)
    X = act_f(np.matmul(zs, A))

    for i in range(1, n_layers):
        A = generate_mixing_matrix(dim_x, dim_x, dtype=np.float32)
        if i == n_layers-1:
            X = np.matmul(X, A)
        else:
            print("X, A", X[0], A[0])
            X = act_f(np.matmul(X, A))
    X += np.random.randn(*X.shape)
    return X


def generate_y(s, dim_y, n_layers, activation, slope=None, dtype=np.float32, seed=None):

    [n, dim_s] = np.shape(s)
    if seed is None:
        seed = 5
    np.random.seed(seed)
    torch.manual_seed(seed)

    if activation == 'lrelu':
        def act_f(x): return lrelu(x, slope).astype(dtype)
    elif activation == 'sigmoid':
        act_f = sigmoid
    elif activation == 'xtanh':
        def act_f(x): return np.tanh(x) + slope * x
    elif activation == 'relu':
        def act_f(x): return relu(x)
    elif activation == 'none':
        def act_f(x): return x
    else:
        raise ValueError('incorrect non linearity: {}'.format(activation))

    A = generate_mixing_matrix(dim_s, dim_y, dtype=np.float32)
    y = act_f(np.matmul(s, A))

    for i in range(1, n_layers):
        A = generate_mixing_matrix(dim_y, dim_y, dtype=np.float32)
        if i == n_layers-1:
            y = np.matmul(y, A)
        else:
            y = act_f(np.matmul(y, A))
    #y = np.argmax()
    return y


def save_data(num_data, num_cluster, n_per_clus, dim_c, dim_z, dim_s, dim_x, dim_y, n_layers, activation, slope=0.5, K_mu=None, dtype=np.float32, seed=None):

    path = '/home/ruxin2/nips2021_zhuankan_data_SB/'
    path += str(num_data)
    path += '_' + str(dim_c)
    path += '_' + str(dim_z)
    path += '_' + str(dim_s)
    path += '_' + str(dim_x)
    path += '_' + str(dim_y)
    path += '_' + str(n_layers)
    path += '_' + str(activation)
    path += '_' + str(seed)
    n = num_data
    z, s, mu_z, sigma_z, mu_s, sigma_s = generate_zs(
        n, num_cluster, dim_c, dim_z, dim_s, seed)
    print("z", z[:5])
    print("s", s[:5])
    c, label = generate_c(num_cluster, s, z, seed, dtype)
    c = to_one_hot(label, num_cluster)[0].astype(dtype)
    label = to_one_hot(label, num_cluster)[0].astype(dtype)
    # c, label,K_mu = generate_c(num_cluster, n_per_clus, dim_c, dim_s, K_mu,seed,dtype)
    # label = to_one_hot(label,num_cluster)[0].astype(dtype)
    # z,s,mu_z,sigma_z,mu_s,sigma_s = generate_zs(n, dim_z,dim_s,seed)
    X = generate_x(z, s, dim_x, n_layers, activation, slope, dtype, seed)
    y = generate_y(s, dim_y, n_layers, activation, slope, dtype, seed)
    print('Creating dataset {} ...'.format(path))
    dir_path = '/'.join(path.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    np.savez_compressed(path, s=s, z=z, x=X, y=y, c=c, label=label,
                        mu_z=mu_z, sigma_z=sigma_z, mu_s=mu_s, sigma_s=sigma_s)
    print(' ... done')
    return path


class SyntheticDataset(Dataset):
    def __init__(self, path, device='cpu'):
        self.device = device
        self.path = path
        try:
            data = np.load(path)
        except:
            pass
        self.data = data
        self.s = torch.from_numpy(data['s']).to(self.device)
        self.z = torch.from_numpy(data['z']).to(self.device)
        self.y = torch.from_numpy(data['y']).to(self.device)
        self.x = torch.from_numpy(data['x']).to(self.device)
        self.c = torch.from_numpy(data['c']).to(self.device)
        self.mu_z = torch.from_numpy(data['mu_z']).to(self.device)
        self.sigma_z = torch.from_numpy(data['sigma_z']).to(self.device)
        self.mu_s = torch.from_numpy(data['mu_s']).to(self.device)
        self.sigma_s = torch.from_numpy(data['sigma_s']).to(self.device)
        self.label = torch.from_numpy(data['label']).to(self.device)
        self.len = self.x.shape[0]
        self.dim_s = self.s.shape[1]
        self.dim_z = self.z.shape[1]
        self.dim_c = self.c.shape[1]
        self.dim_x = self.x.shape[1]
        self.dim_y = self.y.shape[1]
        #self.nps = int(self.len / self.dim_c)
        print('data loaded on {}'.format(self.x.device))

    def get_dims(self):
        return self.dim_x, self.dim_y, self.dim_s, self.dim_z, self.dim_c

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], \
            self.y[index], \
            self.s[index], \
            self.z[index], \
            self.c[index], \
            self.label[index]
