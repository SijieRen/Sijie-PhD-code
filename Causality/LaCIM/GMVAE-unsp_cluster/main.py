"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Main file to execute the model on the MNIST dataset

"""
from causal_dataloader import get_dataset, get_dataset_2D, DataLoaderX, save_results_as_xlsx
from model.GMVAE import *
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch
import os
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
matplotlib.use('agg')

#########################################################
# Input Parameters
#########################################################
parser = argparse.ArgumentParser(
    description='PyTorch Implementation of DGM Clustering')

# Used only in notebooks
parser.add_argument('-f', '--file',
                    help='Path for input file. First line should contain number of lines to search in')

# Dataset
# parser.add_argument('--dataset', type=str, choices=['mnist'],
#                     default='mnist', help='dataset (default: mnist)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')

# GPU
parser.add_argument('--cuda', type=int, default=1,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0,
                    help='set gpu id to use (default: 0)')

# Training
parser.add_argument('--epochs', type=int, default=100,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_val', default=200, type=int,
                    help='mini-batch size of validation (default: 200)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay_epoch', default=-1, type=int,
                    help='Reduces the learning rate every decay_epoch')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')

# Architecture
parser.add_argument('--num_classes', type=int, default=10,
                    help='number of classes (default: 10)')
parser.add_argument('--gaussian_size', default=64, type=int,
                    help='gaussian size (default: 64)')
parser.add_argument('--input_size', default=784, type=int,
                    help='input size (default: 784)')

# Partition parameters
parser.add_argument('--train_proportion', default=1.0, type=float,
                    help='proportion of examples to consider for training only (default: 1.0)')

# Gumbel parameters
parser.add_argument('--init_temp', default=1.0, type=float,
                    help='Initial temperature used in gumbel-softmax (recommended 0.5-1.0, default:1.0)')
parser.add_argument('--decay_temp', default=1, type=int,
                    help='Set 1 to decay gumbel temperature at every epoch (default: 1)')
parser.add_argument('--hard_gumbel', default=0, type=int,
                    help='Set 1 to use the hard version of gumbel-softmax (default: 1)')
parser.add_argument('--min_temp', default=0.5, type=float,
                    help='Minimum temperature of gumbel-softmax after annealing (default: 0.5)')
parser.add_argument('--decay_temp_rate', default=0.013862944, type=float,
                    help='Temperature decay rate at every epoch (default: 0.013862944)')

# Loss function parameters
parser.add_argument('--w_gauss', default=1, type=float,
                    help='weight of gaussian loss (default: 1)')
parser.add_argument('--w_categ', default=1, type=float,
                    help='weight of categorical loss (default: 1)')
parser.add_argument('--w_rec', default=1, type=float,
                    help='weight of reconstruction loss (default: 1)')
parser.add_argument('--w_balance1', default=0.01, type=float,
                    help='weight of imbalance loss-Y (default: 0.01)')
parser.add_argument('--w_balance2', default=0.01, type=float,
                    help='weight of imbalance loss-mutual (default: 0.01)')
parser.add_argument('--if_balance', default=0, type=int,
                    help='if use weight of imbalance loss ')
parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                    default='bce', help='desired reconstruction loss function (default: bce)')

# Others
parser.add_argument('--verbose', default=0, type=int,
                    help='print extra information at every epoch.(default: 0)')
parser.add_argument('--random_search_it', type=int, default=20,
                    help='iterations of random search (default: 20)')

# Causal Dataset
parser.add_argument('--root', type=str, default='/home/botong/Dataset/')
parser.add_argument('--dataset', type=str, default='AD')
parser.add_argument('--dataset_type', type=str, default='test')
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--data_process', type=str, default='none')
parser.add_argument('--in_channel', type=int, default=1)
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--crop_size', type=int, default=48)
parser.add_argument('--transpose', type=int, default=1)
parser.add_argument('--flip', type=int, default=1)
parser.add_argument('--shift', type=int, default=5)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--decoder_type', type=int, default=0)
parser.add_argument('--worker', type=int, default=2)
parser.add_argument('--mse_loss', type=int, default=0)

# parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--zs_dim', type=int, default=256)
parser.add_argument('--u_dim', type=int, default=4)
parser.add_argument('--us_dim', type=int, default=8)
parser.add_argument('--is_use_u', type=int, default=1)
parser.add_argument('--fix_mu', type=int, default=0)
parser.add_argument('--fix_var', type=int, default=0)
parser.add_argument('--KLD_type', type=int, default=1)
parser.add_argument('--env_num', type=int, default=8)
parser.add_argument('--is_sample', type=int, default=0)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--test_ep', type=int, default=50)
parser.add_argument('--eval_train', type=int, default=0)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--lambd', type=float, default=0.1)
parser.add_argument('--z_ratio', type=float, default=0.5)


parser.add_argument('--xlsx_name', type=str, default="CMNIST_unsp_cluster.xls")


args = parser.parse_args()

if args.cuda == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

# Random Seed
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if args.cuda:
    torch.cuda.manual_seed(SEED)

#########################################################
# Read Data
#########################################################
# if args.dataset == "mnist":
#   print("Loading mnist dataset...")
#   # Download or load downloaded MNIST dataset
  # train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
#   test_dataset = datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor())
if 'mnist' in args.dataset:
    train_loader = DataLoader(get_dataset_2D(root=args.root, args=args, fold='traintest',
                                             transform=transforms.ToTensor()
                                             ),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(get_dataset_2D(root=args.root, args=args, fold='test',
                                            transform=transforms.ToTensor()
                                            ),
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)
    train_test_loader = DataLoader(get_dataset_2D(root=args.root, args=args, fold='train',
                                                  transform=transforms.ToTensor()
                                                  ),
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=True)
    val_loader = None
elif 'NICO' in args.dataset:
    train_loader = DataLoader(get_dataset_2D(root=args.root, args=args, fold='traintest',
                                             transform=transforms.Compose([
                                                 transforms.RandomResizedCrop(
                                                     (256, 256)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             ])
                                             ),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(get_dataset_2D(root=args.root, args=args, fold='test',
                                            transform=transforms.Compose([
                                                transforms.Resize((256, 256)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])
                                            ),
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)
    train_test_loader = DataLoader(get_dataset_2D(root=args.root, args=args, fold='train',
                                                  transform=transforms.Compose([
                                                      transforms.Resize(
                                                          (256, 256)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(
                                                          (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                  ])
                                                  ),
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=True)
    val_loader = None
elif 'AD' in args.dataset:
    # if args.dataset == 'AD':
    train_loader = DataLoader(get_dataset(root=args.root, args=args, fold='traintest',
                                          transform=None, aug=0, select_env=-1),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(get_dataset(root=args.root, args=args, fold='test',
                                         transform=None, aug=0, select_env=-1),
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)
    train_test_loader = DataLoader(get_dataset(root=args.root, args=args, fold='train',
                                               transform=None, aug=0, select_env=-1),
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=True)
    val_loader = None


#########################################################
# Data Partition
#########################################################
def partition_dataset(n, proportion=0.8):
    train_num = int(n * proportion)
    indices = np.random.permutation(n)
    train_indices, val_indices = indices[:train_num], indices[train_num:]
    return train_indices, val_indices

# if args.train_proportion == 1.0:
#   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
#   val_loader = test_loader
# else:
#   train_indices, val_indices = partition_dataset(len(train_dataset), args.train_proportion)
#   # Create data loaders for train, validation and test datasets
#   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_indices))
#   val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_val, sampler=SubsetRandomSampler(val_indices))
#   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)


# Calculate flatten size of each input data
# print(train_loader)
if 'mnist' in args.dataset:
    args.input_size = (3*28**2)  # args.batch_size *
elif 'NICO' in args.dataset:
    args.input_size = (3*256**2)  # args.batch_size *
elif 'AD' in args.dataset:
    args.input_size = (48*48*48)  # args.batch_size *

print(args.input_size)
#########################################################
# Train and Test Model
#########################################################
# for epoch in range(1, args.num_epochs + 1):
#   if 'mnist' in args.dataset or  'NICO' in args.dataset:
#       for (data, labels, env, u) in data_loader:
#           pass

#   elif 'AD' in self.args.dataset:
#       for (data, u, us, labels, env) in data_loader:
#           pass

gmvae = GMVAE(args)

# Training Phase
history_loss = gmvae.train(train_loader, val_loader)

# Testing Phase

print("Testing phase...")
root = "./"

test_result = gmvae.test(test_loader)
save_results_as_xlsx(root, "test_"+args.xlsx_name, test_result, args)

train_result = gmvae.test(train_test_loader)
save_results_as_xlsx(root, "train_"+args.xlsx_name, train_result, args)

# print("Accuracy: %.5lf, NMI: %.5lf" % (accuracy, nmi))
