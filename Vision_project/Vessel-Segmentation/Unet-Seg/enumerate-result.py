import pickle
import numpy as np
import os
import torch
from PIL import Image
import pickle
import numpy as np
import os
import torch
from PIL import Image
import copy

import argparse

def get_logger(opt, fold=0):
    import logging
    logger = logging.getLogger('AL')
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler('{}training_{}.log'.format(opt.save_dir, fold))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger




parser = argparse.ArgumentParser(description='our')
parser.add_argument('--load_dir', type=str,
                    default='/data/ruxin2/VisionPicturesProject-04-17/VisionPicturesProject/Vessel-Segmentation/results/Unet-Seg_12-15/12-15-021/2020-12-15-17-40-40_SGD_0.100000_60.000000_size_256_ep_200_0_R_Maculae/results.pkl'
                    , help='learning rate (default: 0.0001)')
parser.add_argument('--order', type=int, default= 1, help='learning rate (default: 0.0001)')
parser.add_argument('--save_dir', type=str, default= '/home', help='')

args = parser.parse_args()


picklefile = open(args.load_dir, 'rb')
data = pickle.load(picklefile, encoding='iso-8859-1')
print(len(data))
print(data)
data[28]['test_results']['loss']
print(len(data[1]))
#print(len(data[1]['test_results']['AUC_generate']))

best_val_loss = 1
best_test_loss = 1
best_val_loss_epoch = 0

#val_result = 'val_results3'
#test_result = 'test_results3'

#print(data[60]['val_results']['acc_2_0'])
print('For generate result:')
for i in range(1, len(data)+1):
    if data[i]['val_results']['loss'] < best_val_loss:
        best_val_loss = data[i]['val_results']['loss']
        #val_loss_0 = data[i]['val_results']['AUC_generate']
        best_val_loss_epoch = i

print('the best val loss is {:.4f}, at epoch {}, the val dice-coff is {:.4f}'\
      .format(best_val_loss, best_val_loss_epoch, 1 - best_val_loss))
print('At this epoch {:.4f}, the test loss is {:.4f}, the test dice-coff is {:.4f}'\
      .format(best_val_loss_epoch,data[best_val_loss_epoch]['test_results']['loss'], \
              1 - data[best_val_loss_epoch]['test_results']['loss']))

