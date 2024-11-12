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
                    default='/data/ruxin2/VisionPicturesProject-04-17/VisionPicturesProject/Classification/results/Ecur_11-03/order1-ours-all-01/2020-11-03-22-08-52_Mixed_0.000200_60.000000_size_256_ep_120_0_R_Maculae/results.pkl'
                    , help='learning rate (default: 0.0001)')
parser.add_argument('--order', type=int, default= 1, help='learning rate (default: 0.0001)')
parser.add_argument('--save_dir', type=str, default= '/home', help='')

args = parser.parse_args()
logger = get_logger(args)


load_dir = args.load_dir
print('-'*20)
print('order%s'%(args.order))
print('-'*20)
picklefile = open(load_dir, 'rb')
data = pickle.load(picklefile)# 'iso-8859-1')
print(len(data))
#print(data)
data[28]['test_results_list'][0]['AUC_average_all']# data[ep][test\trsin\val][grade][acc\auc]
print(len(data[1]))
#print(len(data[1]['test_results']['AUC_generate']))

best_val_acc_all = 0
best_val_AUC_all = 0
best_mean_val_AUC = 0
best_mean_val_acc = 0

val_acc_result_0 = 0
best_val_AUC_epoch = 0
best_val_acc_epoch = 0
for i in range(1, len(data)+1):
    mean_val_acc = 0
    for j in range(6 - args.order):
        mean_val_acc += data[i]['val_results_list'][j]['acc_average_all']
    mean_val_acc /= (6 - args.order)
    if mean_val_acc > best_mean_val_acc:
        best_mean_val_acc = copy.deepcopy(mean_val_acc)
        best_val_acc_epoch = i


for i in range(1, len(data)+1):
    mean_val_auc = 0
    for j in range(6 - args.order):
        mean_val_auc += data[i]['val_results_list'][j]['AUC_average_all']
    mean_val_auc /= (6 - args.order)
    if mean_val_auc > best_mean_val_AUC:
        best_mean_val_AUC = copy.deepcopy(mean_val_auc)
        best_val_AUC_epoch = i

print('best val AUC epoch is {}'.format(best_val_AUC_epoch))
#print('the best val AUC is {:.4f}, at epoch {}, the test AUC is {:.4f}'.format(best_val_AUC_generate, best_val_AUC_epoch_generate,\
                                                                       #data[best_val_AUC_epoch_generate][test_result]['AUC_generate']))
full_results = data
test_results_list = data[best_val_acc_epoch]['test_results_list']
test_acc_mean = 0.0
logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (best_val_acc_epoch, best_mean_val_acc))
for ss in range(len(test_results_list)):
    test_acc_mean = test_acc_mean + full_results[best_val_acc_epoch]['test_results_list'][ss][
        'acc_average_all']
    logger.info('test_acc at grade %d: %0.4f' % (
        ss, full_results[best_val_acc_epoch]['test_results_list'][ss]['acc_average_all']))
logger.info('mean test acc: %0.4f' % (test_acc_mean / len(test_results_list)))

test_results_list = data[best_val_AUC_epoch]['test_results_list']
test_auc_mean = 0.0
logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (best_val_AUC_epoch, best_mean_val_AUC))
for ss in range(len(test_results_list)):
    test_auc_mean = test_auc_mean + full_results[best_val_AUC_epoch]['test_results_list'][ss][
        'AUC_average_all']
    logger.info('test_auc at grade %d: %0.4f' % (
        ss, full_results[best_val_AUC_epoch]['test_results_list'][ss]['AUC_average_all']))
logger.info('mean test auc: %0.4f' % (test_auc_mean / len(test_results_list)))


###################
