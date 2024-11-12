# -*- coding: utf-8 -*-
import pickle
import numpy as np
import os
import torch
from PIL import Image
import copy
import xlrd
import xlwt
import os
from xlutils.copy import copy as copy_

import random

import sys

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
parser.add_argument('--save_dir', type=str, default= './', help='')
parser.add_argument('--load_col', type=int, default= 0, help='')

parser.add_argument('--write_col', type=int, default= 0, help='')

args = parser.parse_args()
logger = get_logger(args)

###
####enumerate the best AUC_average_all epoch, then print all the information in excel
###

workbook1 = xlrd.open_workbook(args.load_dir)
# get all the sheets
#print('process ', fold)
print(workbook1.sheet_names())
sheet1_name = workbook1.sheet_names()[0]
# get the sheet content by using sheet index
sheet1 = workbook1.sheet_by_index(0)  # sheet index begins from [0]
# sheet1 = workbook1.sheet_by_name('Sheet1')
print('info of sheet1 :', sheet1.name, sheet1.nrows, sheet1.ncols)

for ii in range(sheet1.ncols):
    if sheet1.row_values(0)[ii] == 'results_root':
        args.load_col = ii

pickle_load_list = sheet1.col_values(args.load_col)

for ii in range(1, len(pickle_load_list)):
    print('The ',ii, 'th file')
    #########enumerate every file in loop

    load_dir = pickle_load_list[ii] + 'results.pkl'
    print('-'*20)
    print('order%s'%(args.order))
    print('-'*20)
    picklefile = open(load_dir, 'rb')
    data = pickle.load(picklefile)# 'iso-8859-1')
    print('len(data): ',len(data))
    #print(data)
    #data[28]['test_results_list'][0]['AUC_average_all']# data[ep][test\trsin\val][grade][acc\auc]
    print('len(data[1]): ',len(data[1]))
    print("len(data[1]['test_results_list']): ",len(data[1]['test_results_list']))

    ws_path = args.load_dir[:-4] + '_tongji' + args.load_dir[-4:]
    save_dir = args.load_dir[:-4] + '_tongji' + args.load_dir[-4:]
    rb = xlrd.open_workbook(ws_path)  # , formatting_info=True)
    wb = copy_(rb)
    ws = wb.get_sheet(0)


    best_val_acc_all = 0
    best_val_AUC_all = 0
    best_mean_val_AUC = 0
    best_mean_val_acc = 0

    val_acc_result_0 = 0
    best_val_AUC_epoch = 0
    best_val_acc_epoch = 0
    for i in range(1, len(data)+1):
        mean_val_acc = 0
        for j in range(len(data[1]['val_results_list'])):
            mean_val_acc += data[i]['val_results_list'][j]['acc_average_all']
        mean_val_acc /= (len(data[1]['val_results_list']))
        if mean_val_acc > best_mean_val_acc:
            best_mean_val_acc = copy.deepcopy(mean_val_acc)
            best_val_acc_epoch = i



    for i in range(1, len(data)+1):
        mean_val_auc = 0
        for j in range(len(data[1]['val_results_list'])):
            mean_val_auc += data[i]['val_results_list'][j]['AUC_average_all']
        mean_val_auc /= (len(data[1]['val_results_list']))
        if mean_val_auc > best_mean_val_AUC:
            best_mean_val_AUC = copy.deepcopy(mean_val_auc)
            best_val_AUC_epoch = i

    print('best val AUC epoch is {}'.format(best_val_AUC_epoch))
    #print('the best val AUC is {:.4f}, at epoch {}, the test AUC is {:.4f}'.format(best_val_AUC_generate, best_val_AUC_epoch_generate,\

                                                                           #data[best_val_AUC_epoch_generate][test_result]['AUC_generate']))
    full_results = data
    test_results_list = data[best_val_AUC_epoch]['test_results_list']
    test_acc_mean = 0.0
    logger.info('best val acc epoch %d, val acc: %0.4f test acc:' % (best_val_AUC_epoch, best_mean_val_acc))
    for ss in range(len(test_results_list)):
        ws.write(ss+1, sheet1.ncols+1+ii*2,
                 full_results[best_val_AUC_epoch]['test_results_list'][ss]['acc_average_all'])


        test_acc_mean = test_acc_mean + full_results[best_val_AUC_epoch]['test_results_list'][ss][
            'acc_average_all']
        logger.info('test_acc at grade %d: %0.4f' % (
            ss, full_results[best_val_AUC_epoch]['test_results_list'][ss]['acc_average_all']))
    logger.info('mean test acc: %0.4f' % (test_acc_mean / len(test_results_list)))
    ws.write(len(test_results_list)+1, sheet1.ncols+1+ii*2, test_acc_mean / len(test_results_list))

    test_results_list = data[best_val_AUC_epoch]['test_results_list']
    test_auc_mean = 0.0
    logger.info('best val auc epoch %d, val auc: %0.4f, test auc:' % (best_val_AUC_epoch, best_mean_val_AUC))
    for ss in range(len(test_results_list)):
        ws.write(ss+1, sheet1.ncols+2+ii*2,
                 full_results[best_val_AUC_epoch]['test_results_list'][ss][
                     'AUC_average_all'])

        test_auc_mean = test_auc_mean + full_results[best_val_AUC_epoch]['test_results_list'][ss][
            'AUC_average_all']
        logger.info('test_auc at grade %d: %0.4f' % (
            ss, full_results[best_val_AUC_epoch]['test_results_list'][ss]['AUC_average_all']))
    logger.info('mean test auc: %0.4f' % (test_auc_mean / len(test_results_list)))
    ws.write(len(test_results_list)+1, sheet1.ncols+2+ii*2, test_auc_mean / len(test_results_list))


    wb.save(save_dir)






###################
