import pickle
import numpy as np
import os
import torch
from PIL import Image

picklefile = open('/root/Github/temp_project/VisionPictyresProject/Clas_results/alidata1/JPG800SIZE-95_cn_2_ep_120_wd_0_eye_R_center_Maculae/log.pkl', 'rb')
data = pickle.load(picklefile, encoding='iso-8859-1')
print(len(data))
print(len(data[1]))

val_acc_result = 0
best_val_acc_epoch = 0
for i in range(1, len(data)+1):
    if data[i]['val_results']['acc'].item() > val_acc_result:
        val_acc_result = data[i]['val_results']['acc']
        best_val_acc_epoch = i

print('the best val acc epoch is :', best_val_acc_epoch)
print('in best val epoch, the val acc is :', data[best_val_acc_epoch]['val_results']['acc'])
print('in best val epoch, the test acc is :', data[best_val_acc_epoch]['test_results']['acc'])

val_auc_result = 0
best_val_auc_epoch = 0
for i in range(1, len(data)+1):
    if data[i]['val_results']['auc'].item() > val_auc_result:
        val_auc_result = data[i]['val_results']['auc']
        best_val_auc_epoch = i

print('the best val auc epoch is :', best_val_auc_epoch)
print('in best val epoch, the val auc is :', data[best_val_auc_epoch]['val_results']['auc'])
print('in best val epoch, the test auc is :', data[best_val_auc_epoch]['test_results']['auc'])




