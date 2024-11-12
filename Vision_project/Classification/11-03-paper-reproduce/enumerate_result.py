import pickle
import numpy as np
import os
import torch
from PIL import Image

picklefile = open('/data/ruxin/10Gpu-trans-test', 'rb')
data = pickle.load(picklefile, encoding='iso-8859-1')
print(len(data))
print(data)
data[28]['test_results_list'][4]['AUC_average_all']
print(len(data[1]))
#print(len(data[1]['test_results']['AUC_generate']))

best_val_acc_generate = 0
best_val_AUC_generate = 0
best_val_acc_minus = 0
best_val_AUC_minus = 0

val_acc_result_0 = 0
best_val_acc_epoch_0 = 0
#val_result = 'val_results3'
#test_result = 'test_results3'
print('For Grade6:')
#print(data[60]['val_results']['acc_2_0'])
print('For generate result:')
for i in range(1, len(data)+1):
    if data[i][val_result]['AUC_generate'] > best_val_AUC_generate:
        best_val_AUC_generate = data[i][val_result]['AUC_generate']
        #val_loss_0 = data[i]['val_results']['AUC_generate']
        best_val_AUC_epoch_generate = i

print('the best val AUC is {:.4f}, at epoch {}, the test AUC is {:.4f}'.format(best_val_AUC_generate, best_val_AUC_epoch_generate,\
                                                                       data[best_val_AUC_epoch_generate][test_result]['AUC_generate']))

for i in range(1, len(data)+1):
    if data[i][val_result]['acc_generate'] > best_val_acc_generate:
        best_val_acc_generate = data[i][val_result]['acc_generate']
        #val_loss_0 = data[i]['val_results']['AUC_generate']
        best_val_acc_epoch_generate = i

print('the best val acc is {:.4f}, at epoch {}, the test AUC is {:.4f}'.format(best_val_acc_generate, best_val_acc_epoch_generate,\
                                                                       data[best_val_acc_epoch_generate][test_result]['acc_generate']))

print('For minus result:')
for i in range(1, len(data)+1):
    if data[i][val_result]['AUC_minus'] > best_val_AUC_minus:
        best_val_AUC_minus = data[i][val_result]['AUC_minus']
        #val_loss_0 = data[i]['val_results']['AUC_generate']
        best_val_AUC_epoch_minus = i

print('the best val AUC is {:.4f}, at epoch {}, the test AUC is {:.4f}'.format(best_val_AUC_minus, best_val_AUC_epoch_minus,\
                                                                       data[best_val_AUC_epoch_minus][test_result]['AUC_minus']))

for i in range(1, len(data)+1):
    if data[i][val_result]['acc_minus'] > best_val_acc_minus:
        best_val_acc_minus = data[i][val_result]['acc_minus']
        #val_loss_0 = data[i]['val_results']['AUC_generate']
        best_val_acc_epoch_minus = i

print('the best val acc is {:.4f}, at epoch {}, the test AUC is {:.4f}'.format(best_val_acc_minus, best_val_acc_epoch_minus,\
                                                                       data[best_val_acc_epoch_minus][test_result]['acc_minus']))
best_val_AUC_average = 0
best_val_AUC_epoch_average = 0
best_val_acc_average = 0
best_val_acc_average_epoch = 0
print('For average reasult:')
for i in range(1, len(data)+1):
    if data[i][val_result]['AUC_average'] > best_val_AUC_average:
        best_val_AUC_average = data[i][val_result]['AUC_average']
        #val_loss_0 = data[i]['val_results']['AUC_generate']
        best_val_AUC_epoch_average = i

print('the best val AUC is {:.4f}, at epoch {}, the test AUC is {:.4f}'.format(best_val_AUC_average, best_val_AUC_epoch_average,\
                                                                       data[best_val_AUC_epoch_average][test_result]['AUC_average']))

for i in range(1, len(data)+1):
    if data[i][val_result]['acc_average'] > best_val_acc_average:
        best_val_acc_average = data[i][val_result]['acc_average']
        #val_loss_0 = data[i]['val_results']['AUC_generate']
        best_val_acc_epoch_average = i

print('the best val acc is {:.4f}, at epoch {}, the test AUC is {:.4f}'.format(best_val_acc_average, best_val_acc_epoch_average,\
                                                                       data[best_val_acc_epoch_average][test_result]['acc_average']))


'''

val_acc_result_1 = 0
best_val_acc_epoch_1 = 0
#print(data[60]['val_results']['acc_2_1'])
for i in range(1, len(data)+1):
    if data[i]['val_results']['acc_2_1'] > val_acc_result_1:
        val_acc_result_1 = data[i]['val_results']['acc_2_1']
        val_loss_1 = data[i]['val_results']['loss_2']
        best_val_acc_epoch_1 = i

print('for margin 1 :')
print('the best val acc epoch is :', best_val_acc_epoch_1)
print('in best val epoch, the val acc and loss are :', data[best_val_acc_epoch_1]['val_results']['acc_2_1'], data[best_val_acc_epoch_1]['val_results']['loss_2'])
print('in best val epoch, the test acc_1 and loss are :', data[best_val_acc_epoch_1]['test_results']['acc_1'], data[best_val_acc_epoch_1]['test_results']['loss_1'])
print('in best val epoch, the test acc and loss of 2 are :', data[best_val_acc_epoch_1]['test_results']['acc_2_1'], data[best_val_acc_epoch_1]['test_results']['loss_2'])
print('in best val epoch, the test acc and loss of 3 are :', data[best_val_acc_epoch_1]['test_results']['acc_3_1'], data[best_val_acc_epoch_1]['test_results']['loss_3'])
print('in best val epoch, the test acc and loss of 4 are :', data[best_val_acc_epoch_1]['test_results']['acc_4_1'], data[best_val_acc_epoch_1]['test_results']['loss_4'])
print('in best val epoch, the test acc and loss of 5 are :', data[best_val_acc_epoch_1]['test_results']['acc_5_1'], data[best_val_acc_epoch_1]['test_results']['loss_5'])
print('in best val epoch, the test acc and loss of 6 are :', data[best_val_acc_epoch_1]['test_results']['acc_6_1'], data[best_val_acc_epoch_1]['test_results']['loss_6'])





val_acc_result_2 = 0
best_val_acc_epoch_2 = 0
#print(data[60]['val_results']['acc_2_2'])
for i in range(1, len(data)+1):
    if data[i]['val_results']['acc_2_2'] > val_acc_result_2:
        val_acc_result_2 = data[i]['val_results']['acc_2_2']
        val_loss_2 = data[i]['val_results']['loss_2']
        best_val_acc_epoch_2 = i

print('for margin 2 :')
print('the best val acc epoch is :', best_val_acc_epoch_2)
print('in best val epoch, the val acc and loss are :', data[best_val_acc_epoch_2]['val_results']['acc_2_2'], data[best_val_acc_epoch_2]['val_results']['loss_2'])
print('in best val epoch, the test acc_1 and loss are :', data[best_val_acc_epoch_2]['test_results']['acc_2'], data[best_val_acc_epoch_2]['test_results']['loss_1'])
print('in best val epoch, the test acc and loss of 2 are :', data[best_val_acc_epoch_2]['test_results']['acc_2_2'], data[best_val_acc_epoch_2]['test_results']['loss_2'])
print('in best val epoch, the test acc and loss of 3 are :', data[best_val_acc_epoch_2]['test_results']['acc_3_2'], data[best_val_acc_epoch_2]['test_results']['loss_3'])
print('in best val epoch, the test acc and loss of 4 are :', data[best_val_acc_epoch_2]['test_results']['acc_4_2'], data[best_val_acc_epoch_2]['test_results']['loss_4'])
print('in best val epoch, the test acc and loss of 5 are :', data[best_val_acc_epoch_2]['test_results']['acc_5_2'], data[best_val_acc_epoch_2]['test_results']['loss_5'])
print('in best val epoch, the test acc and loss of 6 are :', data[best_val_acc_epoch_2]['test_results']['acc_6_2'], data[best_val_acc_epoch_2]['test_results']['loss_6'])

acc_0 = 0
for i in range(1, len(data)+1):
    if data[i]['val_results']['acc_0'] > val_acc_result_2:
        val_acc_result_2 = data[i]['val_results']['acc_o']
        val_loss_2 = data[i]['val_results']['loss_1']
        best_val_acc_epoch_0 = i
print('in best val epoch, the test acc_1 and loss are :', data[best_val_acc_epoch_2]['test_results']['acc_2'], data[best_val_acc_epoch_2]['test_results']['loss_1'])

P1_ability = []
#print(data[i]['train_result']['P1'])

for i in range(1, len(data)+1):
    small_list = []
    p_9 = 0
    p_8 = 0
    p_7 = 0
    p1list = data[i]['train_results']['P1'].cpu().numpy().tolist()
    for j in range(len(p1list)):
        if p1list[j] > 0.9 :
            p_9 +=1
        if p1list[j] > 0.8 :
            p_8 +=1
        if p1list[j] > 0.7 :
            p_7 +=1
    p_9 /= len(p1list)
    small_list.append(p_9)
    p_8 /= len(p1list)
    small_list.append(p_8)
    p_7 /= len(p1list)
    small_list.append(p_7)
    P1_ability.append(small_list)

print(P1_ability[45])
print(data[10]['train_results']['P1'].cpu().numpy().tolist())
'''
'''
correct_5_6_1  = 0
correct_5_6_2  = 0

for i in range(len(data[best_val_acc_epoch_1]['test_results']['label_5'])):
    if not abs(data[best_val_acc_epoch_1]['test_results']['pred_5'][i] - data[best_val_acc_epoch_1]['test_results']['label_5'][i]) > 1:
        correct_5_6_1 += 1

print(correct_5_6_1/912)

for i in range(len(data[best_val_acc_epoch_1]['test_results']['label_5'])):
    if not abs(data[best_val_acc_epoch_2]['test_results']['pred_5'][i] - data[best_val_acc_epoch_1]['test_results']['label_5'][i]) > 2:
        correct_5_6_2 += 1

print(correct_5_6_2/912)

'''