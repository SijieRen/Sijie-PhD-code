import pickle
import pickle
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve,auc
import xlrd
import xlwt
import os
from xlutils.copy import copy
#from sklearn.utils.fixes import signature

save_dir = '/root/Github/temp_project/VisionPictyresProject/Clas_results/alidata1/JPG400SIZE-95_cn_2_ep_80/'
picklefile = open(os.path.join(save_dir,'log.pkl'), 'rb')
data = pickle.load(picklefile, encoding='iso-8859-1')

test = []
test = data[6]['test_results']['acc']


#print(data)
#print(test)

epoch = 80 # choose the epoch we want
ROC_score = []
ROC_label = []


auc_score_beforeSM = data[epoch]['test_results']['pred'] #pred means possibility of label 1 or 0 , like [0.6, 0.5]
auc_score = torch.softmax(torch.from_numpy(auc_score_beforeSM.astype('float32')), dim=1).numpy()
auc_label = data[epoch]['test_results']['label']
#auc_score = torch.softmax(auc_score_beforeSM, dim=1)
#assume that we have get the auc_softmax
#print('auc score is :', auc_score)
#print('actual label is :', auc_label)
#print('after argmax', np.max(auc_score, axis=1))
auc_score_after_max = auc_score[:,1]
print(len(auc_score_after_max))
#make the score in sequence for ROC_Curve
for i in range(len(auc_score_after_max)):
    a = np.argmax(auc_score_after_max)
    ROC_score.append(auc_score_after_max[a])
    ROC_label.append(auc_label[a])
    auc_score_after_max[a] = 0

#print('the size of pred is :', len(data[epoch]['test_results']['pred']))
#print('the size of label is :', len(data[epoch]['test_results']['label']))

print('the size of ROC score is :', len(ROC_score))
print('the size of ROC label is :', len(ROC_label))

#print('auc_score_beforeSM is :', auc_score_beforeSM)
print('ROC_score is :', ROC_score)
print('ROC_label IS :', ROC_label)
#todo insert the code of drawing ROC and PR curve

#todo plot P-R Curve
for i in range(1):
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    y_true = ROC_label
    y_scores = ROC_score
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.plot(recall,precision)
    #plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(save_dir,'PR_Curve.jpg'))

#todo plot ROC-Curve
for i in range(1):
    plt.figure("ROC Curve")
    y_true = ROC_label
    y_scores = ROC_score
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(save_dir,'ROC_Curve.jpg'))


pred_score = torch.softmax(torch.from_numpy(data[epoch]['test_results']['pred'].astype('float32')), dim=1).numpy()
bad_sample_index = []
bad_sample_name = []
bad_0_to_1_sample = [] #0 to 1 means, it should be 0, but judged to 1
bad_0_to_1_score = []
bad_1_to_0_sample = [] #1 to 0 means, it should be 1, but judged to 0
bad_1_to_0_score = []
# choose the best_val_epoch
for i in range(len(data[epoch]['test_results']['pred_result'])):
    # if pred_result not equal target, record it down
    if not data[epoch]['test_results']['pred_result'][i].item() == data[epoch]['test_results']['label'][i]:
        # bad_sample_name.append(image path/numpy)
        if data[epoch]['test_results']['label'][i] == 0:
            bad_0_to_1_sample.append(data[epoch]['test_results']['data_path'][i])  # which means it is 0. but judged to 1
            bad_0_to_1_score.append(pred_score[i][1])
        else:
            bad_1_to_0_sample.append(data[epoch]['test_results']['data_path'][i]) #wich means it is 1, but judged to 0
            bad_1_to_0_score.append(pred_score[i][0])

print('bad_0_to_1_score is :', bad_0_to_1_score)
print('bad_0_to_1_sample is :',bad_0_to_1_sample[3] )

print('bad_1_to_0_score is :', bad_1_to_0_score)
print('bad_1_to_0_sample is :',bad_1_to_0_sample[3] )


bad_0_to_1_sample_in_sequence = []
bad_0_to_1_score_in_sequence = []
for i in range(len(bad_0_to_1_score)):
    a = np.argmax(bad_0_to_1_score)
    bad_0_to_1_score_in_sequence.append(bad_0_to_1_score[a])
    bad_0_to_1_sample_in_sequence.append(bad_0_to_1_sample[a])
    bad_0_to_1_score[a] = 0

bad_1_to_0_sample_in_sequence = []
bad_1_to_0_score_in_sequence = []
for i in range(len(bad_1_to_0_score)):
    a = np.argmax(bad_1_to_0_score)
    bad_1_to_0_score_in_sequence.append(bad_1_to_0_score[a])
    bad_1_to_0_sample_in_sequence.append(bad_1_to_0_sample[a])
    bad_1_to_0_score[a] = 0

print('bad 0 to 1 possibility is :', bad_0_to_1_score_in_sequence)
print('bad 1 to 0 possibility is :', bad_1_to_0_score_in_sequence)
print(bad_1_to_0_sample_in_sequence[3])

#todo insert the code of open the bad sample pictures
i = 3
image1 = Image.open(bad_1_to_0_sample_in_sequence[i])
plt.figure("dog")
plt.imshow(image1)
plt.show()
print('the possibility is : {}, the label of this should be : {}'.format(bad_1_to_0_score_in_sequence[i], 1))


j = 2
image2 = Image.open(bad_0_to_1_sample_in_sequence[j])
image2.show()
print('the possibility is : {}, the label of this should be : {}'.format(bad_0_to_1_score_in_sequence[j], 0))



def writeExcel():
    # add an empty excel as oringinal one
    rb = xlrd.open_workbook("/alidata1/RA_Label_Index/Empty.xlsx")   # make an empty excel for writing
    wb = copy(rb)
    ws = wb.get_sheet(0)
    ws.write(0, 0, 'Bad_0_to_1_score')
    ws.write(0, 1, 'Bad_0_to_1_sample')
    ws.write(0, 2, 'Bad_1_to_0_score')
    ws.write(0, 3, 'Bad_1_to_0_sample')

    for i in range(len(bad_0_to_1_score_in_sequence)):
        ws.write(i+1, 0, bad_0_to_1_score_in_sequence[i].item())
        ws.write(i+1, 1, bad_0_to_1_sample_in_sequence[i])
        save_dir_0_to_1 = save_dir + 'bad_0_to_1/'
        if not os.path.exists(save_dir_0_to_1):
            os.makedirs(save_dir_0_to_1)
        img = Image.open(bad_0_to_1_sample_in_sequence[i])
        str_1 = str(bad_0_to_1_score_in_sequence[i].item())
        str_2 = str(bad_0_to_1_sample_in_sequence[i]).replace('/', '_')[24:]
        str_3 = str_1 + '_' + str_2
        print(str_3)
        img.save(os.path.join(save_dir_0_to_1, str_3))


    for i in range(len(bad_1_to_0_score_in_sequence)):
        ws.write(i+1, 2, bad_1_to_0_score_in_sequence[i].item())
        ws.write(i+1, 3, bad_1_to_0_sample_in_sequence[i])
        save_dir_1_to_0 = save_dir + 'bad_1_to_0/'
        if not os.path.exists(save_dir_1_to_0):
            os.makedirs(save_dir_1_to_0)
        img = Image.open(bad_1_to_0_sample_in_sequence[i])
        str_1 = str(bad_1_to_0_score_in_sequence[i].item())
        str_2 = str(bad_1_to_0_sample_in_sequence[i]).replace('/', '_')[24:]
        str_3 = str_1 + '_' + str_2
        print(str_3)
        img.save(os.path.join(save_dir_1_to_0, str_3))


    # TO add _Optim to the end of the saving files name
    wb.save(os.path.join(save_dir, 'Bad_sample_index.xls'))
#writeExcel()


'''
    for i in range(0, len(file_name_list)):

        # print('inter i:', i)
        for j in range(1, len(col_1_0)):
            # print('inter j:', j)
            if str(col_1_0[j]) in str(file_name_list[i]):

                # writeExcel(i, 1, str(col_1_8[j]))
                ws.write(row, 0, str(file_name_list[i]))
                ws.write(row, 1, str(col_1_8[j]))
                print('write suss: ', str(col_1_8[j]))


                if col_1_8[j] <= -0.5:
                    ws.write(row, 2, 1)
                else:
                    ws.write(row, 2, 0)

                ran_num = random.randint(1, 10000)

                if ran_num >= 9000:
                    ws.write(row, 3, 'test')
                elif ran_num >= 8000:
                    ws.write(row, 3, 'val')
                else:
                    ws.write(row, 3, 'train')

                row += 1
'''

