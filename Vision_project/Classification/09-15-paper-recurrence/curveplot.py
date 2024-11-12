import pickle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
parser.add_argument('--order', type=int, default=1, help='learning rate (default: 0.0001)')
parser.add_argument('--save_dir', type=str, default='/home', help='')

args = parser.parse_args()
# logger = get_logger(args)

picklefile = open(args.load_dir, 'rb')
data = pickle.load(picklefile, encoding='iso-8859-1')

loss_train_D = []
loss_train_G = []
loss_train_C = []
loss_train_R = []
loss_test = []
loss_val = []

acc_train = []
auc_test = []
auc_val = []
ep = []

print(len(data))
#print(data[1]['train_results']['loss_D'])
for i in range(1, len(data) + 1):
    # lossM.append(data[i]['train_results']['loss_M'])
    # lossM1.append(data[i]['train_results']['loss_M1'])
    loss_train_D.append(data[i]['train_results']['loss_M_generate'])
    #loss_train_G.append(data[i]['train_results']['loss_G'])
    #loss_train_C.append(data[i]['train_results']['loss_C'])
    # loss_train_R.append(data[i]['train_results']['loss_R'])

    auc_test.append(data[i]['test_results_list'][2]['AUC_average_all'])
    loss_test.append(data[i]['val_results_list'][2]['loss'])

    ep.append(i)
# print(data[2]['train_results']['loss_M'])
# print(lossM[0:-1:2])
# print(ep)
fig = plt.figure(figsize=(10, 5))  # 设置图大小 figsize=(6,3)
# plt.plot(ep[:],lossM1[:],  c='red',label = 'lossM1')
plt.plot(ep[:], loss_train_D[:], c='green', label='loss_M_generate')
#plt.plot(ep[:], loss_train_G[:], c='red', label='loss_trian_G')
#plt.plot(ep[:], loss_train_C[:], c='blue', label='loss_trian_C')
# plt.plot(ep[:], loss_train_R[:], c='yellow', label='loss_trian_R')
print(len(ep))
print(len(auc_test))
# plt.plot(ep[:], acc_train[:], c='yellow', label='auc_trian')
plt.plot(ep[:], auc_test[:], c='black', label='auc_test')
plt.plot(ep[:], loss_test[:], c='maroon', label='loss_test')

# plt.plot(ep[:],loss_U1[:],  c='maroon',label = 'lossU1')
# plt.plot(ep[:],loss_M_generate[:],  c='yellow',label = 'loss_M_generate')
# plt.plot(ep[:],loss_M_minus[:],  c='blue',label = 'loss_M_minus')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('loss-auc-curve')
plt.legend(loc='upper right')
plt.savefig('./loss-acc-curve-2.png')
plt.show()
print('Finish ploting!')

