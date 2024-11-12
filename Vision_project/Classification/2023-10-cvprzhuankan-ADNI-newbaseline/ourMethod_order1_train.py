# coding=utf-8
import argparse
import os
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import ExtractorData, Order1Data
from model import *
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter


# clip_gradient(optimizer, opt.clip)

def train(opt):
    
    loss_D_writer = SummaryWriter(log_dir="summary_loss_D")
    loss_G_writer = SummaryWriter(log_dir="summary_loss_G")
    loss_res_writer = SummaryWriter(log_dir="summary_loss_res")
    loss_gen_writer = SummaryWriter(log_dir="summary_loss_gen")

    # Datasets
    dataset_train = Order1Data(data_index='./data/dataset_order%s.xls' % '1',
                               data_root=opt.data_root, mode='train')
    dataset_test = Order1Data(data_index='./data/dataset_order%s.xls' % '1',
                              data_root=opt.data_root, mode='test')

    dataloader_train = DataLoader(dataset_train, batch_size=opt.batchsize, shuffle=True,
                                  num_workers=8)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batchsize, shuffle=True,
                                 num_workers=8)

    # Model
    ## extractor
    extractor = RN18_extrator().cuda()
    extractor.eval()
    extractor.load_state_dict(torch.load(opt.extractor_path)['extractor_state_dict'])
    ## generator
    G_net = Generator_LSTM_1(in_channel=64).cuda()
    ## discriminator
    D_net = Discriminator(in_channels=64).cuda()
    ## classifer
    C_net = RN18_Classifer().cuda()
    C_Res_net = RN18_Res_Classifer().cuda()

    # optimizer
    optimizer_C_Res = torch.optim.SGD(params=C_Res_net.parameters(),
                                      lr=opt.lr2,
                                      weight_decay=opt.wd2,
                                      momentum=opt.momentum)
    optimizer_C = torch.optim.SGD(params=C_net.parameters(),
                                  lr=opt.lr2,
                                  weight_decay=opt.wd2,
                                  momentum=opt.momentum)
    scheduler_C = torch.optim.lr_scheduler.StepLR(optimizer_C,step_size=23000,gamma=0.5)
    scheduler_C_Res = torch.optim.lr_scheduler.StepLR(optimizer_C_Res,step_size=23000,gamma=0.5)

    optimizer_G = torch.optim.RMSprop(params=G_net.parameters(), lr=opt.lr, weight_decay=opt.wd)
    optimizer_D = torch.optim.RMSprop(params=D_net.parameters(), lr=opt.lr, weight_decay=opt.wd)

    # train
    if not os.path.exists(opt.savepath):
        os.makedirs(opt.savepath)
    global_step = 0
    best_auc = 0
    best_acc = 0

    for epoch in range(int(opt.epoch)):

        C_net.train()
        C_Res_net.train()

        total_step = len(dataloader_train)

        data_size = 128
        for step, (data_3D_t, data_3D_T, label, grad1, grad2) in enumerate(dataloader_train):
            batch_size = data_3D_t.size(0)
            data_3D_t, data_3D_T, label, grad1, grad2 = data_3D_t.cuda().float(), \
                                                        data_3D_t.cuda().float(), \
                                                        label.cuda().float(), \
                                                        grad1.cuda().float(), \
                                                        grad2.cuda().float()

            data_3D_t = F.interpolate(data_3D_t, size=(data_size, data_size, data_size), mode='trilinear')
            data_3D_T = F.interpolate(data_3D_T, size=(data_size, data_size, data_size), mode='trilinear')

            featuremap_t = extractor(data_3D_t)
            featuremap_T = extractor(data_3D_T)

            # generate feature
            z = torch.randn(batch_size, 100, 1, 1, 1).cuda()
            generated_feature = G_net(featuremap_t, z, grad1, grad2)
            # compute res feature
            res_feature = generated_feature - featuremap_t

            if epoch % opt.D_epoch != 0:
                for p in extractor.parameters():
                    p.requires_grad = False
                for p in C_Res_net.parameters():
                    p.requires_grad = True
                for p in G_net.parameters():
                    p.requires_grad = False
                for p in D_net.parameters():  # 训练四次D
                    p.requires_grad = True
                for p in C_net.parameters():
                    p.requires_grad = True
                D_net.train()
                G_net.eval()
                real_loss = D_net(featuremap_T).mean(0).view(1)
                fake_loss = D_net(generated_feature).mean(0).view(1)
                loss_D = real_loss - fake_loss
                
                loss_D_writer.add_scalar("loss_D", loss.detach(), epoch)
                
                optimizer_D.zero_grad()
                for p in D_net.parameters():
                    p.data.clamp_(-opt.wcl, opt.wcl)
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
            else:
                for p in extractor.parameters():
                    p.requires_grad = False
                for p in C_Res_net.parameters():
                    p.requires_grad = True
                for p in G_net.parameters():
                    p.requires_grad = True
                for p in D_net.parameters():  # 训练四次D
                    p.requires_grad = False
                for p in C_net.parameters():
                    p.requires_grad = True
                D_net.eval()
                G_net.train()

            # pred
            pred_t = torch.softmax(C_net(featuremap_t), dim=1)
            pred_T = torch.softmax(C_net(featuremap_T), dim=1)
            pred_res = torch.softmax(C_Res_net(res_feature), dim=1)
            pred_gen = torch.softmax(C_net(generated_feature), dim=1)

            pred1 = pred_t[::, 1] + pred_t[::, 0] * pred_res[::, 1]
            pred2 = pred_t[::, 2] + (1 - pred_t[::, 2]) * pred_res[::, 2]
            pred = torch.cat([(1 - pred1 - pred2).unsqueeze(1), pred1.unsqueeze(1), pred2.unsqueeze(1)], dim=1)

            # loss
            loss_res = F.cross_entropy(pred, label)
            loss_gen = F.cross_entropy(pred_gen, label)
            ## regularizer
            P_t = F.one_hot(torch.argmax(pred_t, dim=1), num_classes=3)
            P_T = F.one_hot(torch.argmax(pred_t, dim=1), num_classes=3)
            tmp = torch.where(P_t == P_T, pred_t - pred_T + opt.margin, torch.zeros_like(P_t).float().cuda())

            reg_mono = torch.sum(torch.clamp(tmp, 0)) / batch_size
            ## all loss and backward
            if epoch % opt.D_epoch != 0:
                # train D
                # loss
                loss = opt.lw_lambda * loss_res + opt.lw_alpha * reg_mono + opt.lw_gamma * loss_gen
                loss_gen_writer.add_scalar("loss_gen", loss_gen.detach(), epoch)
                loss_res_writer.add_scalar("loss_res", loss_res.detach(), epoch)
                # backward
                optimizer_C.zero_grad()
                optimizer_C_Res.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_C.step()
                optimizer_C_Res.step()
                scheduler_C.step()
                scheduler_C_Res.step()
                
            else:
                # train G
                # loss
                loss_G = D_net(generated_feature).mean(0).view(1)
                loss = opt.lw_lambda * loss_res + opt.lw_alpha * reg_mono + opt.lw_gamma * loss_gen
                
                loss_gen_writer.add_scalar("loss_gen", loss_gen.detach(), epoch)
                loss_res_writer.add_scalar("loss_res", loss_res.detach(), epoch)
                
                loss_G_mse = F.mse_loss(generated_feature, featuremap_T)
                
                loss_G_writer.add_scalar("loss_G", loss_G.detach()+loss_G_mse.detach(), epoch)
                    
                loss = loss_G + loss + loss_G_mse
                # backward
                optimizer_C.zero_grad()
                optimizer_C_Res.zero_grad()
                optimizer_G.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_G.step()
                optimizer_C.step()
                optimizer_C_Res.step()
                scheduler_C.step()
                scheduler_C_Res.step()

            global_step += 1
            if step % 10 == 0 or step == total_step - 1:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f ' %
                      (datetime.datetime.now(), global_step, epoch + 1, opt.epoch,
                       optimizer_C.param_groups[0]['lr'], loss.item()))

        # eval
        G_net.eval()
        C_Res_net.eval()
        C_net.eval()
        D_net.eval()
        pred_all = []
        pred_gen_all = []
        pred_res_all = []
        pred_average_all = []
        label_all = []
        for step, (data_3D_t, data_3D_T, label, grad1, grad2) in enumerate(dataloader_test):
            batch_size = data_3D_t.size(0)
            data_3D_t, data_3D_T, label, grad1, grad2 = data_3D_t.cuda().float(), \
                                                        data_3D_t.cuda().float(), \
                                                        label.cuda().float(), \
                                                        grad1.cuda().float(), \
                                                        grad2.cuda().float()

            data_3D_t = F.interpolate(data_3D_t, size=(data_size, data_size, data_size), mode='trilinear')

            # inference
            featuremap_t = extractor(data_3D_t)
            ## generate feature
            z = torch.randn(batch_size, 100, 1, 1, 1).cuda()
            generated_feature = G_net(featuremap_t, z, grad1, grad2)
            ## res
            res_feature = generated_feature - featuremap_t
            ## pred
            pred_t = torch.softmax(C_net(featuremap_t), dim=1)
            pred_res = torch.softmax(C_Res_net(res_feature), dim=1)
            pred_gen = torch.softmax(C_net(generated_feature), dim=1)
            pred1 = pred_t[::, 1] + pred_t[::, 0] * pred_res[::, 1]
            pred2 = pred_t[::, 2] + (1 - pred_t[::, 2]) * pred_res[::, 2]
            pred = torch.cat([(1 - pred1 - pred2).unsqueeze(1), pred1.unsqueeze(1), pred2.unsqueeze(1)], dim=1)
            pred_average = (pred_gen + pred) / 2

            pred_res = torch.softmax(pred_res, dim=1).detach().cpu().numpy()
            pred_res_all.extend(pred_res)

            pred = torch.softmax(pred, dim=1).detach().cpu().numpy()
            pred_all.extend(pred)

            pred_gen = torch.softmax(pred_gen, dim=1).detach().cpu().numpy()
            pred_gen_all.extend(pred_gen)

            pred_average = torch.softmax(pred_average, dim=1).detach().cpu().numpy()
            pred_average_all.extend(pred_average)

            label = label.cpu().numpy()
            label_all.extend(label)

        auc = roc_auc_score(label_all, pred_all, multi_class='ovr')
        auc_res = roc_auc_score(label_all, pred_res_all, multi_class='ovr')
        auc_gen = roc_auc_score(label_all, pred_gen_all, multi_class='ovr')
        auc_average = roc_auc_score(label_all, pred_average_all, multi_class='ovr')
        if auc > best_auc:
            states = {
                'metric': auc,
                'C_Net_dict': C_net.state_dict(),
                'D_Net_dict': D_net.state_dict(),
                'G_Net_dict': G_net.state_dict(),
                'C_Res_Net_dict': C_Res_net.state_dict(),
            }
            torch.save(states, os.path.join(opt.savepath,
                                            'model-ourMethod-order1-best.pth'))
            best_auc = auc
        
        acc = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_all, axis=1))
        acc_res = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_res_all, axis=1))
        acc_gen = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_gen_all, axis=1))
        acc_average = accuracy_score(np.argmax(label_all, axis=1), np.argmax(pred_average_all, axis=1))
        best_acc = acc if acc > best_acc else best_acc
        
        print("auc:{:.4f} | best_auc:{:.4f} | acc:{:.4f} | best_acc:{:.4f}".format(auc, best_auc, acc, best_acc))

        # if epoch > 100:
        #     states = {
        #         'C_Net_dict': C_net.state_dict(),
        #         'D_Net_dict': D_net.state_dict(),
        #         'G_Net_dict': G_net.state_dict(),
        #         'C_Res_Net_dict': C_Res_net.state_dict(),
        #     }
        #     torch.save(states, os.path.join(opt.savepath,
        #                                     'model-ourMethod-order1-%s.pth') % (str(epoch)))
    loss_D_writer.close()
    loss_G_writer.close()
    loss_res_writer.close()
    loss_gen_writer.close()


def main():
    parser = argparse.ArgumentParser()

    # Model config
    parser.add_argument('--baseline', type=str,
                        default='RN18', help='baseline option')

    # Optimizer config
    parser.add_argument('--epoch', type=int,
                        default=5e3, help='epoch number')
    parser.add_argument('--D_epoch', type=int,
                        default=6, help='epoch number for D_net')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--lr2', type=float,
                        default=1e-4, help='learning rate for classifer')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='SGD momentum rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--wd', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--wd2', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--wcl', type=float,
                        default=0.01, help='weight cliping limit for D_net')

    # Checkpoints config
    parser.add_argument('--savepath', type=str, default='./checkpoints/baseline/')
    parser.add_argument('--extractor_path', type=str,
                        default='./checkpoints/extractor/model-best-scale=0.5.pth')

    # Datasets config
    parser.add_argument('--data_root', type=str,
                        default='/root/autodl-tmp/ImageAll_V1_part1/npy_processed_data/',
                        help='path to box train dataset')

    # Hyper-parameter
    parser.add_argument('--margin', type=float,
                        default=0, help='margin h-p for reg_mono')
    parser.add_argument('--lw_gamma', type=float,
                        default=1, help='loss weight for loss_gen')
    parser.add_argument('--lw_alpha', type=float,
                        default=1, help='loss weight for reg_mono')
    parser.add_argument('--lw_lambda', type=float,
                        default=1, help='loss weight for loss_res')
    parser.add_argument('--lw_beta', type=float,
                        default=1, help='loss weight for loss_G_mse')

    opt = parser.parse_args()
    train(opt)


if __name__ == '__main__':
    main()

    # a = torch.randn([1])
    # print(a.shape)
