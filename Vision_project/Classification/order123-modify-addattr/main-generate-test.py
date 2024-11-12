from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models import *
from utils import *
import time
import pickle
import copy
import datetime


def main():
    # Training settings
    args = get_opts()
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader_list, test_loader_list = get_all_dataloader(args)

    model_1 = RN18_front(final_tanh = args.final_tanh).cuda()
    model_2_res = RN18_last(num_classes=args.class_num).cuda()
    model_2_generate = RN18_last(num_classes=args.class_num).cuda()

    # here need to load pretrain model
    # model_1 and model_2_generate can from the same model

    if args.G_net_type == 'G_net':
        G_net = Generator(feature_num=len(args.feature_list)+1,
                          final_tanh=args.final_tanh,
                          is_ESPCN=args.is_ESPCN, scale_factor=args.scale_factor, mid_channel=args.dw_midch,
                          dw_type=args.dw_type).cuda()
        
    elif args.G_net_type == 'U_net':
        G_net = UNet(n_channels=128,
                     n_classes=128,
                     bilinear=args.bi_linear,
                     feature_num=len(args.feature_list)+1,
                     final_tanh=args.final_tanh,
                     is_ESPCN=args.is_ESPCN, scale_factor=args.scale_factor, mid_channel=args.dw_midch,
                     dw_type=args.dw_type).cuda()

    elif 'Big' in args.gantype:
        G_net = BigGenerator(gantype=args.gantype, n_class=len(args.feature_list)).cuda()
    elif 'Half' in args.gantype:
        G_net = HalfBigGenerator(gantype=args.gantype, n_class=len(args.feature_list)).cuda()
    elif 'localadjust' in args.gantype:
        G_net = BigGenerator1(gantype=args.gantype, n_class=len(args.feature_list)).cuda()

    if 'Big' in args.discritype:
        D_net = BigDiscriminator1(n_class=len(args.feature_list), SCR=args.SCR).cuda()
    else:
        D_net = Discriminator(128).cuda()
    

    load_pytorch_model(model_1, os.path.join(args.load_dir, 'best_val_auc_model_1.pt'), )
    load_pytorch_model(model_2_generate, os.path.join(args.load_dir, 'best_val_auc_model_2_generate.pt'), )
    load_pytorch_model(model_2_res, os.path.join(args.load_dir, 'best_val_auc_model_2_res.pt'), )
    load_pytorch_model(G_net, os.path.join(args.load_dir, 'best_val_auc_G_net.pt'), )
    load_pytorch_model(D_net, os.path.join(args.load_dir, 'best_val_auc_D_net.pt'), )


    args = init_metric(args)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        test_results_list = []
        val_results_list = []
        for ss in range(len(val_loader_list)):
            test_results_list.append(
                evaluate(args, model_1, model_2_generate, model_2_res, G_net, test_loader_list[ss], epoch))
            #val_results_list.append(
                #evaluate(args, model_1, model_2_generate, model_2_res, G_net, val_loader_list[ss], epoch))


        one_epoch_time = time.time() - start_time
        args.logger.info('one epoch time is %f' % (one_epoch_time))



if __name__ == '__main__':
    main()
