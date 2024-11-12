# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from utils import get_dataset_2D_env as get_dataset_2D
from torchvision import transforms
from models import *
import torch.nn.functional as F
import torch.nn as nn
import copy
from shutil import copyfile
from eval_unpooled_env_t import *

def main():
    args = get_opt()
    args = make_dirs(args)
    logger = get_logger(args)
    args.logger = logger
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
    if args.dataset == 'AD':
        train_loader = DataLoaderX(get_dataset(args=args, fold='train', aug = args.aug),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=True,
                                   drop_last = True)
        test_loader = DataLoaderX(get_dataset(args=args, fold='test', aug = 0),
                                   batch_size=args.test_batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                  drop_last = True)
        val_loader = None
    elif 'mnist' in args.dataset:
        train_loader = DataLoader(get_dataset_2D(root = args.root, args=args, fold='train',
                                                 transform=transforms.Compose([
                                                     transforms.RandomHorizontalFlip(p=0.5),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True,
                                  drop_last = True)
        test_loader = DataLoaderX(get_dataset_2D(root = args.root, args=args, fold='test',
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True)
        val_loader = None
    else:
        train_loader = DataLoaderX(get_dataset_2D(root = args.root, args=args, fold='train',
                                                  transform=transforms.Compose([
                                                            transforms.RandomResizedCrop((256, 256)),
                                                            transforms.RandomHorizontalFlip(p=0.5),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                            
                                                       ])),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=True,
                                   drop_last = True)
        test_loader = DataLoaderX(get_dataset_2D(root = args.root, args=args, fold='test',
                                                 transform=transforms.Compose([
                                                            transforms.Resize((256, 256)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                            
                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True)
        val_loader = None

    if args.model == 'VAE' or args.model == 'VAE_old':
        if args.dataset == 'AD':
            model = Generative_model(in_channel = args.in_channel,
                                 u_dim = args.u_dim,
                                 us_dim = args.us_dim,
                                 zs_dim = args.zs_dim,
                                 num_classes = args.num_classes,
                                 is_use_u = args.is_use_u,
                             ).cuda()
        else:
            model = Generative_model_2D(in_channel=args.in_channel,
                                     u_dim=args.u_dim,
                                     us_dim=args.us_dim,
                                     zs_dim=args.zs_dim,
                                     num_classes=args.num_classes,
                                     is_use_u=args.is_use_u,
                                        is_sample = args.is_sample,
                                     ).cuda()
    elif args.model == 'VAE_f':
        if args.dataset == 'AD':
            model = Generative_model_f(in_channel = args.in_channel,
                                 u_dim = args.u_dim,
                                 us_dim = args.us_dim,
                                 zs_dim = args.zs_dim,
                                 num_classes = args.num_classes,
                                 is_use_u = args.is_use_u,
                             ).cuda()
        elif 'mnist' in args.dataset:
            if args.smaller_net:
                model = Generative_model_f_2D_unpooled_env_t_mnist_shared_s(in_channel=args.in_channel,
                                                             u_dim=args.u_dim,
                                                             us_dim=args.us_dim,
                                                             zs_dim=args.zs_dim,
                                                             num_classes=args.num_classes,
                                                             is_use_u=args.is_use_u,
                                                             is_sample=args.is_sample,
                                                             decoder_type=1,
                                                             total_env=args.env_num,
                                                             more_shared=args.more_shared,
                                                             more_layer=args.more_layer,
                                                             args=args
                                                             ).cuda()
            else:
                model = Generative_model_f_2D_unpooled_env_t(in_channel=args.in_channel,
                                          u_dim=args.u_dim,
                                          us_dim=args.us_dim,
                                          zs_dim=args.zs_dim,
                                          num_classes=args.num_classes,
                                          is_use_u=args.is_use_u,
                                          is_sample=args.is_sample,
                                                   decoder_type=1,
                                                   total_env=args.env_num,
                                                   more_shared=args.more_shared,
                                                   more_layer=args.more_layer,
                                                         args = args
                                          ).cuda()
        else:
            model = Generative_model_f_2D_unpooled_env_t(in_channel=args.in_channel,
                                     u_dim=args.u_dim,
                                     us_dim=args.us_dim,
                                     zs_dim=args.zs_dim,
                                     num_classes=args.num_classes,
                                     is_use_u=args.is_use_u,
                                        is_sample = args.is_sample,
                                                   decoder_type=0,
                                                   total_env=args.env_num,
                                                         args=args,
                                     ).cuda()
    elif args.model == 'sVAE':
        if args.dataset == 'AD':
            model = sVAE(in_channel=args.in_channel,
                     u_dim=args.u_dim,
                     us_dim=args.us_dim,
                     zs_dim=args.zs_dim,
                     num_classes=args.num_classes,
                     is_use_u=args.is_use_u,
                     ).cuda()
        else:
            model = sVAE_2D(in_channel=args.in_channel,
                         u_dim=args.u_dim,
                         us_dim=args.us_dim,
                         zs_dim=args.zs_dim,
                         num_classes=args.num_classes,
                         is_use_u=args.is_use_u,
                         ).cuda()
    elif args.model == 'sVAE_f':
        if args.dataset == 'AD':
            model = sVAE_f(in_channel=args.in_channel,
                     u_dim=args.u_dim,
                     us_dim=args.us_dim,
                     zs_dim=args.zs_dim,
                     num_classes=args.num_classes,
                     is_use_u=args.is_use_u,
                     ).cuda()
        else:
            model = sVAE_f_2D(in_channel=args.in_channel,
                         u_dim=args.u_dim,
                         us_dim=args.us_dim,
                         zs_dim=args.zs_dim,
                         num_classes=args.num_classes,
                         is_use_u=args.is_use_u,
                         ).cuda()
    elif args.model == 'MM_F':
        if args.dataset == 'AD':
            model = MM_F(in_channel=args.in_channel,
                     u_dim=args.u_dim,
                     us_dim=args.us_dim,
                     num_classes=args.num_classes,
                     is_use_u=args.is_use_u,
                     ).cuda()
        else:
            model = MM_F_2D(in_channel=args.in_channel,
                         u_dim=args.u_dim,
                         us_dim=args.us_dim,
                         num_classes=args.num_classes,
                         is_use_u=args.is_use_u,
                         ).cuda()
    elif args.model == 'MM_F_f':
        if args.dataset == 'AD':
            model = MM_F_f(in_channel=args.in_channel,
                             u_dim=args.u_dim,
                             us_dim=args.us_dim,
                             num_classes=args.num_classes,
                             is_use_u=args.is_use_u,
                           zs_dim=args.zs_dim,
                     ).cuda()
        else:
            model = MM_F_ff_2D(in_channel=args.in_channel,
                             u_dim=args.u_dim,
                             us_dim=args.us_dim,
                             num_classes=args.num_classes,
                             is_use_u=args.is_use_u,
                              zs_dim=args.zs_dim,
                         ).cuda()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    
    best_acc = -1
    check = torch.load('%s/best_acc.pth.tar' % args.eval_path)
    # model.alpha = nn.Parameter(torch.FloatTensor([-1]))
    model.load_state_dict(check['state_dict'], strict=True)
    # model.alpha = nn.Parameter(torch.FloatTensor([-1, -1]))
    model = model.cuda()
    epoch = 1
    # pred_test, test_acc = evaluate_xy_full(epoch, model, test_loader, args)
    pred_test, test_acc, total_acc = evaluate_xy_full_plot(epoch, model, test_loader, args)
    #pred_test, test_acc = evaluate_xy_true_full(epoch, model, test_loader, args)
    # pred_test, test_acc = evaluate_xy(epoch, model, test_loader, args)
    # pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
    import xlwt
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('Sheet1')
    
    # print('total acc')
    for i in range(len(total_acc)):
        worksheet.write(i, 0, total_acc[i].avg)
        # print(total_acc[i].avg)
    workbook.save(os.path.join('./results/test_ep.xls'))
    if test_acc >= best_acc:
        best_acc = copy.deepcopy(test_acc)
        best_acc_ep = copy.deepcopy(epoch)
        is_best = 1
        logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
                    % (test_acc, args.test_ep, args.lr2, args.reg2, args.sample_num))

    # for test_ep in [50, 80, 100, 200]:
    #     for lr2 in [0.0005, 0.0002, 0.0001]:
    #         for wd2 in [0.005, 0.001, 0.0001]:
    #             for sample_num in [5, 10]:
    #                 temp_args = copy.deepcopy(args)
    #                 temp_args.sample_num = sample_num
    #                 temp_args.test_ep = test_ep
    #                 temp_args.lr2 = lr2
    #                 temp_args.reg2 = wd2
    #                 model.args = temp_args
    #                 # logger.info('raw pred')
    #                 pred_test, test_acc = evaluate_xy_full(epoch, model, test_loader, temp_args)
    #                 logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
    #                             % (test_acc, test_ep, lr2, wd2, sample_num))
    #                 # logger.info('inner pred')
    #                 # pred_test, test_acc = evaluate_22(epoch, model, test_loader, temp_args)
    #                 logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
    #                             % (test_acc, test_ep, lr2, wd2, sample_num))
    #                 if test_acc >= best_acc:
    #                     best_acc = copy.deepcopy(test_acc)
    #                     best_acc_ep = copy.deepcopy(epoch)
    #                     is_best = 1
                        
    logger.info('best_acc: %0.4f' % (best_acc))
    
    logger.info('model save path: %s'%args.model_save_dir)
    logger.info('*' * 50)
    logger.info('*' * 50)
    
if __name__ =='__main__':
    main()