# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from torchvision import transforms
from models import *
import sys

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
        train_loader = DataLoaderX(get_dataset(root = args.root, args=args, fold='train', aug = args.aug),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=True,
                                   drop_last = True)
        test_loader = DataLoaderX(get_dataset(root = args.root, args=args, fold='test', aug = 0),
                                   batch_size=args.test_batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,)
        val_loader = None
    elif 'mnist' in args.dataset:
        train_loader = DataLoader(get_dataset_2D_env(root = args.root, args=args, fold='train',
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
        test_loader = DataLoaderX(get_dataset_2D_env(root = args.root, args=args, fold='test',
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True,)
        val_loader = None
    else:
        train_loader = DataLoaderX(get_dataset_2D(args=args, fold='train',
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
        test_loader = DataLoaderX(get_dataset_2D(args=args, fold='test',
                                                 transform=transforms.Compose([
                                                            transforms.Resize((256, 256)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                            
                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True)
        val_loader = DataLoaderX(get_dataset_2D(args=args, fold='val',
                                                 transform=transforms.Compose([
                                                     transforms.Resize((256, 256)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True)
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
                model = Generative_model_2D_mnist(in_channel=args.in_channel,
                                                 u_dim=args.u_dim,
                                                 us_dim=args.us_dim,
                                                 zs_dim=args.zs_dim,
                                                 num_classes=args.num_classes,
                                                 is_use_u=args.is_use_u,
                                                 is_sample=args.is_sample,
                                                 ).cuda()
            else:
                model = Generative_model_f_2D_28(in_channel=args.in_channel,
                                          u_dim=args.u_dim,
                                          us_dim=args.us_dim,
                                          zs_dim=args.zs_dim,
                                          num_classes=args.num_classes,
                                          is_use_u=args.is_use_u,
                                          is_sample=args.is_sample,
                                          ).cuda()
        else:
            model = Generative_model_f_2D(in_channel=args.in_channel,
                                     u_dim=args.u_dim,
                                     us_dim=args.us_dim,
                                     zs_dim=args.zs_dim,
                                     num_classes=args.num_classes,
                                     is_use_u=args.is_use_u,
                                        is_sample = args.is_sample,
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
        elif 'mnist' in args.dataset:
            model = sVAE_f_mnist_2D(in_channel=args.in_channel,
                              u_dim=args.u_dim,
                              us_dim=args.us_dim,
                              zs_dim=args.zs_dim,
                              num_classes=args.num_classes,
                              is_use_u=args.is_use_u,
                              decoder_type=args.decoder_type,
                              ).cuda()
        else:
            model = sVAE_f_2D(in_channel=args.in_channel,
                         u_dim=args.u_dim,
                         us_dim=args.us_dim,
                         zs_dim=args.zs_dim,
                         num_classes=args.num_classes,
                         is_use_u=args.is_use_u,
                              decoder_type = args.decoder_type,
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
        elif 'NICO' in args.dataset:
            model = MM_F_ff_2D_NICO(in_channel=args.in_channel,
                               u_dim=args.u_dim,
                               us_dim=args.us_dim,
                               num_classes=args.num_classes,
                               is_use_u=args.is_use_u,
                               zs_dim=args.zs_dim,
                               ).cuda()
        else:
            if args.smaller_net:
                model = MM_F_ff_2D_mnist(in_channel=args.in_channel,
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
    elif args.model == 'MM_F_f_L':
        model = MM_F_ff_2D_mnist_L(in_channel=args.in_channel,
                                 u_dim=args.u_dim,
                                 us_dim=args.us_dim,
                                 num_classes=args.num_classes,
                                 is_use_u=args.is_use_u,
                                 zs_dim=args.zs_dim,
                                 ).cuda()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params, '%0.4f M'%(pytorch_total_params / 1e6))
    
    if args.inference_only and args.load_path != '':
        model.load_state_dict(torch.load(os.path.join(args.load_path, 'best_acc.pth.tar'))['state_dict'])
        test_loader_NICO = DataLoaderX(get_dataset_NICO_inter(args=args,
                                                 transform=transforms.Compose([
                                                     transforms.Resize((256, 256)),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True)
        pred_test, label_test, test_acc = evaluate_only(model, test_loader_NICO, args)
        save_pred_label_as_xlsx(args.model_save_dir, 'pred.xls', pred_test, label_test, test_loader_NICO, args)
        logger.info('model save path: %s' % args.model_save_dir)
        print('test_acc', test_acc)
        exit(123)
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    
    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay, args.lr_controler)
        _, _ = train(epoch, model, optimizer, train_loader, args)
        if val_loader is not None:
            pred_val, val_acc = evaluate(model, val_loader, args)
        else:
            pred_val = None
            val_acc = -1
        pred_test, test_acc = evaluate(model, test_loader, args)
        if test_acc >= best_acc:
            best_acc = copy.deepcopy(test_acc)
            best_acc_ep = copy.deepcopy(epoch)
            is_best = 1
        else:
            is_best = 0
        other_info = {
            'pred_val': pred_val,
            'pred_test': pred_test,
        }
        checkpoint(epoch, args.model_save_dir, model, is_best, other_info, logger)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d, val_acc: %0.4f'
                    %(epoch, test_acc, best_acc, best_acc_ep, val_acc))
    logger.info('model save path: %s'%args.model_save_dir)
    xlsx_name = '%s_model_%s_u_%d_fold_%d_env_%d.xls'%\
                (os.path.basename(sys.argv[0][:-3]), args.model, args.is_use_u, args.fold, args.env_num)
    save_results_as_xlsx('./results/', xlsx_name, best_acc, best_acc_ep, auc=None, args=args)
    logger.info('xls save path: %s' % xlsx_name)
    logger.info('*' * 50)
    logger.info('*' * 50)
    
if __name__ =='__main__':
    main()