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


def evaluate_xy_full_plot(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    BCE_loss = AverageMeter()
    pred_pos_num = 0
    error_index = []
    counter = 0
    total_acc = [AverageMeter() for i in range(args.test_ep + 1)]
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        z_init, s_init = None, None
        # print('batch_begin', batch_begin)
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in [0, 1]:
                    pred_y, recon_x, mu, logvar, _, _, z, s, zs = model(x, env_idx, feature=1, is_train=1)
                    # print('sample idx %d, env: %d, y: %d, pred' % (ss, env_idx, target.cpu().detach().numpy()), pred_y)
                    
                    if z_init is None:
                        z_init, s_init = z, s
                        # min_rec_loss = - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long(), reduction='none')
                        min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                              (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                              reduction='none').mean(1)
                        BCE_loss.update(min_rec_loss.mean(), x.size(0))
                    else:
                        # new_loss = - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long(), reduction='none')
                        new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                          (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                          reduction='none').mean(1)
                        BCE_loss.update(new_loss.mean(), x.size(0))
                        for i in range(x.size(0)):
                            if new_loss[i] < min_rec_loss[i]:
                                min_rec_loss[i] = new_loss[i]
                                z_init[i], s_init[i] = z[i], s[i]
        
        z, s = z_init, s_init
        pred_y = model.get_y(s)
        # print('end init', pred_y)
        accuracy_init.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                         reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()),
                             x.size(0))
        total_acc[0].update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                        reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()),
                            x.size(0))
        z.requires_grad = True
        s.requires_grad = True
        if args.eval_optim == 'sgd':
            optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        else:
            optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        
        for i in range(args.test_ep):
            optimizer.zero_grad()
            recon_x, pred_y = model.get_x_y(z, s)
            
            total_acc[i + 1].update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                                reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()),
                                    x.size(0))
            
            if 'mnist' in args.dataset:
                BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                                             reduction='none')
            else:
                BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 256 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 256 ** 2),
                                             reduction='none')
            loss = BCE.mean(1)
            for idx in range(x.size(0)):
                if args.use_best == 1:
                    # print(i, cls_i, best_loss[batch_begin + idx][cls_i], loss[idx])
                    if best_loss[batch_begin + idx] >= loss[idx].item():
                        best_loss[batch_begin + idx] = copy.deepcopy(loss[idx].item())
                        # print('is best ', cls_i, best_loss[batch_begin + idx][cls_i])
                else:
                    best_loss[batch_begin + idx] = copy.deepcopy(loss[idx].item())
            loss = loss.mean()
            if i % 100 == 0 and i > 0:
                pred_y = model.get_y(s)
                # print('%d, y: %d, loss is %0.4f, s norm: %0.4f' % (i, target.cpu().detach().numpy(), loss.item(), torch.norm(s)), pred_y)
            loss.backward()
            optimizer.step()
        _, pred_y = model.get_x_y(z, s)
        # print('final pred_y', pred_y)
        # for i in range(x.size(0)):
        #     if np.argmax(np.array(pred_y[i].detach().cpu().numpy())) != target[i].detach().cpu().numpy():
        #         if not os.path.exists('./bad_cases/'):
        #             os.makedirs('./bad_cases/')
        #         filename = dataloader.dataset.image_path_list[batch_begin+i].split('/')[-1]
        #         print('tagret file, ', filename[:-4]+'_%d'%target[i].detach().cpu().numpy()+'.png')
        #         copyfile(dataloader.dataset.image_path_list[batch_begin+i], os.path.join('./bad_cases/', filename[:-4]+'_%d'%target[i].detach().cpu().numpy()+'.png'))
        # error_index.append(dataloader.dataset.image_path_list[batch_begin+i])
        accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()))
        args.logger.info(
            'init_acc: %0.4f, after acc: %0.4f, BCE loss: %0.4f' % (accuracy_init.avg, accuracy.avg, BCE_loss.avg))
        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    # print('counter', counter)
    args.logger.info(
        'init_acc: %0.4f, after acc: %0.4f, BCE loss: %0.4f' % (accuracy_init.avg, accuracy.avg, BCE_loss.avg))
    
    # print(error_index)
    return pred, accuracy.avg, total_acc




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
                model = Generative_model_f_2D_unpooled_env_t_mnist_prior(in_channel=args.in_channel,
                                                             zs_dim=args.zs_dim,
                                                             num_classes=args.num_classes,
                                                             decoder_type=1,
                                                             total_env=args.env_num,
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