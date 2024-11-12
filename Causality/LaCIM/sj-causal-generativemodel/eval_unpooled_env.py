# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from utils import get_dataset_2D_env as get_dataset_2D
from torchvision import transforms
from models import *
import torch.nn.functional as F
import torch.nn as nn

def clip_to_sphere(tens, radius=5, channel_dim=1):
    radi2 = torch.sum(tens**2, dim=channel_dim, keepdim=True)
    mask = torch.gt(radi2, radius**2).expand_as(tens)
    tens[mask] = torch.sqrt(
        tens[mask]**2 / radi2.expand_as(tens)[mask] * radius**2)
    return tens

def evaluate_22(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    pred_pos_num = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        #for cls_i in range(args.num_classes):
            # choose the best init point
            # batch_loss = 10000 * np.ones((x.size(0), 1))
            # z_init, s_init = torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1), \
            #        torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1)
        z_init, s_init = None, None
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in range(args.env_num):
                    pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                    #print(mu.mean(), logvar.mean())
                    # if z_init is None:
                    #     z_init, s_init = zs[:, :args.zs_dim//2], zs[:, args.zs_dim//2:]
                    #     min_loss = F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1), reduction='none')
                    #     #min_pred, _ = torch.min(pred_y, dim=1)
                    #     # print(min_pred.size())
                    # else:
                    #     cls_loss = F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1), reduction='none')
                    #     for i in range(x.size(0)):
                    #         if cls_loss[i] < min_loss[i]:
                    #             min_loss[i] = cls_loss[i]
                    #             z_init[i], s_init[i] = zs[i, :args.zs_dim//2], zs[i, args.zs_dim//2:]

                    if z_init is None:
                        z_init, s_init = z, s
                        min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2), (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                             reduction='none').mean(1)
                    else:
                        new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2), (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                             reduction='none').mean(1)
                        for i in range(x.size(0)):
                            if new_loss[i] < min_rec_loss[i]:
                                min_rec_loss[i] = new_loss[i]
                                z_init[i], s_init[i] = z[i], s[i]

                    # if z_init is None:
                    #     z_init, s_init = z, s
                    #     min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2), (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                    #                          reduction='none').mean(1) + F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1), reduction='none')
                    # else:
                    #     new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2), (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                    #                          reduction='none').mean(1) + F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1), reduction='none')
                    #     for i in range(x.size(0)):
                    #         if new_loss[i] < min_rec_loss[i]:
                    #             min_rec_loss[i] = new_loss[i]
                    #             z_init[i], s_init[i] = z[i], s[i]
                    
                    
                    # if args.sample_type == 0:
                    #     if z_init is None:
                    #         z_init, s_init = z, s
                    #         min_pred, _ = torch.min(pred_y, dim=1)
                    #         # print(min_pred.size())
                    #     else:
                    #         for i in range(x.size(0)):
                    #             if pred_y[i,:].mean() < min_pred[i]:
                    #                 min_pred[i] = pred_y[i,:].mean()
                    #                 z_init[i], s_init[i] = z[i], s[i]
                    # elif args.sample_type == 1:
                    #     if z_init is None:
                    #         z_init, s_init = z, s
                    #         min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                    #                              reduction='none').mean(1)
                    #     else:
                    #         new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                    #                              reduction='none').mean(1)
                    #         for i in range(x.size(0)):
                    #             if new_loss[i] < min_rec_loss[i]:
                    #                 min_rec_loss[i] = new_loss[i]
                    #                 z_init[i], s_init[i] = z[i], s[i]
                    # elif args.sample_type == 2:
                    #     if z_init is None:
                    #         z_init, s_init = z, s
                    #         min_cls_loss = F.cross_entropy(pred_y, target, reduction='none')
                    #         # min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                    #         #                      reduction='none').mean(1)
                    #     else:
                    #         # new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                    #         #                      reduction='none').mean(1)
                    #         new_loss = F.cross_entropy(pred_y, target, reduction='none')
                    #         for i in range(x.size(0)):
                    #             if new_loss[i] < min_cls_loss[i]:
                    #                 min_cls_loss[i] = new_loss[i]
                    #                 z_init[i], s_init[i] = z[i], s[i]
                    # elif args.sample_type == 3:
                    #     if z_init is None:
                    #         z_init, s_init = z, s
                    #         min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                    #                              reduction='none').mean(1)
                    #     else:
                    #         new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                    #                              reduction='none').mean(1)
                    #         for i in range(x.size(0)):
                    #             if new_loss[i] < min_rec_loss[i]:
                    #                 min_rec_loss[i] = new_loss[i]
                    #                 z_init[i], s_init[i] = z[i], s[i]
                    # z, s = torch.randn(x.size(0), args.zs_dim // 2).cuda().clamp(-1,1), \
                    #       torch.randn(x.size(0), args.zs_dim // 2).cuda().clamp(-1,1)
                    # recon_x, pred_y = model.get_x_y(z, s)
                    # BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                    #                              reduction='none')
                    # #cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
                    # loss = BCE.mean(1) #+ args.gamma2 * cls_loss
                    # for ii in range(x.size(0)):
                    #     if batch_loss[ii] >= loss[ii].item():
                    #         batch_loss[ii] = copy.deepcopy(loss[ii].item())
                    #         z_init[ii] = z[ii]
                    #         s_init[ii] = s[ii]
        # print(z_init, s_init)
        z, s = z_init, s_init
        _, pred_y = model.get_x_y(z, s)
        accuracy_init.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()))
        
        z.requires_grad = True
        s.requires_grad = True
        if args.eval_optim == 'sgd':
            optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        else:
            optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        
        for i in range(args.test_ep):
            optimizer.zero_grad()
            if args.decay_test_lr:
                if i >= args.test_ep // 2:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr2 * args.lr_decay ** (epoch // args.lr_controler)
            recon_x, pred_y = model.get_x_y(z, s)
            
            if 'mnist' in args.dataset:
                BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                                             reduction='none')
            else:
                BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 256 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 256 ** 2),
                                             reduction='none')
            #cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
            # print(BCE.size(), cls_loss.size())
            # print(BCE.mean(1).size(), BCE.mean(0).size())
            loss = BCE.mean(1)  # + args.gamma2 * cls_loss
            # if i % 10 == 0 and i > 0 and batch_idx % 10 == 0 and batch_idx > 0:
            #     args.logger.info('inner ep %d, %0.4f for cls %d, cur_acc: %0.4f' %
            #                      (i, loss.mean().item(), 0, accuracy.avg))
            for idx in range(x.size(0)):
                # if best_loss[batch_begin + idx][cls_i] >= loss[idx]:
                #     best_loss[batch_begin + idx][cls_i] = loss[idx]
                if args.use_best == 1:
                    # print(i, cls_i, best_loss[batch_begin + idx][cls_i], loss[idx])
                    if best_loss[batch_begin + idx] >= loss[idx].item():
                        best_loss[batch_begin + idx] = copy.deepcopy(loss[idx].item())
                        # print('is best ', cls_i, best_loss[batch_begin + idx][cls_i])
                else:
                    best_loss[batch_begin + idx] = copy.deepcopy(loss[idx].item())
            # if i == args.test_ep - 1 and cls_i == args.num_classes-1:
            #     print(i, cls_i, best_loss[batch_begin:batch_begin+x.size(0), :])
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            # print(i, cls_i, z, s)
        _, pred_y = model.get_x_y(z, s)
        pred_pos_num = pred_pos_num + np.where(np.argmax(np.array(pred_y.detach().cpu().numpy()). \
                                      reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[0]
        accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                     reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()))
        
        batch_begin = batch_begin + x.size(0)
    #args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f'%(accuracy_init.avg, accuracy.avg))
    return pred, accuracy.avg



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
            model = Generative_model_f_2D_unpooled_env(in_channel=args.in_channel,
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
                                          ).cuda()
        else:
            model = Generative_model_f_2D_unpooled_env(in_channel=args.in_channel,
                                     u_dim=args.u_dim,
                                     us_dim=args.us_dim,
                                     zs_dim=args.zs_dim,
                                     num_classes=args.num_classes,
                                     is_use_u=args.is_use_u,
                                        is_sample = args.is_sample,
                                                   decoder_type=0,
                                                   total_env=args.env_num
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
    check = torch.load('%s/best_acc.pth.tar'%args.eval_path)
    #model.alpha = nn.Parameter(torch.FloatTensor([-1]))
    model.load_state_dict(check['state_dict'], strict=True)
    #model.alpha = nn.Parameter(torch.FloatTensor([-1, -1]))
    model = model.cuda()
    
    
    # temp_args = copy.deepcopy(args)
    # temp_args.test_ep = 0
    # pred_test, test_acc = evaluate_22(1, model, test_loader, temp_args)
    # logger.info('test acc: %0.4f, test_ep: %d'
    #             % (test_acc, 0))
    
    # temp, args.test_ep = args.test_ep, 0
    # pred_test, test_acc = evaluate_22(1, model, test_loader, args)
    # if test_acc >= best_acc:
    #     best_acc = copy.deepcopy(test_acc)
    #     best_acc_ep = copy.deepcopy(1)
    #     is_best = 1
    # logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
    #             % (test_acc, args.test_ep, args.lr2, args.reg2, args.sample_num))
    # args.test_ep = temp
    #args.env_num = 1
    for optim_type in ['Adam']:# , 'sgd'
        for test_ep in [50, 80, 100]:
            for lr2 in [0.0005]: # 0.001
                for wd2 in [0.005, 0.01, 0.02]: # 0.01
                    for sample_num in [10]:
                        temp_args = copy.deepcopy(args)
                        temp_args.eval_optim = optim_type
                        temp_args.sample_num = sample_num
                        temp_args.test_ep = test_ep
                        temp_args.lr2 = lr2
                        temp_args.reg2 = wd2
                        pred_test, test_acc = evaluate_22(1, model, test_loader, temp_args)
                        logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d, optims: %s'
                                    % (test_acc, test_ep, lr2, wd2, sample_num, optim_type))
                        if test_acc >= best_acc:
                            best_acc = copy.deepcopy(test_acc)
                            best_acc_ep = copy.deepcopy(1)
                            is_best = 1
                            best_args = copy.deepcopy(temp_args)
                        else:
                            is_best = 0
    print('best acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d, optims: %s'
                                    % (test_acc, best_args.test_ep, best_args.lr2, best_args.reg2, best_args.sample_num, best_args.eval_optim))
    
if __name__ =='__main__':
    main()