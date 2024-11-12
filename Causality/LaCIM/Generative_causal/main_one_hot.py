# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from torchvision import transforms
from models import *
import torch.nn.functional as F

def train(epoch, model, optimizer, dataloader, args):
    all_zs = np.zeros((len(dataloader.dataset), args.zs_dim))
    RECON_loss = AverageMeter()
    KLD_loss = AverageMeter()
    classify_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    model.train()
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
            
        u = torch.FloatTensor(x.size(0), args.env_num).cuda()
        u.zero_()
        u.scatter_(1, env.unsqueeze(1), 1)
        us = u
        
        if args.model == 'VAE' or args.model == 'sVAE' or args.model == 'VAE_f' or args.model == 'sVAE_f':
            # print(u.size(), us.size())
            _, recon_x, mu, logvar, mu_prior, logvar_prior, zs = model(x, u, us, feature=1)
            pred_y = model.get_pred_y(x, u, us)
        elif args.model == 'VAE_old':
            q_y_s, _, _, _, _, _ = model(x, u, us, feature=0)
            pred_y = model.get_pred_y(x, u, us)
            _, recon_x, mu, logvar, mu_prior, logvar_prior, zs = model(x, u, us, feature=1)
        else:
            pred_y = model(x, u, us)
        # except:
        #     continue
        if args.model == 'VAE' or args.model == 'sVAE' or args.model == 'VAE_f' or args.model == 'sVAE_f':
            if args.solve == 'IRM':
                recon_loss, kld_loss = VAE_loss(recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, args)
                # cls_loss = F.nll_loss(torch.log(pred_y), target)
                cls_loss = torch.FloatTensor([0.0]).cuda()
                for ss in range(args.env_num):
                    if torch.sum(env == ss) == 0:
                        continue
                    cls_loss = torch.add(cls_loss,
                                         torch.sum(env == ss) * (
                                                     F.cross_entropy(pred_y[env == ss, :], target[env == ss]) + \
                                                     args.alpha * penalty(pred_y[env == ss, :],
                                                                          target[env == ss])))
                    # print('IRM loss:', F.cross_entropy(pred_y[env == ss, :], target[env == ss]), args.alpha * penalty(pred_y[env == ss, :],target[env == ss]))
                cls_loss = cls_loss / pred_y.size(0)
                # print(recon_loss, args.beta * kld_loss, args.gamma * cls_loss, args.beta, args.gamma)
                loss = recon_loss + args.beta * kld_loss + args.gamma * cls_loss
            else:
                recon_loss, kld_loss = VAE_loss(recon_x, x, mu, logvar, mu_prior, logvar_prior, zs, args)
                cls_loss = F.nll_loss(torch.log(pred_y), target)
                loss = recon_loss + args.beta * kld_loss + args.alpha * cls_loss
            
            all_zs[batch_begin:batch_begin + x.size(0), :] = \
                zs.detach().view(x.size(0), args.zs_dim).cpu().numpy()
            batch_begin = batch_begin + x.size(0)
        elif args.model == 'VAE_old':
            recon_loss, kld_loss = VAE_old_loss(pred_y, recon_x, x, mu, logvar, mu_prior,
                                                logvar_prior, zs, q_y_s, target, args)
            cls_loss = F.nll_loss(torch.log(pred_y), target)
            loss = recon_loss + args.beta * kld_loss + args.alpha * cls_loss
            
            all_zs[batch_begin:batch_begin + x.size(0), :] = \
                zs.detach().view(x.size(0), args.zs_dim).cpu().numpy()
            batch_begin = batch_begin + x.size(0)
        else:
            loss = F.cross_entropy(pred_y, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' \
                or args.model == 'VAE_f' or args.model == 'sVAE_f':
            RECON_loss.update(recon_loss.item(), x.size(0))
            KLD_loss.update(kld_loss.item(), x.size(0))
            classify_loss.update(cls_loss.item(), x.size(0))
        all_loss.update(loss.item(), x.size(0))
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
        
        if batch_idx % 10 == 0:
            print(
                'epoch [{}/{}], batch: {}, rec_loss:{:.4f}, kld_loss:{:.4f} cls_loss:{:.4f}, overall_loss:{:.4f},acc:{:.4f}'
                .format(epoch,
                        args.epochs,
                        batch_idx,
                        RECON_loss.avg,
                        KLD_loss.avg * args.beta,
                        classify_loss.avg,
                        all_loss.avg,
                        accuracy.avg * 100))
    
    if args.model == 'VAE' or args.model == 'VAE_old' or args.model == 'sVAE' or \
            args.model == 'VAE_f' or args.model == 'sVAE_f':
        all_zs = all_zs[:batch_begin]
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, all_loss.avg))
    
    return all_zs, accuracy.avg


def evaluate_22(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    pred_pos_num = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        
        # for cls_i in range(args.num_classes):
        # choose the best init point
        # batch_loss = 10000 * np.ones((x.size(0), 1))
        # z_init, s_init = torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1), \
        #        torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1)
        z_init, s_init = None, None
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in range(args.env_num):
                    u = torch.FloatTensor(x.size(0), args.env_num).cuda()
                    u.zero_()
                    u.scatter_(1, torch.ones(x.size(0), 1).long().cuda() * env_idx, 1)
                    us = u
                    pred_y, recon_x, mu, logvar, _, _, zs = model(x, u, us, feature=1)
                    #print(recon_x.size(), x.size())
                    
                    if z_init is None:
                        z_init, s_init = zs[:, :args.zs_dim//2], zs[:, args.zs_dim//2:]
                        min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                              (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                              reduction='none').mean(1) + \
                                       F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1), reduction='none')
                    else:
                        new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                          (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                          reduction='none').mean(1) + \
                                   F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1), reduction='none')
                        for i in range(x.size(0)):
                            if new_loss[i] < min_rec_loss[i]:
                                min_rec_loss[i] = new_loss[i]
                                z_init[i], s_init[i] = zs[i, :args.zs_dim//2], zs[i, args.zs_dim//2:]
                    
                    # if z_init is None:
                    #     z_init, s_init = zs[:, :args.zs_dim//2], zs[:, args.zs_dim//2:]
                    #     min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                    #                                           (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                    #                                           reduction='none').mean(1)
                    # else:
                    #     new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                    #                                       (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                    #                                       reduction='none').mean(1)
                    #     for i in range(x.size(0)):
                    #         if new_loss[i] < min_rec_loss[i]:
                    #             min_rec_loss[i] = new_loss[i]
                    #             z_init[i], s_init[i] = zs[i, :args.zs_dim//2], zs[i, args.zs_dim//2:]
                    
                    # if z_init is None:
                    #     z_init, s_init = zs[:, :args.zs_dim//2], zs[:, args.zs_dim//2:]
                    #     min_pred, _ = torch.min(pred_y, dim=1)
                    #     # print(min_pred.size())
                    # else:
                    #     for i in range(x.size(0)):
                    #         if pred_y[i, :].mean() < min_pred[i]:
                    #             min_pred[i] = pred_y[i, :].mean()
                    #             z_init[i], s_init[i] = zs[i, :args.zs_dim//2], zs[i, args.zs_dim//2:]
                    
        z, s = z_init, s_init
        #z, s = torch.randn(z.size()).cuda(), torch.randn(s.size()).cuda()
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
            # cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
            # print(BCE.size(), cls_loss.size())
            # print(BCE.mean(1).size(), BCE.mean(0).size())
            loss = BCE.mean(1)  # + args.gamma2 * cls_loss
            if i % 10 == 0 and i > 0 and batch_idx % 10 == 0 and batch_idx > 0:
                args.logger.info('inner ep %d, %0.4f for cls %d, cur_acc: %0.4f' %
                                 (i, loss.mean().item(), 0, accuracy.avg))
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
        # s = torch.randn(s.size()).cuda()
        # z = torch.randn(z.size()).cuda()
        _, pred_y = model.get_x_y(z, s)
        #print('pred_y, target', pred_y, target)
        pred_pos_num = pred_pos_num + np.where(np.argmax(np.array(pred_y.detach().cpu().numpy()). \
                                                         reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[
            0]
        accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        batch_begin = batch_begin + x.size(0)
    args.logger.info('pred_pos_sample: %d' % pred_pos_num)
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
        train_loader = DataLoaderX(get_dataset_2D_env(root = args.root, args=args, fold='train',
                                                  transform=transforms.Compose([
                                                            transforms.RandomResizedCrop((256, 256)),
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
                                                            transforms.Resize((256, 256)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                            
                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True)
        val_loader = None
        # val_loader = DataLoaderX(get_dataset_2D_env(args=args, fold='val',
        #                                          transform=transforms.Compose([
        #                                              transforms.Resize((256, 256)),
        #                                              transforms.ToTensor(),
        #                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #
        #                                          ])),
        #                           batch_size=args.test_batch_size,
        #                           shuffle=False,
        #                           num_workers=1,
        #                           pin_memory=True)
    if args.model == 'VAE_f':
        if args.dataset == 'AD':
            model = Generative_model_f(in_channel = args.in_channel,
                                 u_dim = args.u_dim,
                                 us_dim = args.us_dim,
                                 zs_dim = args.zs_dim,
                                 num_classes = args.num_classes,
                                 is_use_u = args.is_use_u,
                             ).cuda()
        elif 'mnist' in args.dataset:
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
                              decoder_type = args.decoder_type,
                         ).cuda()
    else:
        print('not valid model')
        exit(123)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.reg)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    
    best_acc = -1
    for epoch in range(1, args.epochs + 1):
        #pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.lr_controler)
        _, _ = train(epoch, model, optimizer, train_loader, args)
        if val_loader is not None:
            pred_val, val_acc = evaluate(model, val_loader, args)
        else:
            pred_val = None
            val_acc = -1

        is_best = 0
        pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
        if test_acc >= best_acc:
            best_acc = copy.deepcopy(test_acc)
            best_acc_ep = copy.deepcopy(epoch)
            is_best = 1
            logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
                        % (test_acc, args.test_ep, args.lr2, args.reg2, args.sample_num))

        # temp, args.test_ep = args.test_ep, 0
        # pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
        # if test_acc >= best_acc:
        #     best_acc = copy.deepcopy(test_acc)
        #     best_acc_ep = copy.deepcopy(epoch)
        #     is_best = 1
        #     logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
        #                 % (test_acc, args.test_ep, args.lr2, args.reg2, args.sample_num))
        # args.test_ep = temp
        for test_ep in [5]:
            for lr2 in [0.0001]:
                for wd2 in [0.0001]:
                    for sample_num in [10]:
                        temp_args = copy.deepcopy(args)
                        temp_args.sample_num = sample_num
                        temp_args.test_ep = test_ep
                        temp_args.lr2 = lr2
                        temp_args.reg2 = wd2
                        pred_test, test_acc = evaluate_22(epoch, model, test_loader, temp_args)
                        if test_acc >= best_acc:
                            best_acc = copy.deepcopy(test_acc)
                            best_acc_ep = copy.deepcopy(epoch)
                            is_best = 1
                            logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
                                        % (test_acc, test_ep, lr2, wd2, sample_num))
        
        other_info = {
            'pred_val': pred_val,
            'pred_test': pred_test,
        }
        checkpoint(epoch, args.model_save_dir, model, is_best, other_info, logger)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d, val_acc: %0.4f'
                    %(epoch, test_acc, best_acc, best_acc_ep, val_acc))
    logger.info('model save path: %s'%args.model_save_dir)
    logger.info('*' * 50)
    logger.info('*' * 50)
    
if __name__ =='__main__':
    main()