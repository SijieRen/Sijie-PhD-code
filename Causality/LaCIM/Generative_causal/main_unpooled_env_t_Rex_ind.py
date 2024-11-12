# coding:utf8
from __future__ import print_function
import torch.optim as optim
from torch import nn, optim, autograd
from utils import *
from utils import get_dataset_2D_env as get_dataset_2D
from torchvision import transforms
from models import *
import torch.nn.functional as F
import torch.nn as nn

def mean_nll(logits, y):
    return F.nll_loss(torch.log(logits), y)

def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def VAE_loss(recon_x, x, mu, logvar, args):
    """
    pred_y: predicted y
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    q_y_s: prior
    beta: tradeoff params
    """
    x = x * 0.5 + 0.5
    if args.mse_loss:
        BCE = F.mse_loss(recon_x.view(-1, 3 * args.image_size ** 2), x.view(-1, 3 * args.image_size ** 2),
                                     reduction='mean')
    else:
        BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2), x.view(-1, 3 * args.image_size ** 2),
                                 reduction='mean')
    
    KLD = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
    
    return BCE, KLD



def clip_to_sphere(tens, radius=5, channel_dim=1):
    radi2 = torch.sum(tens**2, dim=channel_dim, keepdim=True)
    mask = torch.gt(radi2, radius**2).expand_as(tens)
    tens[mask] = torch.sqrt(
        tens[mask]**2 / radi2.expand_as(tens)[mask] * radius**2)
    return tens

def train(epoch, model, optimizer, dataloader, args):
    all_zs = np.zeros((len(dataloader.dataset), args.zs_dim))
    RECON_loss = AverageMeter()
    KLD_loss = AverageMeter()
    classify_loss = AverageMeter()
    all_loss = AverageMeter()
    accuracy = AverageMeter()
    batch_begin = 0
    model.train()
    args.fix_mu = 1
    args.fix_var = 1
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        loss = torch.FloatTensor([0.0]).cuda()

        recon_loss = torch.FloatTensor([0.0]).cuda()
        kld_loss = torch.FloatTensor([0.0]).cuda()
        cls_loss = torch.FloatTensor([0.0]).cuda()
        for ss in range(args.env_num):
            if torch.sum(env == ss) <= 1:
                continue
            _, recon_x, mu, logvar, z, s, zs = model(x[env == ss,:,:,:], ss, feature=1)
            pred_y = model.get_y_by_zs(mu, logvar)
            recon_loss_t, kld_loss_t = VAE_loss(recon_x, x[env == ss,:,:,:], mu, logvar, args)
            cls_loss_t = F.nll_loss(torch.log(pred_y), target[env == ss])
            
            if ss == 0:
                before_cls_loss = F.nll_loss(torch.log(pred_y), target[env == ss])
            else:
                after_cls_loss = F.nll_loss(torch.log(pred_y), target[env == ss])

            accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target[env == ss].detach().cpu().numpy()),
                            pred_y.size(0))
            recon_loss = torch.add(recon_loss, torch.sum(env == ss) * recon_loss_t)
            kld_loss = torch.add(kld_loss, torch.sum(env == ss) * kld_loss_t)
            cls_loss = torch.add(cls_loss, torch.sum(env == ss) * cls_loss_t)
            # irm_loss = torch.add(irm_loss, torch.sum(env == ss) * irm_loss_t)
        recon_loss = recon_loss / x.size(0)
        kld_loss = kld_loss / x.size(0)
        cls_loss = cls_loss / x.size(0)

        RECON_loss.update(recon_loss.item(), x.size(0))
        KLD_loss.update(kld_loss.item(), x.size(0))
        classify_loss.update(cls_loss.item(), x.size(0))
        loss = torch.add(loss, recon_loss + args.beta * kld_loss + args.gamma * cls_loss + args.alpha * F.mse_loss(before_cls_loss, after_cls_loss))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss.update(loss.item() + args.alpha * F.mse_loss(before_cls_loss, after_cls_loss).item(), x.size(0))
        
        
        if batch_idx % 10 == 0:
            args.logger.info(
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
    args.logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch, args.epochs, all_loss.avg))
    
    return all_zs, accuracy.avg

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
        pred_y = model.get_pred_y(x)
        accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))

        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f' % (accuracy_init.avg, accuracy.avg))
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
    if 'mnist' in args.dataset:
        train_loader = DataLoader(get_dataset_2D(root = args.root, args=args, fold='train',
                                                 transform=transforms.Compose([
                                                     transforms.RandomHorizontalFlip(p=0.5),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False,
                                  drop_last = True)
        test_loader = DataLoader(get_dataset_2D(root = args.root, args=args, fold='test',
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

                                                 ])),
                                  batch_size=args.test_batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=False)
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
                                   num_workers=0,
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
                                  num_workers=0,
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
                model = Generative_model_f_2D_unpooled_env_t_mnist_more_indvidual(in_channel=args.in_channel,
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
                                                             args=args,
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
                                                         args = args,
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
    for epoch in range(1, args.epochs + 1):
        #pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.lr_controler)
        _, _ = train(epoch, model, optimizer, train_loader, args)

        if args.eval_train == 1:
            _, train_acc = evaluate(epoch, model, train_loader, args)
            logger.info('train test acc: %0.4f' % train_acc)
        if val_loader is not None:
            pred_val, val_acc = evaluate(epoch, model, val_loader, args)
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

        other_info = {
            'pred_val': pred_val,
            'pred_test': pred_test,
        }
        checkpoint(epoch, args.model_save_dir, model, is_best, other_info, logger)
        logger.info('epoch %d:, current test_acc: %0.4f, best_acc: %0.4f, at ep %d, val_acc: %0.4f'
                    %(epoch, test_acc, best_acc, best_acc_ep, val_acc))
        xlsx_name = 'Data_%s_%s_model_unpooled_%s_u_%d_fold_%d.xls' % \
                    (args.dataset, args.dataset_type, args.model, args.is_use_u, args.fold)
        save_results_as_xlsx('./results/', xlsx_name, best_acc, best_acc_ep, auc=None, args=args)
    logger.info('model save path: %s'%args.model_save_dir)
    logger.info('*' * 50)
    logger.info('*' * 50)
    
if __name__ =='__main__':
    main()