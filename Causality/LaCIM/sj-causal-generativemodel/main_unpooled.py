# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from utils import get_dataset_2D_env as get_dataset_2D
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
    args.fix_mu = 1
    args.fix_var = 1
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        loss = torch.FloatTensor([0.0]).cuda()
        pred_y, recon_x, mu, logvar, z, s, zs = model(x, env, target, feature=1)
        
        recon_loss, kld_loss = VAE_loss(recon_x, x, mu, logvar, mu, logvar, zs, args)
        cls_loss = F.cross_entropy(pred_y, target)
        RECON_loss.update(recon_loss.item(), x.size(0))
        KLD_loss.update(kld_loss.item(), x.size(0))
        classify_loss.update(cls_loss.item(), x.size(0))
        loss = torch.add(loss, recon_loss + args.beta * kld_loss + args.gamma * cls_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss.update(loss.item(), x.size(0))
        accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
        
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


def evaluate(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        for cls_i in range(args.num_classes):
            z,s = 0.1 * torch.zeros(x.size(0), args.zs_dim//2).cuda(), 0.1 * torch.zeros(x.size(0), args.zs_dim//2).cuda()
            z.requires_grad = True
            s.requires_grad = True
            if args.eval_optim == 'sgd':
                optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
            else:
                optimizer = optim.Adam(params=[z,s], lr=args.lr2, weight_decay=args.reg2)
            
            for i in range(args.test_ep):
                optimizer.zero_grad()
                if args.decay_test_lr:
                    if i >= args.test_ep // 2:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = args.lr2 * args.lr_decay ** (epoch // args.lr_controler)
                recon_x, pred_y = model.get_x_y(z, s)
                
                if 'mnist' in args.dataset:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2), reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 256 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 256 ** 2), reduction='none')
                cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
                # print(BCE.size(), cls_loss.size())
                # print(BCE.mean(1).size(), BCE.mean(0).size())
                loss = BCE.mean(1) + args.gamma2 * cls_loss
                if i % 10 == 0 and i > 0 and batch_idx%10 == 0 and batch_idx > 0:
                    args.logger.info('inner ep %d, %0.4f for cls %d, cur_acc: %0.4f' %
                                     (i, loss.mean().item(), cls_i, accuracy.avg))
                for idx in range(x.size(0)):
                    if best_loss[batch_begin + idx][cls_i] >= loss[idx]:
                        best_loss[batch_begin + idx][cls_i] = loss[idx]
                    # if args.use_best == 1:
                    #     #print(i, cls_i, best_loss[batch_begin + idx][cls_i], loss[idx])
                    #     if best_loss[batch_begin + idx][cls_i] >= loss[idx].item():
                    #         best_loss[batch_begin + idx][cls_i] = copy.deepcopy(loss[idx].item())
                    #         #print('is best ', cls_i, best_loss[batch_begin + idx][cls_i])
                    # else:
                    #     best_loss[batch_begin + idx][cls_i] = copy.deepcopy(loss[idx].item())
                # if i == args.test_ep - 1 and cls_i == args.num_classes-1:
                #     print(i, cls_i, best_loss[batch_begin:batch_begin+x.size(0), :])
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                #print(i, cls_i, z, s)
        accuracy.update(compute_acc(-1 * np.array(best_loss[batch_begin:batch_begin+x.size(0), :]).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        # if accuracy.avg >= 0.85:
        #     for idx in range(x.size(0)):
        #         if np.argmin(best_loss[batch_begin+idx, :]) != target.detach().cpu().numpy()[idx]:
        #             args.logger.info(str(best_loss[batch_begin+idx, :]) +
        #                              '%s'%dataloader.dataset.image_path_list[batch_begin+idx])
        #print(accuracy.avg, best_loss[batch_begin:batch_begin+x.size(0), :])
        batch_begin = batch_begin + x.size(0)
    return pred, accuracy.avg


def evaluate_alpha(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        for cls_i in range(args.num_classes):
            z, s = torch.zeros(x.size(0), args.zs_dim // 2).cuda(), torch.zeros(x.size(0), args.zs_dim // 2).cuda()
            z.requires_grad = True
            s.requires_grad = True
            optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
            
            for i in range(args.test_ep):
                optimizer.zero_grad()
                recon_x, pred_y = model.get_x_y(z, s)
                if 'mnist' in args.dataset:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                                                 reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 256 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 256 ** 2),
                                                 reduction='none')
                cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
                # print(BCE.size(), cls_loss.size())
                # print(BCE.mean(1).size(), BCE.mean(0).size())
                loss = BCE.mean(1) + args.gamma * cls_loss
                if i % 10 == 0 and i > 0 and batch_idx % 10 == 0 and batch_idx > 0:
                    args.logger.info('inner ep %d, %0.4f for cls %d, cur_acc: %0.4f' %
                                     (i, loss.mean().item(), cls_i, accuracy.avg))
                for idx in range(x.size(0)):
                    if best_loss[batch_begin + idx][cls_i] >= loss[idx]:
                        best_loss[batch_begin + idx][cls_i] = loss[idx]
                loss = loss.mean()
                loss.backward()
                optimizer.step()
        shift_pred = torch.exp(model.alpha * torch.FloatTensor(best_loss[batch_begin:batch_begin + x.size(0), :]).cuda() + 0.9) / \
                     torch.sum(torch.exp(model.alpha * torch.FloatTensor(best_loss[batch_begin:batch_begin + x.size(0), :]).cuda() + 0.9),
                               dim=1).unsqueeze(1).repeat(1, args.num_classes)
        accuracy.update(compute_acc(-1 * shift_pred.detach().cpu().numpy(), target.detach().cpu().numpy()), x.size(0))
        
        # print(accuracy.avg, best_loss[batch_begin:batch_begin+x.size(0), :])
        batch_begin = batch_begin + x.size(0)
    return pred, accuracy.avg

def train_alpha(model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    model.alpha.requires_grad = True
    optimizer_alpha = optim.Adam(params=[model.alpha], lr=args.lr3, weight_decay=args.reg3)
    
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        for cls_i in range(args.num_classes):
            z, s = torch.zeros(x.size(0), args.zs_dim // 2).cuda(), torch.zeros(x.size(0), args.zs_dim // 2).cuda()
            z.requires_grad = True
            s.requires_grad = True
            optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
            
            for i in range(args.test_ep):
                optimizer.zero_grad()
                recon_x, pred_y = model.get_x_y(z, s)
                if 'mnist' in args.dataset:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                                                 reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 256 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 256 ** 2),
                                                 reduction='none')
                cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
                # print(BCE.size(), cls_loss.size())
                # print(BCE.mean(1).size(), BCE.mean(0).size())
                loss = BCE.mean(1) + args.gamma * cls_loss
                if i % 10 == 0 and i > 0 and batch_idx % 10 == 0 and batch_idx > 0:
                    args.logger.info('inner ep %d, %0.4f for cls %d, cur_acc: %0.4f, with alpha %0.4f' %
                                     (i, loss.mean().item(), cls_i, accuracy.avg, model.alpha.detach()))
                for idx in range(x.size(0)):
                    if best_loss[batch_begin + idx][cls_i] >= loss[idx]:
                        best_loss[batch_begin + idx][cls_i] = loss[idx]
                loss = loss.mean()
                loss.backward()
                optimizer.step()
        shift_pred = torch.exp( model.alpha * torch.FloatTensor(best_loss[batch_begin:batch_begin + x.size(0), :]).cuda() + 0.9) / \
                     torch.sum(torch.exp(model.alpha * torch.FloatTensor(best_loss[batch_begin:batch_begin + x.size(0), :]).cuda() + 0.9),
                               dim=1).unsqueeze(1).repeat(1, args.num_classes)
        other_loss = F.nll_loss(torch.log(shift_pred), target)
        other_loss.backward()
        optimizer_alpha.step()
        accuracy.update(compute_acc(shift_pred.detach().cpu().numpy().
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        # print(accuracy.avg, best_loss[batch_begin:batch_begin+x.size(0), :])
        batch_begin = batch_begin + x.size(0)
    args.logger.info('training alpha is %0.4f'%model.alpha.detach())
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
            model = Generative_model_f_2D_unpooled(in_channel=args.in_channel,
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
            model = Generative_model_f_2D_unpooled(in_channel=args.in_channel,
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
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args.lr_decay, args.lr_controler)
        if args.train_alpha:
            _, _ = train(epoch, model, optimizer, train_loader, args)
            _, train_acc = train_alpha(model, train_loader, args)
            # logger.info('train test acc: %0.4f'%train_acc)
            if val_loader is not None:
                pred_val, val_acc = evaluate_alpha(epoch, model, val_loader, args)
            else:
                pred_val = None
                val_acc = -1
            pred_test, test_acc = evaluate_alpha(epoch, model, test_loader, args)
        else:
            _, _ = train(epoch, model, optimizer, train_loader, args)
            if args.eval_train == 1:
                _, train_acc = evaluate(epoch, model, train_loader, args)
                logger.info('train test acc: %0.4f'%train_acc)
            if val_loader is not None:
                pred_val, val_acc = evaluate(epoch, model, val_loader, args)
            else:
                pred_val = None
                val_acc = -1
            pred_test, test_acc = evaluate(epoch, model, test_loader, args)
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
    logger.info('*' * 50)
    logger.info('*' * 50)
    
if __name__ =='__main__':
    main()