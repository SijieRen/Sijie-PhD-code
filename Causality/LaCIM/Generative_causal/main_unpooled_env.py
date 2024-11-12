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
            pred_y = model.get_pred_y(x[env == ss,:,:,:], ss)
            #print(recon_x.size(), x[env == ss,:,:,:].size())
            recon_loss_t, kld_loss_t = VAE_loss(recon_x, x[env == ss,:,:,:], mu, logvar, mu, logvar, zs, args)
            cls_loss_t = F.nll_loss(torch.log(pred_y), target[env == ss])
            accuracy.update(compute_acc(pred_y.detach().cpu().numpy(), target[env == ss].detach().cpu().numpy()),
                            pred_y.size(0))
            recon_loss = torch.add(recon_loss, torch.sum(env == ss) * recon_loss_t)
            kld_loss = torch.add(kld_loss, torch.sum(env == ss) * kld_loss_t)
            cls_loss = torch.add(cls_loss, torch.sum(env == ss) * cls_loss_t)
        recon_loss = recon_loss / x.size(0)
        kld_loss = kld_loss / x.size(0)
        cls_loss = cls_loss / x.size(0)

        RECON_loss.update(recon_loss.item(), x.size(0))
        KLD_loss.update(kld_loss.item(), x.size(0))
        classify_loss.update(cls_loss.item(), x.size(0))
        loss = torch.add(loss, recon_loss + args.beta * kld_loss + args.gamma * cls_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss.update(loss.item(), x.size(0))
        
        
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
    pred_pos_num = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        for cls_i in range(args.num_classes):
            # choose the best init point
            # batch_loss = 10000 * np.ones((x.size(0), 1))
            # z_init, s_init = torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1), \
            #        torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1)
            with torch.no_grad():
                for ss in range(args.sample_num):
                    for env_idx in range(args.env_num):
                        pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                    
                        #z, s = torch.randn(x.size(0), args.zs_dim // 2).cuda().clamp(-1,1), \
                        #       torch.randn(x.size(0), args.zs_dim // 2).cuda().clamp(-1,1)
                        #recon_x, pred_y = model.get_x_y(z, s)
                        # BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                        #                              reduction='none')
                        # #cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
                        # loss = BCE.mean(1) #+ args.gamma2 * cls_loss
                        # for ii in range(x.size(0)):
                        #     if batch_loss[ii] >= loss[ii].item():
                        #         batch_loss[ii] = copy.deepcopy(loss[ii].item())
                        #         z_init[ii] = z[ii]
                        #         s_init[ii] = s[ii]
            #print(z_init, s_init)
            #z, s = z_init, s_init
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
                cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
                # print(BCE.size(), cls_loss.size())
                # print(BCE.mean(1).size(), BCE.mean(0).size())
                loss = BCE.mean(1)# + args.gamma2 * cls_loss
                if i % 10 == 0 and i > 0 and batch_idx % 10 == 0 and batch_idx > 0:
                    args.logger.info('inner ep %d, %0.4f for cls %d, cur_acc: %0.4f' %
                                     (i, loss.mean().item(), cls_i, accuracy.avg))
                for idx in range(x.size(0)):
                    # if best_loss[batch_begin + idx][cls_i] >= loss[idx]:
                    #     best_loss[batch_begin + idx][cls_i] = loss[idx]
                    if args.use_best == 1:
                        # print(i, cls_i, best_loss[batch_begin + idx][cls_i], loss[idx])
                        if best_loss[batch_begin + idx][cls_i] >= loss[idx].item():
                            best_loss[batch_begin + idx][cls_i] = copy.deepcopy(loss[idx].item())
                            # print('is best ', cls_i, best_loss[batch_begin + idx][cls_i])
                    else:
                        best_loss[batch_begin + idx][cls_i] = copy.deepcopy(loss[idx].item())
                # if i == args.test_ep - 1 and cls_i == args.num_classes-1:
                #     print(i, cls_i, best_loss[batch_begin:batch_begin+x.size(0), :])
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # print(i, cls_i, z, s)
            
        # print(np.argmin(np.array(best_loss[batch_begin:batch_begin+x.size(0), :]). \
        #                             reshape((x.size(0), args.num_classes))), axis=1)
        # print(np.argmin(np.array(best_loss[batch_begin:batch_begin + x.size(0), :]). \
        #                 reshape((x.size(0), args.num_classes))), axis=0)
        pred_pos_num = pred_pos_num + np.where(np.argmin(np.array(best_loss[batch_begin:batch_begin+x.size(0), :]). \
                                    reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[0]
        accuracy.update(compute_acc(-1 * np.array(best_loss[batch_begin:batch_begin+x.size(0), :]).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        # if accuracy.avg >= 0.85:
        #     for idx in range(x.size(0)):
        #         if np.argmin(best_loss[batch_begin+idx, :]) != target.detach().cpu().numpy()[idx]:
        #             args.logger.info(str(best_loss[batch_begin+idx, :]) +
        #                              '%s'%dataloader.dataset.image_path_list[batch_begin+idx])
        #print(accuracy.avg, best_loss[batch_begin:batch_begin+x.size(0), :])
        batch_begin = batch_begin + x.size(0)
    args.logger.info('pred_pos_sample: %d'%pred_pos_num)
    return pred, accuracy.avg


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
        # for cls_i in range(args.num_classes):
        # choose the best init point
        # batch_loss = 10000 * np.ones((x.size(0), 1))
        # z_init, s_init = torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1), \
        #        torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1)
        z_init, s_init = None, None
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in range(args.env_num):
                    pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                    # print(mu.mean(), logvar.mean())
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
                        min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                              (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                              reduction='none').mean(1)
                    else:
                        new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                          (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
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
            # cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
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
                                                         reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[
            0]
        accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()))
        
        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f' % (accuracy_init.avg, accuracy.avg))
    return pred, accuracy.avg


def evaluate_2(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    pred_pos_num = 0
    total_correct = 0
    loss_all = np.zeros((dataloader.dataset.__len__(), ))
    pred_all = np.zeros((dataloader.dataset.__len__(), ))
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        batch_loss = 10000 * np.ones((x.size(0), 2))
        for cls_i in range(2):
            # choose the best init point
            
            z_init, s_init = torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1), \
                             torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1)
            with torch.no_grad():
                for ss in range(1):
                    for env_idx in range(args.env_num):
                        pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)
                        #print(pred_y, target)
                        
                        # z, s = torch.randn(x.size(0), args.zs_dim // 2).cuda().clamp(-1,1), \
                        #       torch.randn(x.size(0), args.zs_dim // 2).cuda().clamp(-1,1)
                        # recon_x, pred_y = model.get_x_y(z, s)
                        BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2),
                                                     (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                                                     reduction='none')
                        loss_all[batch_begin:batch_begin+x.size(0)] = BCE.mean(1).cpu().numpy()
                        pred_all[batch_begin:batch_begin+x.size(0)] = pred_y[:,1].cpu().numpy()
                        #batch_loss[:, cls_i] = BCE.mean(1).detach().cpu().numpy()
        # correct = np.sum(
        #     np.argmin(batch_loss, axis=1) == target.detach().cpu().numpy())
        # correct = np.sum(
        #     np.argmin(pred_y.detach().cpu().numpy(), axis=1) == target.detach().cpu().numpy())
        # print('%d/%d, %0.2f' % (correct, x.size(0), correct / x.size(0)))
        # total_correct = total_correct + correct
                        # cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i,
                        #                            reduction='none')
                        # loss = BCE.mean(1) + args.gamma2 * cls_loss
                        # for ii in range(x.size(0)):
                        #     if batch_loss[ii] >= loss[ii].item():
                        #         batch_loss[ii] = copy.deepcopy(loss[ii].item())
                        #         z_init[ii] = z[ii]
                        #         s_init[ii] = s[ii]
            # print(z_init, s_init)
        #     z, s = z_init, s_init
        #     z.requires_grad = True
        #     s.requires_grad = True
        #     if args.eval_optim == 'sgd':
        #         optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        #     else:
        #         optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        #
        #     for i in range(args.test_ep):
        #         optimizer.zero_grad()
        #         if args.decay_test_lr:
        #             if i >= args.test_ep // 2:
        #                 for param_group in optimizer.param_groups:
        #                     param_group['lr'] = args.lr2 * args.lr_decay ** (epoch // args.lr_controler)
        #         recon_x, pred_y = model.get_x_y(z, s)
        #
        #         if 'mnist' in args.dataset:
        #             BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
        #                                          reduction='none')
        #         else:
        #             BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 256 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 256 ** 2),
        #                                          reduction='none')
        #         cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
        #         # print(BCE.size(), cls_loss.size())
        #         # print(BCE.mean(1).size(), BCE.mean(0).size())
        #         loss = BCE.mean(1)# + args.gamma2 * cls_loss
        #         if i % 10 == 0 and i > 0 and batch_idx % 10 == 0 and batch_idx > 0:
        #             args.logger.info('inner ep %d, %0.4f for cls %d, cur_acc: %0.4f' %
        #                              (i, loss.mean().item(), cls_i, accuracy.avg))
        #         for idx in range(x.size(0)):
        #             # if best_loss[batch_begin + idx][cls_i] >= loss[idx]:
        #             #     best_loss[batch_begin + idx][cls_i] = loss[idx]
        #             if args.use_best == 1:
        #                 # print(i, cls_i, best_loss[batch_begin + idx][cls_i], loss[idx])
        #                 if best_loss[batch_begin + idx][cls_i] >= loss[idx].item():
        #                     best_loss[batch_begin + idx][cls_i] = copy.deepcopy(loss[idx].item())
        #                     # print('is best ', cls_i, best_loss[batch_begin + idx][cls_i])
        #             else:
        #                 best_loss[batch_begin + idx][cls_i] = copy.deepcopy(loss[idx].item())
        #         # if i == args.test_ep - 1 and cls_i == args.num_classes-1:
        #         #     print(i, cls_i, best_loss[batch_begin:batch_begin+x.size(0), :])
        #         loss = loss.mean()
        #         loss.backward()
        #         optimizer.step()
        #         # print(i, cls_i, z, s)
        # # print(np.argmin(np.array(best_loss[batch_begin:batch_begin+x.size(0), :]). \
        # #                             reshape((x.size(0), args.num_classes))), axis=1)
        # # print(np.argmin(np.array(best_loss[batch_begin:batch_begin + x.size(0), :]). \
        # #                 reshape((x.size(0), args.num_classes))), axis=0)
        # pred_pos_num = pred_pos_num + np.where(np.argmin(np.array(best_loss[batch_begin:batch_begin + x.size(0), :]). \
        #                                                  reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[
        #     0]
        # accuracy.update(compute_acc(-1 * np.array(best_loss[batch_begin:batch_begin + x.size(0), :]).
        #                             reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        # if accuracy.avg >= 0.85:
        #     for idx in range(x.size(0)):
        #         if np.argmin(best_loss[batch_begin+idx, :]) != target.detach().cpu().numpy()[idx]:
        #             args.logger.info(str(best_loss[batch_begin+idx, :]) +
        #                              '%s'%dataloader.dataset.image_path_list[batch_begin+idx])
        # print(accuracy.avg, best_loss[batch_begin:batch_begin+x.size(0), :])
        batch_begin = batch_begin + x.size(0)
    print('total_correct', total_correct, total_correct/ dataloader.dataset.__len__())
    args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    import scipy
    import scipy.stats
    print('spearman :', scipy.stats.spearmanr(loss_all, pred_all))
    return pred, accuracy.avg


def evaluate_all(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    pred_pos_num = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        batch_loss = 10000 * np.ones((x.size(0), args.num_classes))
        # z_init, s_init = torch.zeros(x.size(0), args.num_classes, args.zs_dim // 2).cuda().clamp(-1, 1), \
        #                  torch.zeros(x.size(0), args.num_classes, args.zs_dim // 2).cuda().clamp(-1, 1)
        # for cls_i in range(args.num_classes):
        #     for ss in range(args.sample_num):
        #         z, s = torch.randn(x.size(0), args.zs_dim // 2).cuda().clamp(-5, 5), \
        #                torch.randn(x.size(0), args.zs_dim // 2).cuda().clamp(-5, 5)
        #         recon_x, pred_y = model.get_x_y(z, s)
        #         BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
        #                                      reduction='none')
        #         cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
        #         loss = BCE.mean(1) + args.gamma2 * cls_loss
        #         for ii in range(x.size(0)):
        #             if batch_loss[ii] >= loss[ii].item():
        #                 batch_loss[ii] = copy.deepcopy(loss[ii].item())
        #                 z_init[ii] = z[ii]
        #                 s_init[ii] = s[ii]
        # print(z_init, s_init)
        
        
        # z_init, s_init = torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1), \
        #                  torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1)
        batch_loss = 10000 * np.ones((x.size(0), args.num_classes))
        
        for cls_i in range(args.num_classes):
            # choose the best init point
            z_init, s_init = torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1), \
                             torch.zeros(x.size(0), args.zs_dim // 2).cuda().clamp(-1, 1)
            with torch.no_grad():
                for ss in range(args.sample_num):
                    for env_idx in range(args.env_num):
                        pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1)

                        # z, s = torch.randn(x.size(0), args.zs_dim // 2).cuda().clamp(-1,1), \
                        #       torch.randn(x.size(0), args.zs_dim // 2).cuda().clamp(-1,1)
                        # recon_x, pred_y = model.get_x_y(z, s)
                        BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2),
                                                     (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                                                     reduction='none')
                        cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i,
                                                   reduction='none')
                        loss = BCE.mean(1) + args.gamma2 * cls_loss
                        choose_pos = 0
                        for ii in range(x.size(0)):
                            if batch_loss[ii][cls_i] >= loss[ii].item():
                                batch_loss[ii][cls_i] = copy.deepcopy(loss[ii].item())
                                z_init[ii] = z[ii]
                                s_init[ii] = s[ii]
                         #   if batch_loss[ii][1] > batch_loss[ii][0]:
                         #       choose_pos = choose_pos + 1
                        #correct = np.sum(np.argmin(batch_loss, axis=1) == target.detach().cpu().numpy())
                        #print(target)
                        #print('bs, pos_pred, correct', x.size(0), choose_pos, correct)
        z, s = z_init, s_init
        z.requires_grad = True
        s.requires_grad = True
        if args.eval_optim == 'sgd':
            optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        else:
            optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        
        
        for i in range(args.test_ep):
            optimizer.zero_grad()
            all_loss = torch.FloatTensor([0.0]).cuda()
            for cls_i in range(args.num_classes):
                # choose the best init point
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
                cls_loss = F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
                # print(BCE.size(), cls_loss.size())
                # print(BCE.mean(1).size(), BCE.mean(0).size())
                loss = BCE.mean(1) + args.gamma2 * cls_loss
                if i % 10 == 0 and i > 0 and batch_idx % 10 == 0 and batch_idx > 0:
                    args.logger.info('inner ep %d, %0.4f for cls %d, cur_acc: %0.4f' %
                                     (i, loss.mean().item(), cls_i, accuracy.avg))
                for idx in range(x.size(0)):
                    # if best_loss[batch_begin + idx][cls_i] >= loss[idx]:
                    #     best_loss[batch_begin + idx][cls_i] = loss[idx]
                    if args.use_best == 1:
                        # print(i, cls_i, best_loss[batch_begin + idx][cls_i], loss[idx])
                        if best_loss[batch_begin + idx][cls_i] >= loss[idx].item():
                            best_loss[batch_begin + idx][cls_i] = copy.deepcopy(loss[idx].item())
                            # print('is best ', cls_i, best_loss[batch_begin + idx][cls_i])
                    else:
                        best_loss[batch_begin + idx][cls_i] = copy.deepcopy(loss[idx].item())
                # if i == args.test_ep - 1 and cls_i == args.num_classes-1:
                #     print(i, cls_i, best_loss[batch_begin:batch_begin+x.size(0), :])
                loss = loss.mean()
                all_loss = torch.add(all_loss, loss)

            all_loss.backward()
            optimizer.step()

        pred_pos_num = pred_pos_num + np.where(np.argmin(np.array(best_loss[batch_begin:batch_begin + x.size(0), :]). \
                                                         reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[
            0]
        accuracy.update(compute_acc(-1 * np.array(best_loss[batch_begin:batch_begin + x.size(0), :]).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        batch_begin = batch_begin + x.size(0)
    args.logger.info('pred_pos_sample: %d' % pred_pos_num)
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
        shift_pred = torch.exp(model.alpha.repeat(x.size(0), 1) * torch.FloatTensor(best_loss[batch_begin:batch_begin + x.size(0), :]).cuda() + 0.9) / \
                     torch.sum(torch.exp(model.alpha.repeat(x.size(0), 1) * torch.FloatTensor(best_loss[batch_begin:batch_begin + x.size(0), :]).cuda() + 0.9),
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
                    args.logger.info('inner ep %d, %0.4f for cls %d, cur_acc: %0.4f, with alpha %0.4f, %0.4f' %
                                     (i, loss.mean().item(), cls_i, accuracy.avg, model.alpha[0].detach(), model.alpha[1].detach()))
                for idx in range(x.size(0)):
                    if best_loss[batch_begin + idx][cls_i] >= loss[idx]:
                        best_loss[batch_begin + idx][cls_i] = loss[idx]
                loss = loss.mean()
                loss.backward()
                optimizer.step()
        shift_pred = torch.exp( model.alpha.repeat(x.size(0), 1) * torch.FloatTensor(best_loss[batch_begin:batch_begin + x.size(0), :]).cuda() + 0.9) / \
                     torch.sum(torch.exp(model.alpha.repeat(x.size(0), 1) * torch.FloatTensor(best_loss[batch_begin:batch_begin + x.size(0), :]).cuda() + 0.9),
                               dim=1).unsqueeze(1).repeat(1, args.num_classes)
        other_loss = F.nll_loss(torch.log(shift_pred), target)
        other_loss.backward()
        optimizer_alpha.step()
        accuracy.update(compute_acc(shift_pred.detach().cpu().numpy().
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        # print(accuracy.avg, best_loss[batch_begin:batch_begin+x.size(0), :])
        batch_begin = batch_begin + x.size(0)
    args.logger.info('training alpha is %0.4f, %0.4f'%(model.alpha[0].detach(), model.alpha[1].detach()))
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
        train_loader = DataLoaderX(get_dataset_2D(root = args.root, args=args, fold='train',
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
        test_loader = DataLoaderX(get_dataset_2D(root = args.root, args=args, fold='test',
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
        # val_loader = DataLoaderX(get_dataset_2D(args=args, fold='val',
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
    for epoch in range(1, args.epochs + 1):
        # pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
        #check = torch.load('./results/mnist_2/save_model_VAE_f_test_1_1.0_0.1_lr_0.2000_40_0.2000_wd_0.0005_2020-09-11_14-26-51/checkpoints.pth.tar')
        #check = torch.load('./results/mnist_2/save_model_VAE_f_test_1_1.0_0.1_lr_0.2000_40_0.2000_wd_0.0005_2020-09-11_22-33-49/checkpoints.pth.tar')
        # model.alpha = nn.Parameter(torch.FloatTensor([-1]))
        # model.load_state_dict(check['state_dict'], strict=True)
        # model.alpha = nn.Parameter(torch.FloatTensor([-1, -1]))
        # model = model.cuda()
        #_, train_acc = train_alpha(model, train_loader, args)
        #evaluate_alpha(epoch, model, test_loader, args)
        # _, test_acc = evaluate_all(epoch, model, test_loader, args)
        # print('test_acc', test_acc)
        # _, train_acc = evaluate_all(epoch, model, train_loader, args)
        # print('train acc: ', train_acc)
        
        # _, test_acc = evaluate_22(epoch, model, test_loader, args)
        # print('test_acc', test_acc)
        # _, train_acc = evaluate_22(epoch, model, train_loader, args)
        # print('train acc: ', train_acc)
        
        
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
            
            # temp, args.test_ep = args.test_ep, 0
            # pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
            # if test_acc >= best_acc:
            #     best_acc = copy.deepcopy(test_acc)
            #     best_acc_ep = copy.deepcopy(epoch)
            #     is_best = 1
            #     logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
            #                 % (test_acc, args.test_ep, args.lr2, args.reg2, args.sample_num))
            # args.test_ep = temp
            for test_ep in [50]:
                for lr2 in [0.0005]:
                    for wd2 in [0.005]:
                        for sample_num in [5, 10]:
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