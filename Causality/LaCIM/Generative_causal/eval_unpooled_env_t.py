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

def clip_to_sphere(tens, radius=5, channel_dim=1):
    radi2 = torch.sum(tens**2, dim=channel_dim, keepdim=True)
    mask = torch.gt(radi2, radius**2).expand_as(tens)
    tens[mask] = torch.sqrt(
        tens[mask]**2 / radi2.expand_as(tens)[mask] * radius**2)
    return tens


def evaluate_q(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    pred_pos_num = 0
    error_index = []
    counter = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        # print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        if dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_00741_0.10_0.png':# or \
            #dataloader.dataset.image_path_list[batch_begin].split('/')[-1] =='img_00221_0.10_0.png':
            print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
            pass
        print(os.listdir('./bad_cases/'))
        if dataloader.dataset.image_path_list[batch_begin].split('/')[-1][:-4]+'_%d'%target.detach().cpu().numpy()+'.png' in os.listdir('./bad_cases/'):
            counter = counter + 1
            print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        else:
            batch_begin = batch_begin + 1
            continue
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        z_init, s_init = None, None
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in [0]:
                    pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1, is_train=1)
                    print('sample idx %d, y: %d, pred'%(ss, target.cpu().detach().numpy()), pred_y)
                    
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
                   
        _, pred_y = model.get_x_y(z, s)
        print('selected, y: %d, pred' % (target.cpu().detach().numpy()), pred_y)
        
        # for i in range(x.size(0)):
        #     if np.argmax(np.array(pred_y[i].detach().cpu().numpy())) != target[i].detach().cpu().numpy():
        #         if not os.path.exists('./bad_cases/'):
        #             os.makedirs('./bad_cases/')
        #         filename = dataloader.dataset.image_path_list[batch_begin+i].split('/')[-1]
        #         print('tagret file, ', filename[:-4]+'_%d'%target[i].detach().cpu().numpy()+'.png')
        #         copyfile(dataloader.dataset.image_path_list[batch_begin+i], os.path.join('./bad_cases/', filename[:-4]+'_%d'%target[i].detach().cpu().numpy()+'.png'))
                #error_index.append(dataloader.dataset.image_path_list[batch_begin+i])
        accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()))
        
        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    print('counter', counter)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f' % (accuracy_init.avg, accuracy.avg))
    # print(error_index)
    return pred, accuracy.avg


def evaluate_xy(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    all_loss = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    pred_pos_num = 0
    error_index = []
    counter = 0
    cls_i = 1
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        # print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        # if dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_00741_0.10_0.png'  or \
        #     dataloader.dataset.image_path_list[batch_begin].split('/')[-1] =='img_04289_0.10_0.png' or \
        #         dataloader.dataset.image_path_list[batch_begin].split('/')[-1] =='img_03365_0.10_0.png' or \
        #         dataloader.dataset.image_path_list[batch_begin].split('/')[-1] =='img_02654_0.10_0.png':
        #     print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        #     pass
        
        if dataloader.dataset.image_path_list[batch_begin].split('/')[-1] in ['img_02559_0.10_0.png']: # , 'img_02732_0.90_0.png'
            # 2 -> ['img_02044_0.90_0.png', 'img_02610_0.90_0.png']:#['img_03258_0.90_0.png', 'img_03655_0.90_0.png']:
            # error: , 'img_02044_0.10_0.png', 'img_02610_0.10_0.png' right: 'img_00738_0.10_0.png', 'img_00258_0.10_0.png',
            print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        
        # print(os.listdir('./bad_cases/'))
        # if dataloader.dataset.image_path_list[batch_begin].split('/')[-1][
        #    :-4] + '_%d' % target.detach().cpu().numpy() + '.png' in os.listdir('./bad_cases/'):
        #     counter = counter + 1
        #     print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        else:
            batch_begin = batch_begin + 1
            continue
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        z_init, s_init = None, None
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in [0]:
                    pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1, is_train=1)
                    
                    
                    if z_init is None:
                        z_init, s_init = z, s
                        # min_rec_loss =  - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long())
                        bce_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                              (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                              reduction='none').mean(1)
                        cls_loss =  F.nll_loss(torch.log(pred_y), torch.ones(pred_y.size(0), ).long().cuda() * cls_i, reduction='none')
                        min_rec_loss = bce_loss + cls_loss
                        print('sample idx %d, env: %d, y: %d, pred' % (ss, env_idx, target.cpu().detach().numpy()),
                              pred_y, 'BCE loss: ', bce_loss.mean(), 'cls loss: ', cls_loss.mean(), 'all loss:', min_rec_loss)
                        all_loss.update(min_rec_loss.mean(), x.size(0))
                    else:
                        # new_loss = - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long()).item()
                        bce_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                          (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                          reduction='none').mean(1)
                        cls_loss = F.nll_loss(torch.log(pred_y), torch.ones(pred_y.size(0), ).long().cuda() * cls_i,
                                              reduction='none')
                        new_loss = bce_loss + cls_loss
                        print('sample idx %d, env: %d, y: %d, pred' % (ss, env_idx, target.cpu().detach().numpy()),
                              pred_y, 'BCE loss: ', bce_loss.mean(), 'cls loss: ', cls_loss.mean(), 'all loss:', new_loss)
                        all_loss.update(new_loss.mean(), x.size(0))
                        for i in range(x.size(0)):
                            if new_loss[i] < min_rec_loss[i]:
                                min_rec_loss[i] = new_loss[i]
                                z_init[i], s_init[i] = z[i], s[i]
        print('overall min loss:', min_rec_loss, 'mean loss: ', all_loss.avg)
        
        z, s = z_init, s_init
        pred_y = model.get_y(s)
        print('end init', pred_y)
        z.requires_grad = True
        s.requires_grad = True
        if args.eval_optim == 'sgd':
            optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        else:
            optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)

        for i in range(args.test_ep):
            optimizer.zero_grad()
            recon_x, pred_y = model.get_x_y(z, s)

            loss = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                                         reduction='none').mean(1) + F.nll_loss(torch.log(pred_y), )
            loss = BCE.mean(1)
            # for idx in range(x.size(0)):
            #     if args.use_best == 1:
            #         # print(i, cls_i, best_loss[batch_begin + idx][cls_i], loss[idx])
            #         if best_loss[batch_begin + idx] >= loss[idx].item():
            #             best_loss[batch_begin + idx] = copy.deepcopy(loss[idx].item())
            #             # print('is best ', cls_i, best_loss[batch_begin + idx][cls_i])
            #     else:
            #         best_loss[batch_begin + idx] = copy.deepcopy(loss[idx].item())
            loss = loss.mean()
            if i % 50 == 0 and i > 0:
                pred_y = model.get_y(s)
                print('%d, y: %d, loss is %0.4f, s norm: %0.4f'%(i, target.cpu().detach().numpy(), loss.item(), torch.norm(s)), pred_y)
            loss.backward()
            optimizer.step()
        _, pred_y = model.get_x_y(z, s)
        print('final pred_y', pred_y)
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
        
        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    # print('counter', counter)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f' % (accuracy_init.avg, accuracy.avg))
    # print(error_index)
    return pred, accuracy.avg


def evaluate_try(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    BCE_loss = AverageMeter()
    batch_begin = 0
    pred_pos_num = 0
    error_index = []
    counter = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        # print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        if dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_00221_0.10_0.png' or \
                dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_00741_0.10_0.png'  or \
            dataloader.dataset.image_path_list[batch_begin].split('/')[-1] =='img_04289_0.10_0.png' or \
                dataloader.dataset.image_path_list[batch_begin].split('/')[-1] =='img_03365_0.10_0.png' or \
                dataloader.dataset.image_path_list[batch_begin].split('/')[-1] =='img_02654_0.10_0.png':
            print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        #     pass
        # print(os.listdir('./bad_cases/'))
        
        # if dataloader.dataset.image_path_list[batch_begin].split('/')[-1][
        #    :-4] + '_%d' % target.detach().cpu().numpy() + '.png' in os.listdir('./bad_cases/'):
        #     counter = counter + 1
        #     print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        else:
            batch_begin = batch_begin + 1
            continue
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        z_init, s_init = None, None
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in [0, 1]:
                    pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1, is_train=1)
                    # print('sample idx %d, env: %d, y: %d, pred' % (ss, env_idx, target.cpu().detach().numpy()), pred_y)
                    
                    if z_init is None:
                        z_init, s_init = z, s
                        #min_rec_loss = - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long()).item()
                        min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                              (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                              reduction='none').mean(1)
                    else:
                        #new_loss = - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long()).item()
                        new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                          (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                          reduction='none').mean(1)
                        
                        for i in range(x.size(0)):
                            if new_loss[i] < min_rec_loss[i]:
                                min_rec_loss[i] = new_loss[i]
                                z_init[i], s_init[i] = z[i], s[i]
        
        z, s = z_init, s_init
        pred_y = model.get_y(s)
        print('pred_y_init, ', pred_y)
        print('init BCE loss', min_rec_loss)
        
        z.requires_grad = True
        s.requires_grad = True
        if args.eval_optim == 'sgd':
            optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        else:
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
            BCE = BCE.mean(1)
            # for idx in range(x.size(0)):
            #     if args.use_best == 1:
            #         # print(i, cls_i, best_loss[batch_begin + idx][cls_i], loss[idx])
            #         if best_loss[batch_begin + idx] >= loss[idx].item():
            #             best_loss[batch_begin + idx] = copy.deepcopy(loss[idx].item())
            #             # print('is best ', cls_i, best_loss[batch_begin + idx][cls_i])
            #     else:
            #         best_loss[batch_begin + idx] = copy.deepcopy(loss[idx].item())
            if target.cpu().detach() == 1:
                loss = BCE.mean() - F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda()) + \
                       F.cross_entropy(pred_y, torch.zeros(x.size(0)).long().cuda())
            else:
                loss = BCE.mean() + F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda()) - \
                       F.cross_entropy(pred_y, torch.zeros(x.size(0)).long().cuda())
            if i % 50 == 0 and i > 0:
                pred_y = model.get_y(s)
                print('%d, y: %d, loss is %0.4f, s norm: %0.4f' % (
                i, target.cpu().detach().numpy(), loss.item(), torch.norm(s)), pred_y)
            loss.backward()
            optimizer.step()
        _, pred_y = model.get_x_y(z, s)
        print('final pred', pred_y)
        print('final BCE', BCE.mean())
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
        
        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    # print('counter', counter)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f' % (accuracy_init.avg, accuracy.avg))
    # print(error_index)
    return pred, accuracy.avg


def evaluate_xy_true(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    all_loss = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    pred_pos_num = 0
    error_index = []
    counter = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        # print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        # if dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_00741_0.10_0.png' or \
        #         dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_04289_0.10_0.png' or \
        #         dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_03365_0.10_0.png' or \
        #         dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_02654_0.10_0.png':
        #     print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        if dataloader.dataset.image_path_list[batch_begin].split('/')[-1] in ['img_07783_0.10_0.png']:#['img_02732_0.10_0.png']:#['img_02610_0.10_0.png']:#['img_02559_0.10_0.png']:
            print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        #     pass
        # print(os.listdir('./bad_cases/'))
        
        # if dataloader.dataset.image_path_list[batch_begin].split('/')[-1][
        #    :-4] + '_%d' % target.detach().cpu().numpy() + '.png' in os.listdir('./bad_cases/'):
        #     counter = counter + 1
        #     print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        else:
            batch_begin = batch_begin + x.size(0)
            continue
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        z_init, s_init = None, None
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in [0]:
                    for cls_i in range(args.num_classes):
                        pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1, is_train=1)
        
                        if z_init is None:
                            z_init, s_init = z, s
                            # min_rec_loss =  - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long())
                            bce_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                              (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                              reduction='none').mean(1)
                            cls_loss = F.nll_loss(torch.log(pred_y), torch.ones(pred_y.size(0), ).long().cuda() * cls_i,
                                                  reduction='none')
                            min_rec_loss = bce_loss + cls_loss
                            print('sample idx %d, env: %d, y: %d, pred' % (ss, env_idx, target.cpu().detach().numpy()),
                                  pred_y, 'BCE loss: ', bce_loss.mean(), 'cls loss: ', cls_loss.mean(), 'all loss:',
                                  min_rec_loss)
                            all_loss.update(min_rec_loss.mean(), x.size(0))
                        else:
                            # new_loss = - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long()).item()
                            bce_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                              (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                              reduction='none').mean(1)
                            cls_loss = F.nll_loss(torch.log(pred_y), torch.ones(pred_y.size(0), ).long().cuda() * cls_i,
                                                  reduction='none')
                            new_loss = bce_loss + cls_loss
                            print('sample idx %d, env: %d, y: %d, pred' % (ss, env_idx, target.cpu().detach().numpy()),
                                  pred_y, 'BCE loss: ', bce_loss.mean(), 'cls loss: ', cls_loss.mean(), 'all loss:',
                                  new_loss)
                            all_loss.update(new_loss.mean(), x.size(0))
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]
                                    min_cls = copy.deepcopy(cls_i)
        print('overall min loss:', min_rec_loss, 'mean loss: ', all_loss.avg, 'choose cls, ', min_cls)
        pred_y = model.get_y(s_init)
        print('pred_y_init, ', pred_y)
        
        for cls_i in range(args.num_classes):
            z, s = z_init.clone(), s_init.clone()
            # print(s)
            z.requires_grad = True
            s.requires_grad = True
            if args.eval_optim == 'sgd':
                optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
            else:
                optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
            
            for i in range(args.test_ep):
                if i == args.test_ep // 2:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.1
                optimizer.zero_grad()
                recon_x, pred_y = model.get_x_y(z, s)
                
                if 'mnist' in args.dataset:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                                                 reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 256 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 256 ** 2),
                                                 reduction='none')
                loss = BCE.mean(1) + F.nll_loss(torch.log(pred_y), torch.ones(x.size(0)).long().cuda() * cls_i)
                # if cls_i == 1:
                #     if np.abs(loss.item() - before_loss) <= 0.001:
                #         break
                # for idx in range(x.size(0)):
                #     if args.use_best == 1:
                #         # print(i, cls_i, best_loss[batch_begin + idx][cls_i], loss[idx])
                #         if best_loss[batch_begin + idx] >= loss[idx].item():
                #             best_loss[batch_begin + idx] = copy.deepcopy(loss[idx].item())
                #             # print('is best ', cls_i, best_loss[batch_begin + idx][cls_i])
                #     else:
                #         best_loss[batch_begin + idx] = copy.deepcopy(loss[idx].item())
                # if target.cpu().detach() == 0:
                #     loss = BCE.mean() - F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda()) + \
                #            F.cross_entropy(pred_y, torch.zeros(x.size(0)).long().cuda())
                # else:
                #     loss = BCE.mean() + F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda()) - \
                #            F.cross_entropy(pred_y, torch.zeros(x.size(0)).long().cuda())
                if i % 50 == 0 and i > 0:
                    pred_y = model.get_y(s)
                    print('%d, y: %d, loss is %0.4f, s norm: %0.4f, BCE_loss: %0.4f' % (
                        i, target.cpu().detach().numpy(), loss.item(), torch.norm(s), BCE.mean()), pred_y)
                loss.backward()
                optimizer.step()
                
            print(cls_i, 'loss: ', loss.item(), 'final BCE', BCE.mean(), 'cls loss', F.nll_loss(torch.log(pred_y), torch.ones(x.size(0)).long().cuda() * cls_i), 'final pred', pred_y)
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
        
        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    # print('counter', counter)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f' % (accuracy_init.avg, accuracy.avg))
    # print(error_index)
    return pred, accuracy.avg


def evaluate_xy_true_full(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    all_loss = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    pred_pos_num = 0
    error_index = []
    counter = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        print('batch_begin', batch_begin)
        # print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        # if dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_00741_0.10_0.png' or \
        #         dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_04289_0.10_0.png' or \
        #         dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_03365_0.10_0.png' or \
        #         dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_02654_0.10_0.png':
        #     print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        # if dataloader.dataset.image_path_list[batch_begin].split('/')[-1] in [
        #     'img_07783_0.10_0.png']:  # ['img_02732_0.10_0.png']:#['img_02610_0.10_0.png']:#['img_02559_0.10_0.png']:
        #     print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        #     pass
        # print(os.listdir('./bad_cases/'))
        
        # if dataloader.dataset.image_path_list[batch_begin].split('/')[-1][
        #    :-4] + '_%d' % target.detach().cpu().numpy() + '.png' in os.listdir('./bad_cases/'):
        #     counter = counter + 1
        #     print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        # else:
        #     batch_begin = batch_begin + x.size(0)
        #     continue
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        z_init, s_init = None, None
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in [0]:
                    for cls_i in range(args.num_classes):
                        pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1, is_train=1)
                        
                        if z_init is None:
                            z_init, s_init = z, s
                            # min_rec_loss =  - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long())
                            bce_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                              (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                              reduction='none').mean(1)
                            cls_loss = F.nll_loss(torch.log(pred_y), torch.ones(pred_y.size(0), ).long().cuda() * cls_i,
                                                  reduction='none')
                            min_rec_loss = bce_loss + cls_loss
                            # print('sample idx %d, env: %d, y: %d, pred' % (ss, env_idx, target.cpu().detach().numpy()),
                            #       pred_y, 'BCE loss: ', bce_loss.mean(), 'cls loss: ', cls_loss.mean(), 'all loss:',
                            #       min_rec_loss)
                            all_loss.update(min_rec_loss.mean(), x.size(0))
                        else:
                            # new_loss = - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long()).item()
                            bce_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                                                              (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                                                              reduction='none').mean(1)
                            cls_loss = F.nll_loss(torch.log(pred_y), torch.ones(pred_y.size(0), ).long().cuda() * cls_i,
                                                  reduction='none')
                            new_loss = bce_loss + cls_loss
                            # print('sample idx %d, env: %d, y: %d, pred' % (ss, env_idx, target.cpu().detach().numpy()),
                            #       pred_y, 'BCE loss: ', bce_loss.mean(), 'cls loss: ', cls_loss.mean(), 'all loss:',
                            #       new_loss)
                            all_loss.update(new_loss.mean(), x.size(0))
                            for i in range(x.size(0)):
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]
                                    min_cls = copy.deepcopy(cls_i)
        # print('overall min loss:', min_rec_loss, 'mean loss: ', all_loss.avg, 'choose cls, ', min_cls)
        pred_y = model.get_y(s_init)
        # print('pred_y_init, ', pred_y)
        
        for cls_i in range(args.num_classes):
            z, s = z_init.clone(), s_init.clone()
            # print(s)
            z.requires_grad = True
            s.requires_grad = True
            if args.eval_optim == 'sgd':
                optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
            else:
                optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
            
            for i in range(args.test_ep):
                if i == args.test_ep // 2:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.1
                
                recon_x, pred_y = model.get_x_y(z, s)
                
                if 'mnist' in args.dataset:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 28 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 28 ** 2),
                                                 reduction='none')
                else:
                    BCE = F.binary_cross_entropy(recon_x.view(-1, 3 * 256 ** 2), (x * 0.5 + 0.5).view(-1, 3 * 256 ** 2),
                                                 reduction='none')
                loss = BCE.mean(1) + F.nll_loss(torch.log(pred_y), torch.ones(x.size(0)).long().cuda() * cls_i, reduction='none')
                for idx in range(x.size(0)):
                    if best_loss[batch_begin + idx][cls_i] >= BCE.mean(1)[idx].item():
                        best_loss[batch_begin + idx][cls_i] = copy.deepcopy(BCE.mean(1)[idx].item())
                #print(loss, BCE.mean(1), F.nll_loss(torch.log(pred_y), torch.ones(x.size(0)).long().cuda() * cls_i))
                optimizer.zero_grad()
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # if cls_i == 1:
                #     if np.abs(loss.item() - before_loss) <= 0.001:
                #         break
                
                # if target.cpu().detach() == 0:
                #     loss = BCE.mean() - F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda()) + \
                #            F.cross_entropy(pred_y, torch.zeros(x.size(0)).long().cuda())
                # else:
                #     loss = BCE.mean() + F.cross_entropy(pred_y, torch.ones(x.size(0)).long().cuda()) - \
                #            F.cross_entropy(pred_y, torch.zeros(x.size(0)).long().cuda())
                # if i % 50 == 0 and i > 0:
                #     pred_y = model.get_y(s)
                #     print('%d, y: %d, loss is %0.4f, s norm: %0.4f, BCE_loss: %0.4f' % (
                #         i, target.cpu().detach().numpy(), loss.item(), torch.norm(s), BCE.mean()), pred_y)
                
            
            # print(cls_i, 'loss: ', loss.item(), 'final BCE', BCE.mean(), 'cls loss',
            #       F.nll_loss(torch.log(pred_y), torch.ones(x.size(0)).long().cuda() * cls_i), 'final pred', pred_y)
        # for i in range(x.size(0)):
        #     if np.argmax(np.array(pred_y[i].detach().cpu().numpy())) != target[i].detach().cpu().numpy():
        #         if not os.path.exists('./bad_cases/'):
        #             os.makedirs('./bad_cases/')
        #         filename = dataloader.dataset.image_path_list[batch_begin+i].split('/')[-1]
        #         print('tagret file, ', filename[:-4]+'_%d'%target[i].detach().cpu().numpy()+'.png')
        #         copyfile(dataloader.dataset.image_path_list[batch_begin+i], os.path.join('./bad_cases/', filename[:-4]+'_%d'%target[i].detach().cpu().numpy()+'.png'))
        # error_index.append(dataloader.dataset.image_path_list[batch_begin+i])
        accuracy.update(compute_acc(-1 * best_loss[batch_begin:batch_begin+x.size(0), :].
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()))
        
        batch_begin = batch_begin + x.size(0)
        args.logger.info('batch %d, init_acc: %0.4f, after acc: %0.4f' % (batch_begin, accuracy_init.avg, accuracy.avg))
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    # print('counter', counter)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f' % (accuracy_init.avg, accuracy.avg))
    # print(error_index)
    return pred, accuracy.avg


def evaluate_xy_t(epoch, model, dataloader, args):
    model.eval()
    model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    best_loss = 10000 * np.ones((dataloader.dataset.__len__(), 1))
    batch_begin = 0
    pred_pos_num = 0
    error_index = []
    counter = 0
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        # print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        if dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_00741_0.10_0.png' or \
                dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_04289_0.10_0.png' or \
                dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_03365_0.10_0.png' or \
                dataloader.dataset.image_path_list[batch_begin].split('/')[-1] == 'img_02654_0.10_0.png':
            print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
            pass
        # print(os.listdir('./bad_cases/'))
        # if dataloader.dataset.image_path_list[batch_begin].split('/')[-1][
        #    :-4] + '_%d' % target.detach().cpu().numpy() + '.png' in os.listdir('./bad_cases/'):
        #     counter = counter + 1
        #     print(dataloader.dataset.image_path_list[batch_begin].split('/')[-1])
        else:
            batch_begin = batch_begin + 1
            continue
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        z_init, s_init = None, None
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in [0, 1]:
                    pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1, is_train=1)
                    # print('sample idx %d, env: %d, y: %d, pred' % (ss, env_idx, target.cpu().detach().numpy()), pred_y)
                    
                    if z_init is None:
                        z_init, s_init = z, s
                        min_rec_loss = - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long()).item()
                        # min_rec_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                        #                                       (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                        #                                       reduction='none').mean(1)
                    else:
                        new_loss = - F.cross_entropy(pred_y, torch.argmax(pred_y, dim=1).long()).item()
                        # new_loss = F.binary_cross_entropy(recon_x.view(-1, 3 * args.image_size ** 2),
                        #                                   (x * 0.5 + 0.5).view(-1, 3 * args.image_size ** 2),
                        #                                   reduction='none').mean(1)

                        for i in range(x.size(0)):
                            if x.size(0) == 1:
                                if new_loss < min_rec_loss:
                                    min_rec_loss = new_loss
                                    z_init[i], s_init[i] = z[i], s[i]
                            else:
                                if new_loss[i] < min_rec_loss[i]:
                                    min_rec_loss[i] = new_loss[i]
                                    z_init[i], s_init[i] = z[i], s[i]
        
        z, s = z_init, s_init
        pred_y = model.get_y(s)
        print('end init', pred_y)
        z.requires_grad = True
        s.requires_grad = True
        if args.eval_optim == 'sgd':
            optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        else:
            optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        
        # for cls_i in range()
        for i in range(args.test_ep):
            optimizer.zero_grad()
            recon_x, pred_y = model.get_x_y(z, s)
            
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
            if i % 10 == 0 and i > 0:
                pred_y = model.get_y(s)
                print('%d, y: %d, loss is %0.4f, s norm: %0.4f' % (
                i, target.cpu().detach().numpy(), loss.item(), torch.norm(s)), pred_y)
            loss.backward()
            optimizer.step()
        _, pred_y = model.get_x_y(z, s)
        print('final pred_y', pred_y)
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
        
        batch_begin = batch_begin + x.size(0)
    # args.logger.info('pred_pos_sample: %d' % pred_pos_num)
    # print('counter', counter)
    args.logger.info('init_acc: %0.4f, after acc: %0.4f' % (accuracy_init.avg, accuracy.avg))
    # print(error_index)
    return pred, accuracy.avg


def evaluate_xy_full(epoch, model, dataloader, args):
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
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        z_init, s_init = None, None
        # print('batch_begin', batch_begin)
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in [0, 1]:
                    pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1, is_train=1)
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
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()))
        z.requires_grad = True
        s.requires_grad = True
        if args.eval_optim == 'sgd':
            optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        else:
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
    args.logger.info('init_acc: %0.4f, after acc: %0.4f, BCE loss: %0.4f' % (accuracy_init.avg, accuracy.avg, BCE_loss.avg))
    
    # print(error_index)
    return pred, accuracy.avg


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
    total_acc = [AverageMeter() for i in range(args.test_ep+1)]
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        if args.cuda:
            x, target, env, u = x.cuda(), target.cuda().long(), env.cuda().long(), u.cuda()
        z_init, s_init = None, None
        # print('batch_begin', batch_begin)
        with torch.no_grad():
            for ss in range(args.sample_num):
                for env_idx in [0, 1]:
                    pred_y, recon_x, mu, logvar, z, s, zs = model(x, env_idx, feature=1, is_train=1)
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
                                         reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        total_acc[0].update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                         reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()), x.size(0))
        z.requires_grad = True
        s.requires_grad = True
        if args.eval_optim == 'sgd':
            optimizer = optim.SGD(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        else:
            optimizer = optim.Adam(params=[z, s], lr=args.lr2, weight_decay=args.reg2)
        
        for i in range(args.test_ep):
            optimizer.zero_grad()
            recon_x, pred_y = model.get_x_y(z, s)

            total_acc[i+1].update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
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
        pred_y_init, pred_y = model(x, is_train = 0, is_debug=1)
        pred_pos_num = pred_pos_num + np.where(np.argmax(np.array(pred_y.detach().cpu().numpy()). \
                                              reshape((x.size(0), args.num_classes)), axis=1) == 1)[0].shape[0]
        accuracy_init.update(compute_acc(np.array(pred_y_init.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()))
        accuracy.update(compute_acc(np.array(pred_y.detach().cpu().numpy()).
                                    reshape((x.size(0), args.num_classes)), target.detach().cpu().numpy()))

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
                model = Generative_model_f_2D_unpooled_env_t_mnist(in_channel=args.in_channel,
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
    pred_test, test_acc = evaluate_xy_true(epoch, model, test_loader, args)
    # pred_test, test_acc = evaluate_xy(epoch, model, test_loader, args)
    # pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
    if test_acc >= best_acc:
        best_acc = copy.deepcopy(test_acc)
        best_acc_ep = copy.deepcopy(epoch)
        is_best = 1
        logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
                    % (test_acc, args.test_ep, args.lr2, args.reg2, args.sample_num))

    # for test_ep in [50, 80, 100, 200]:
    #     for lr2 in [0.0005, 0.001, 0.0001]:
    #         for wd2 in [0.005, 0.001, 0.0001]:
    #             for sample_num in [5, 10]:
    #                 temp_args = copy.deepcopy(args)
    #                 temp_args.sample_num = sample_num
    #                 temp_args.test_ep = test_ep
    #                 temp_args.lr2 = lr2
    #                 temp_args.reg2 = wd2
    #                 model.args = temp_args
    #                 logger.info('raw pred')
    #                 pred_test, test_acc = evaluate_xy_full(epoch, model, test_loader, temp_args)
    #                 logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
    #                             % (test_acc, test_ep, lr2, wd2, sample_num))
    #                 logger.info('inner pred')
    #                 pred_test, test_acc = evaluate_22(epoch, model, test_loader, temp_args)
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