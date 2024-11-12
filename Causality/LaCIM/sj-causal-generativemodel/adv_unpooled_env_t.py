# coding:utf8
from __future__ import print_function
import torch.optim as optim
from utils import *
from utils import get_dataset_2D_env as get_dataset_2D
from torchvision import transforms
from models import *
import torch.nn.functional as F
import torch.nn as nn
import foolbox
from foolbox.gradient_estimators import CoordinateWiseGradientEstimator as CWGE
import imageio

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
            cls_loss_t = F.cross_entropy(pred_y, target[env == ss])
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

def FGSM_attack_GE(img_helper, grads, epsilon=0.3):
    sign_data_grad = grads.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = img_helper + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def adversarial_attack(model, fmodel, dataloader, args, epsilon=0.3):
    # model.eval()
    # model.zero_grad()
    accuracy = AverageMeter()
    accuracy_init = AverageMeter()
    grad_time = AverageMeter()
    pred = np.zeros((dataloader.dataset.__len__(), args.num_classes))
    batch_begin = 0
    pred_pos_num = 0
    GE = CWGE(1.)
    image_counter = 0
    
    #with torch.no_grad():
    for batch_idx, (x, target, env, u) in enumerate(dataloader):
        for i in range(x.size(0)):
            img_ori = x[i]
            img_helper = img_ori.clone().detach()
            img_helper.requires_grad = False
            # print(img_ori.size())
            # print(img_ori.numpy().transpose((1,2,0)).reshape(28,28,3))
            # start_time = time.time()
            logits = fmodel.predictions(img_ori.numpy())
            accuracy_init.update(compute_acc(logits.
                                        reshape((x.size(0), args.num_classes)), target[i].detach().cpu().numpy()))
            # print('out inference time: ', time.time() - start_time)
            # start_time = time.time()
            
            fliped_label = 1 - np.argmax(logits)
            # print(fmodel.batch_predictions)
            # print(img_ori.size())
            
            start_time = time.time()
            grads = GE(fmodel.batch_predictions, img_ori.numpy(), fliped_label, (0, 1))
            grad_time.update(time.time() - start_time, 1)
            # print('out compute grad time: ', time.time() - start_time)
            # start_time = time.time()
            #opti = torch.optim.SGD([img_helper], lr=1, momentum=0.95)
            img_attack = FGSM_attack_GE(img_helper, torch.from_numpy(grads), epsilon=epsilon)
            # print('attack time: ', time.time() - start_time)
            img_attack = img_attack.numpy()
            #print(img_attack.min(), img_attack.max())
            # save the attack image
            if not os.path.exists('./adv/FGSM_%0.3f_%d_%d/'%(epsilon, args.sample_num, args.test_ep)):
                os.makedirs('./adv/FGSM_%0.3f_%d_%d/'%(epsilon, args.sample_num, args.test_ep))
            
            imageio.imwrite('./adv/FGSM_%0.3f_%d_%d/%s'%(epsilon, args.sample_num, args.test_ep, dataloader.dataset.image_path_list[image_counter].split('/')[-1]),
                            (img_attack.transpose(1,2,0) * 255).astype('uint8'))
            #print('./adv/FGSM_%0.3f/%s'%(epsilon, dataloader.dataset.image_path_list[image_counter].split('/')[-1]))
            pred_y = fmodel.predictions(img_attack)
            
            accuracy.update(compute_acc(pred_y.
                                        reshape((x.size(0), args.num_classes)), target[i].detach().cpu().numpy()))
            image_counter = image_counter + 1
            if image_counter % 10== 0 and image_counter > 0:
                print('%d/%d'%(image_counter, 10000), accuracy_init.avg, accuracy.avg, 'grad time: ', grad_time.avg)
        batch_begin = batch_begin + x.size(0)
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

    if args.model == 'unpooled':
        model = Generative_model_f_2D_unpooled_env_t(
            in_channel=args.in_channel,
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
            is_cuda=args.is_cuda,
        )
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
    epoch = 1
    model.eval()
    if args.is_cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    for i, p in enumerate(model.parameters()):
        p.requires_grad = False
    fmodel = foolbox.models.PyTorchModel(model,  # return logits in shape (bs, n_classes)
                                         bounds=(0., 1.),
                                         num_classes=args.num_classes,
                                         cuda=args.is_cuda)
    _, best_acc = adversarial_attack(model, fmodel, test_loader, args, epsilon=args.epsilon)
    # pred_test, test_acc = evaluate_22(epoch, model, test_loader, args)
    # if test_acc >= best_acc:
    #     best_acc = copy.deepcopy(test_acc)
    #     best_acc_ep = copy.deepcopy(epoch)
    #     is_best = 1
    #     logger.info('test acc: %0.4f, test_ep: %d, lr2: %0.5f, wd2: %0.5f, sample %d'
    #                 % (test_acc, args.test_ep, args.lr2, args.reg2, args.sample_num))

    # for test_ep in [80]:
    #     for lr2 in [0.0005]:
    #         for wd2 in [0.005]:
    #             for sample_num in [10]:
    #                 temp_args = copy.deepcopy(args)
    #                 temp_args.sample_num = sample_num
    #                 temp_args.test_ep = test_ep
    #                 temp_args.lr2 = lr2
    #                 temp_args.reg2 = wd2
    #                 model.args = temp_args
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