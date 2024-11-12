
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from data.transforms import transform_train_2d, transform_test_2d
import torchvision.models as models
# from data.dataset import CalImageFolder
from data.STAS_dataloader import Dataloader_2D
from .utils import generate_random_sampler_for_train
from .train import train
from .valid import validate
from .train_baseline import train_baseline
from .valid_baseline import validate_baseline
from tensorboardX import SummaryWriter
from .utils import adjust_learning_rate
from .save_checkpoint import save_checkpoint
# from models.Metatrainer import DarMo
from models.Maintrainer_stas import Model_STAS_2D, Model_STAS_2D_bs
import torchvision.transforms as transforms
import copy



def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_acc2, best_acc3, best_acc4, best_acc5, best_acc6, best_acc7, best_acc8, best_auc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    pretrained_dict = {}
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model_pre = models.__dict__[args.arch](pretrained=True)
        pretrained_dict = model_pre.state_dict()
    else:
        print("=> creating model '{}' without pretraining".format('args.arch') )

    # model = DarMo(3, 256, args.gd)
    if not args.backbone:
        model = Model_STAS_2D(1, 64, "adjacency_matrix.pkl", args)
    else:
        model = Model_STAS_2D_bs(1, 64, 2, args.if_mmf, args)
    
    model_dict = model.state_dict()
    if len(pretrained_dict) != 0:
        pretrained_dict1 = {'Resnetencoder.ResNet34.' + k: v for k, v in pretrained_dict.items() if
                            'Resnetencoder.ResNet34.' + k in model_dict}
        model_dict.update(pretrained_dict1)
        model.load_state_dict(model_dict)



    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            # model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model.cuda()

    # define loss function (criterion) and optimizer
    if not args.multilabel:
        criterion_cls = nn.CrossEntropyLoss().cuda()
        criterion_gcn = nn.CrossEntropyLoss().cuda()
    else:
        criterion_cls = nn.CrossEntropyLoss().cuda()
        criterion_gcn = nn.MultiLabelSoftMarginLoss().cuda()

    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.mask_only:
        dataset_dir = "data/STAS_dataset_2D_maskOnly.xls"
    else:
        if args.if_randomTest :
            dataset_dir = "data/STAS_dataset_2D_randomTest-82.xls"
        else:
            dataset_dir = "data/STAS_dataset_2D.xls"

    

    train_dataset = Dataloader_2D(
        dataset_dir,
        "train",
        transforms.Compose([
            transforms.RandomRotation(60),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],
                                        std=[0.5]),
                            ]))


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    bac_sampler = generate_random_sampler_for_train(torch_dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, batch_sampler=bac_sampler)

    val_loader_inter = torch.utils.data.DataLoader(
        Dataloader_2D(dataset_dir, 
                      "test_inter", 
                      transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5],
                                                    std=[0.5]),
                                    ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    val_loader_exter = torch.utils.data.DataLoader(
        Dataloader_2D(dataset_dir, 
                      "test_exter", 
                      transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5],
                                                    std=[0.5]),
                                    ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # if args.evaluate:
    #     print ('evaluating')
    #     validate(val_loader, model, criterion_cls, criterion_gcn, valdir, args)
    #     return


    name_inter = "None" + '/' + str(args.saved) + '_inter_'+ str(args.if_transformer) + '_' + str(args.batch_size) \
                + '_' + str(args.lr) + '_' + str(args.epochs) + '_model_'
    name_exter = "None" + '/' + str(args.saved) + '_exter_'+ str(args.if_transformer) + '_' + str(args.batch_size) \
                + '_' + str(args.lr) + '_' + str(args.epochs) + '_model_'
    writer_root='./'+str(args.saved) + '_'+ str(args.if_transformer)+ '/'+'runs/'+ str(args.batch_size) + '_' + str(args.lr)
    writer = SummaryWriter(writer_root)

    best_auc_train = 0
    best_acc_train = 0
    best_auc_train_ep = 0

    best_auc_inter = 0
    best_acc_inter = 0
    best_auc_inter_ep = 0

    best_auc_exter = 0
    best_acc_exter = 0
    best_auc_exter_ep = 0
    best_sensi_train = 0
    best_speci_train = 0
    best_sensi_inter = 0
    best_speci_inter = 0
    best_sensi_exter = 0
    best_speci_exter = 0

    best_auc_inter_init = 0
    best_auc_exter_init = 0

    auc_inter_init = 0
    auc_exter_init = 0

    from datetime import datetime
 
    # 获取当前日期和时间
    now = datetime.now()
    
    # 打印日期和时间，格式为YYYY-MM-DD HH:MM
    timestr = now.strftime('%Y-%m-%d-%H-%M-%S')
    try:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, epoch, args)
            
            if not args.backbone:
                # train for one epoch
                acc_train, auc_train, sensi_train, speci_train = train(train_loader, model, criterion_cls, criterion_gcn, optimizer, epoch, args)

                # evaluate on validation set
                acc_inter, auc_inter, sensi_inter, speci_inter, auc_inter_init = validate(val_loader_inter, model, criterion_cls, criterion_gcn, args)
                acc_exter, auc_exter, sensi_exter, speci_exter, auc_exter_init = validate(val_loader_exter, model, criterion_cls, criterion_gcn, args)
            else:
                # train for one epoch
                acc_train, auc_train, sensi_train, speci_train = train_baseline(train_loader, model, criterion_cls, criterion_gcn, optimizer, epoch, args)

                # evaluate on validation set
                acc_inter, auc_inter, sensi_inter, speci_inter = validate_baseline(val_loader_inter, model, criterion_cls, criterion_gcn, args)
                acc_exter, auc_exter, sensi_exter, speci_exter  = validate_baseline(val_loader_exter, model, criterion_cls, criterion_gcn, args)
            
            # remember best acc@1 and save checkpoint
            is_best_auc_inter = 0
            is_best_auc_exter = 0

            if auc_train >= best_auc_train:
                best_auc_train = copy.deepcopy(auc_train)
                best_acc_train = copy.deepcopy(acc_train)
                best_sensi_train = copy.deepcopy(sensi_train)
                best_speci_train = copy.deepcopy(speci_train)
                best_auc_train_ep = copy.deepcopy(epoch)
                # is_best_train = 1
                print("Train*"*6)
                print('Best AUC: %0.4f, sensitivity: %0.4f, specificity %0.4f, acc: %0.4f, at Best_auc_ep: %d. '
                            % (best_auc_train,best_sensi_train, best_speci_train, best_acc_train, best_auc_train_ep))
                print("Train*"*6)

            if auc_inter >= best_auc_inter:
                best_auc_inter = copy.deepcopy(auc_inter)
                best_acc_inter = copy.deepcopy(acc_inter)
                best_sensi_inter = copy.deepcopy(sensi_inter)
                best_speci_inter = copy.deepcopy(speci_inter)
                best_auc_inter_ep = copy.deepcopy(epoch)
                best_auc_inter_init = copy.deepcopy(auc_inter_init)
                # is_best_train = 1
                print("Inter*"*6)
                print('Best AUC: %0.4f, sensitivity: %0.4f, specificity %0.4f, acc: %0.4f, at Best_auc_ep: %d. ********AUC_init: %0.4f.'
                            % (best_auc_inter, best_sensi_inter, best_speci_inter,  best_acc_inter, best_auc_inter_ep, best_auc_inter_init))
                print("Inter*"*6)
            # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            #                                             and args.rank % ngpus_per_node == 0):
            # if wo need to save checkpoint, uncomment the following!
            # if wo need to save checkpoint, uncomment the following!
            # if wo need to save checkpoint, uncomment the following!
                # save_checkpoint({
                #     'epoch': epoch + 1,
                #     'arch': args.arch,
                #     'state_dict': model.state_dict(),
                #     'acc': best_acc_inter,
                #     'auc': best_auc_inter,
                #     'optimizer': optimizer.state_dict(),
                # }, name_inter, epoch, args.epochs, is_best, timestr)

                if auc_exter >= best_auc_exter:
                    best_auc_exter = copy.deepcopy(auc_exter)
                    best_acc_exter = copy.deepcopy(acc_exter)
                    best_sensi_exter = copy.deepcopy(sensi_exter)
                    best_speci_exter = copy.deepcopy(speci_exter)
                    best_auc_exter_ep = copy.deepcopy(epoch)
                    best_auc_exter_init = copy.deepcopy(auc_exter_init)
                    # is_best_train = 1
                    print("Exter*"*6)
                    print('Best AUC: %0.4f, sensitivity: %0.4f, specificity %0.4f, acc: %0.4f, at Best_auc_ep: %d. ********AUC_init: %0.4f.'
                                % (best_auc_exter,best_sensi_exter, best_speci_exter, best_acc_exter, best_auc_exter_ep, best_auc_exter_init))
                    print("Exter*"*6)
                # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                #                                             and args.rank % ngpus_per_node == 0):
                # if wo need to save checkpoint, uncomment the following!
                # if wo need to save checkpoint, uncomment the following!
                # if wo need to save checkpoint, uncomment the following!
                    # save_checkpoint({
                    #     'epoch': epoch + 1,
                    #     'arch': args.arch,
                    #     'state_dict': model.state_dict(),
                    #     'acc': best_acc_exter,
                    #     'auc': best_auc_exter,
                    #     'optimizer': optimizer.state_dict(),
                    # }, name_exter, epoch, args.epochs, is_best, timestr)
    finally:
        print("*_* "*20)
        print('Best AUC: %0.4f, sensitivity: %0.4f, specificity %0.4f, acc: %0.4f, at Best_auc_ep: %d. '
                            % (best_auc_train,best_sensi_train, best_speci_train, best_acc_train, best_auc_train_ep))
        print('Best AUC: %0.4f, sensitivity: %0.4f, specificity %0.4f, acc: %0.4f, at Best_auc_ep: %d. ********AUC_init: %0.4f.'
                            % (best_auc_inter, best_sensi_inter, best_speci_inter,  best_acc_inter, best_auc_inter_ep, best_auc_inter_init))
        print('Best AUC: %0.4f, sensitivity: %0.4f, specificity %0.4f, acc: %0.4f, at Best_auc_ep: %d. ********AUC_init: %0.4f.'
                                % (best_auc_exter,best_sensi_exter, best_speci_exter, best_acc_exter, best_auc_exter_ep, best_auc_exter_init))
        print("^_^ "*20)
    writer.export_scalars_to_json(writer_root+'/'+str(args.batch_size) + '_' + str(args.lr) +'_all_scalars.json')

    writer.close()