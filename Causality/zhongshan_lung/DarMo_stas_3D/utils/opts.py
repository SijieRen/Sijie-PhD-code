import argparse
# import torchvision.models as models


# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))

def parse_opt(inputs=None):
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#     parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                         choices=model_names,
#                         help='model architecture: ' +
#                              ' | '.join(model_names) +
#                              ' (default: resnet18)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit',
                    #     choices=model_names,
                        help='model architecture: ' +
                             ' | '.join('vit') +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-ph', '--pathhome', default=False, type=bool,
                        help='evaluate model on validation set')

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--trd', default=None, type=str,
                        help='parameter to use.')
    parser.add_argument('--vd', default=None, type=str,
                        help='parameter to use.')
    parser.add_argument('--gd', default=None, type=str,
                        help='parameter to use.')
    parser.add_argument('--saved', default=None, type=str,
                        help='parameter to use.')
    parser.add_argument('--para', default=None, type=str,
                        help='parameter to use.')
    parser.add_argument('--para_cls', default=1, type=int,
                        help='parameter to clasification.')
    parser.add_argument('--para_gcn', default=1, type=int,
                        help='parameter to gcn.')
    parser.add_argument('--para_recon', default=1, type=int,
                        help='parameter to reconstruction.')
    parser.add_argument('--para_kld', default=1, type=int,
                        help='parameter to kld.')
    parser.add_argument('--pra', default=None, type=str,
                        help='time for training.')
    parser.add_argument('--multilabel', default=True, type=bool,
                        help='multilabel training.')
    parser.add_argument('--adam', default=True, type=bool,
                        help='optimizer.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    
    parser.add_argument('--if_transformer', default="res", type=str,
                        help='if we use transformer as backnone structure.')
    parser.add_argument('--patch_size', default=8, type=int,
                        help='if we use transformer as backnone structure.')
    parser.add_argument('--pool', default="mean", type=str,
                        help='pool operation in transformer.')
    parser.add_argument('--mask_only', default=0, type=int,
                        help='use the mask only data or not')
    
    parser.add_argument('--backbone', default=0, type=int,
                        help='run the backbone NN as baseline or not')
    parser.add_argument('--if_mmf', default=0, type=int,
                        help='add the clinical information into baseline or not')
    parser.add_argument('--val_ep', default=20, type=int,
                        help='use the mask only data or not')
    parser.add_argument('--lr2', default=0.00001, type=float,
                        help='run the backbone NN as baseline or not')
    parser.add_argument('--wd2', default=0.0001, type=float,
                        help='add the clinical information into baseline or not')
    
    parser.add_argument('--in_channel', default=1, type=int,
                        help='run the backbone NN as baseline or not')
    parser.add_argument('--crop_size', default=64, type=int,
                        help='add the clinical information into baseline or not')
    parser.add_argument('--shift', default=8, type=int,
                        help='add the clinical information into baseline or not')
    parser.add_argument('--frame_patch_size', default=8, type=int,
                        help='patch size of frames')
    parser.add_argument('--transpose', default=1, type=int,
                        help='whether to transpose the taining data')
    parser.add_argument('--flip', default=1, type=int,
                        help='whether to flip the taining data')
    
    # lr decay lr_controler lr_decay
    parser.add_argument('--lr_decay', default=0.2, type=float,
                        help='whether to transpose the taining data')
    parser.add_argument('--lr_controler', default=30, type=int,
                        help='whether to flip the taining data')
    
    parser.add_argument('--spacing', default="050505", type=str,
                        help='spacing of testing data')
    
    # # 随机种子
    # parser.add_argument('--seed', default=88, type=int,
    #                     help='random seed for torch model')

    parser.add_argument('--if_randomTest', default=0, type=int,
                        help='use the mask only data or not')
    
    
    args = parser.parse_args()

    return args