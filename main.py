# -*- coding: utf-8 -*-

import logging
import argparse
import matplotlib as mpl

import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
# import torch.backends.cudnn as cudnn

from work.utils.utils import *
from work.model.criterion import Criterion
from work.model.grounding_model import GroundingModel
from work.engine import train_epoch, validate_epoch, test_epoch
from work.data.get_dataloader import get_train_loader, get_val_loader, get_test_loader
import warnings
mpl.use('Agg')
warnings.filterwarnings('ignore')


def getargs():

    parser = argparse.ArgumentParser(
        description='Dataloader test')
    parser.add_argument('--num_exp_tokens', default=4, type=int,
                        help='num of expression tokens of exp feature')
    parser.add_argument('--rnn_layers', default=2, type=int, help='num of lstm layers')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")  # d_model
    parser.add_argument('--size', default=512, type=int, help='image size')
    parser.add_argument('--gpu', help='gpu id, split by , ')
    parser.add_argument('--workers', default=4, type=int,
                        help='num workers for data loading')
    parser.add_argument('--nb_epoch', default=120, type=int,
                        help='training epoch')
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last CNN convolutional block")
    parser.add_argument('--stride', action='store_true',
                    help="If true, we replace stride with dilation in the last CNN convolutional block")
    parser.add_argument('--dataset', default='refcoco', type=str,
                        help='refcoco/refcoco+/refcocog/refcocog_umd/flickr/copsref')
    parser.add_argument('--enc_layers', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="size of the feedforward layers")
    parser.add_argument('--embedding_dim', default=1024, type=int)
    parser.add_argument('--rnn_hidden_dim', default=128, type=int)
    parser.add_argument('--max_query_len', default=20, type=int,
                        help="max query len")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--batch_size', default=96, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--clip_max_norm', default=40, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--data_root', type=str, default='./store/ln_data/',
                        help='path to ReferIt splits data folder')
    parser.add_argument('--split_root', type=str, default='./store/data/',
                        help='location of pre-parsed dataset info')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='checkpoint')
    parser.add_argument('--pretrain', default='', type=str, metavar='PATH',
                        help='pretrain weight')
    parser.add_argument('--optimizer', default='adamW',
                        help='optimizer: awamW, sgd, adam, RMSprop')
    parser.add_argument('--savepath', default='store', type=str,
                        help='save dir for model/logs')
    parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N',
                        help='print frequency (default: 1e3)')
    parser.add_argument('--savename', default='default', type=str,
                        help='Name head for saved model')
    parser.add_argument('--test', dest='test', default=False, action='store_true',
                        help='test mode')
    parser.add_argument('--split', default='test', type=str,
                        help='split subset for test')
    parser.add_argument('--cnn_path', default='store/pth/resnet50_detr.pth', type=str,
                        help='pretrained cnn weights')
    args = parser.parse_args()

    # refcoco/refcoco+
    args.split = 'testA' if args.dataset == 'refcoco' or args.dataset == 'refcoco+' else 'test'
    # refcocog
    args.split = 'val' if args.dataset == 'refcocog' else 'test'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = False
    cudnn.deterministic = True

    return args


def train(args):

    # log
    if args.savename == 'default':
        args.savename = f'model_{args.dataset}_batch_{args.batch_size}'

    log_path = f'{args.savepath}/logs/{args.savename}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logging.basicConfig(level=logging.DEBUG,
                        filename=f"{log_path}/log_{args.dataset}",
                        filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(args)

    # Dataset
    train_loader = get_train_loader(args)
    val_loader = get_val_loader(args)

    # model
    model = GroundingModel(args)
    model = torch.nn.DataParallel(model).cuda()
    logging.info(model)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            pretrained_dict = torch.load(args.pretrain)['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            assert (len([k for k, v in pretrained_dict.items()]) != 0)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("=> loaded pretrain model at {}".format(args.pretrain))
            logging.info("=> loaded pretrain model at {}".format(args.pretrain))
        else:
            print(("=> no pretrained file found at '{}'".format(args.pretrain)))
            logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    elif args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint (epoch {})".format(checkpoint['epoch'])))
            logging.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            logging.info(("=> no checkpoint found at '{}'".format(args.resume)))

    # optimizer
    optimizer = get_optimizer(args, model)

    # get criterion
    criterion = Criterion(args)
    best_accu = -float('Inf')

    # train
    for epoch in range(args.nb_epoch):
        adjust_learning_rate(optimizer, epoch, optimizer.param_groups[0]['lr'])
        model.train()
        train_epoch(args, train_loader, model, optimizer, epoch, criterion, args.size)
        model.eval()
        accu_new, miou_new = validate_epoch(args, val_loader, model, epoch, args.size)

        is_best = accu_new > best_accu
        best_accu = max(accu_new, best_accu)
        # save the pth
        save_checkpoint(args,
                        {'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'acc': accu_new,
                         'optimizer': optimizer.state_dict()},
                        is_best,
                        epoch + 1,
                        filename=args.savename)

    print(f'Best Acc: {best_accu}.')


def val(args):

    # Dataset
    train_loader = get_train_loader(args)
    val_loader = get_val_loader(args)

    # model
    model = GroundingModel(args)
    model = torch.nn.DataParallel(model).cuda()
    logging.info(model)

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            pretrained_dict = torch.load(args.pretrain)['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            assert (len([k for k, v in pretrained_dict.items()]) != 0)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("=> loaded pretrain model at {}".format(args.pretrain))
            logging.info("=> loaded pretrain model at {}".format(args.pretrain))
        else:
            print(("=> no pretrained file found at '{}'".format(args.pretrain)))
            logging.info("=> no pretrained file found at '{}'".format(args.pretrain))
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint (epoch {}) Loss{}".format(checkpoint['epoch'], best_loss)))
            logging.info("=> loaded checkpoint (epoch {}) Loss{}".format(checkpoint['epoch'], best_loss))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
            logging.info(("=> no checkpoint found at '{}'".format(args.resume)))


    model.eval()
    accu_new, miou_new = validate_epoch(args, val_loader, model, 0, args.size)

    print(f'accu: {accu_new}, miou: {miou_new}. ')


def test(args):

    # Dataset
    if args.batch_size != 1:
        warnings.warn('metrics may not correct!', Warning)

    test_loader = get_test_loader(args, split=args.split)

    # model
    model = GroundingModel(args)
    model = torch.nn.DataParallel(model).cuda()

    assert args.pretrain is not None
    if os.path.isfile(args.pretrain):
        pretrained_dict = torch.load(args.pretrain)['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()]) != 0)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded pretrain model at {}".format(args.pretrain))
        logging.info("=> loaded pretrain model at {}".format(args.pretrain))
    else:
        print(("=> no pretrained file found at '{}'".format(args.pretrain)))
        logging.info("=> no pretrained file found at '{}'".format(args.pretrain))

    model.eval()
    test_epoch(test_loader, model, args.size)


if __name__ == "__main__":

    args = getargs()
    if args.test:
        print('Starting Test....')
        test(args)
    else:
    	print('Starting Training....')
        train(args)
        