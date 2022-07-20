import os
import time
import math
import random
import argparse
import builtins
import warnings
import functools
import subprocess
builtins.print = functools.partial(print, flush=True)

import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from data import DataPrefetcher, InfiniteSampler
from data.transforms import typical_imagenet_transform
from utils import accuracy, AvgMeter, save_checkpoint, to_python_float, cal_remain_time
from utils.log import setup_writer, setup_logger
from utils.dist import reduce_tensor, synchronize


parser = argparse.ArgumentParser('LinearEvaluation')
parser.add_argument('-expn', '--experiment_name', type=str, default='baseline-')

# model
parser.add_argument('--model', type=str, default='res50', choices=['res18', 'res50', 'res101'])
parser.add_argument('--mb_kernel_size', type=int, default=3)
parser.add_argument('--width', type=int, default=1, choices=[1, 2, 4])
parser.add_argument('--cls-bn', action='store_true', default=False)
parser.add_argument('--target-encoder', action='store_true', default=False)
parser.add_argument('-sbp', '--single-branch-pretrained', action='store_true', default=False)

# optimization
parser.add_argument('-lnc', '--large-norm-config', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=30)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--milestones', nargs='+', type=int, default=[60, 80])
parser.add_argument('--weight-decay', type=float, default=0.)
parser.add_argument('--nesterov', action='store_true', default=False)
parser.add_argument('--scheduler', type=str, default='multistep', choices=['cos', 'multistep'])

# data
parser.add_argument('--data-path', type=str, default='path/to/dataset')
parser.add_argument('-j', '--num-workers', type=int, default=6)

# dir
parser.add_argument('--output_dir', type=str, default='outputs', help='path for saving trained models')
parser.add_argument('--linear-eval-name', type=str, default=None)
parser.add_argument('--resume', action='store_true', default=False)

# misc
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--total-epochs', type=int, default=100, help='total epochs')
parser.add_argument('-bs', '--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--eval-interval', type=int, default=20)
parser.add_argument('--print-interval', type=int, default=None)

# distributed
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--num-machines', default=1, type=int)
parser.add_argument('--machine-rank', default=0, type=int)
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('-md', '--multiprocessing-distributed', default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def build_model(args):
    if args.model == 'res50':
        import models.resnet as resnet
        model = resnet.resnet50(bn='vanilla', width=args.width)
        args.feat_dim = 2048
    else:
        raise ValueError('No such model.')
    return model


def build_optimizer(args, classifier):
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
    return optimizer


def load_weights(args, model, logger):
    ckpt_tar = os.path.join(args.pretrained_file_name, 'last_epoch_ckpt.pth.tar')
    ckpt = torch.load(ckpt_tar, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    missing_keys = []
    matched_state_dict = {}

    if args.single_branch_pretrained:
        key_word = ''
    else:
        if args.target_encoder:
            key_word = 'target_encoder.'
        else:
            key_word = 'online_encoder.'

    for name, param in state_dict.items():
        if name.startswith(key_word):
            name = name.replace(key_word, '')
        else:
            continue

        if name not in model.state_dict() or name.startswith('fc'):
            missing_keys.append(name)
        else:
            matched_state_dict[name] = param
    del state_dict
    pretrained_epochs = ckpt['start_epoch']

    model.load_state_dict(matched_state_dict, strict=False)
    if args.rank == 0:
        logger.info('Missing keys: {}'.format(missing_keys))
        logger.info('Model at epoch {} is loaded.'.format(pretrained_epochs))
    del matched_state_dict

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def resume(args, classifier, optimizer, logger):
    checkpoint_tar = os.path.join(args.eval_file_name, 'linear_eval_ckpt.pth.tar')
    if not os.path.isfile(checkpoint_tar):
        print('No checkpoint found at {}'.format(args.eval_file_name))

    if args.rank == 0:
        print('Loading checkpoint {}'.format(checkpoint_tar))

    # Map model to be loaded to specified single gpu.
    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(checkpoint_tar, map_location=loc)

    args.start_epoch = checkpoint['start_epoch']
    classifier.load_state_dict(checkpoint['classifier'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if args.rank == 0:
        logger.info('Loaded checkpoint {} (at epoch {})'.format(checkpoint_tar, args.start_epoch + 1))


def modify_args(args):
    # modify the args
    args.lr = args.lr * (args.batch_size / 256)

    if args.large_norm_config:
        args.lr = 0.2
        args.batch_size = 1024
        args.total_epochs = 80
        args.nesterov = True
        args.scheduler = 'cos'

    return args


def get_local_dataloader(args):
    from data.datasets import build_dataset
    train_set = build_dataset(typical_imagenet_transform(train=True), args.data_path, True, False)

    sampler = None
    batch_size = args.batch_size
    if args.world_size > 1:
        batch_size = batch_size // dist.get_world_size()
        sampler = InfiniteSampler(len(train_set), shuffle=True, seed=0, rank=args.rank, world_size=args.world_size)
    dataloader_kwargs = {'num_workers': args.num_workers, 'pin_memory': False}
    dataloader_kwargs['sampler'] = sampler
    dataloader_kwargs['batch_size'] = batch_size
    dataloader_kwargs['shuffle'] = False
    dataloader_kwargs['drop_last'] = True
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_kwargs)

    if args.rank == 0:
        eval_set = build_dataset(typical_imagenet_transform(train=False), args.data_path, False, False)
        eval_loader = torch.utils.data.DataLoader(eval_set, 100, False, num_workers=2, pin_memory=False)
        return train_loader, eval_loader
    else:
        return train_loader, None


def adjust_learning_rate(args, optimizer, iters, ITERS_PER_EPOCH):
    total_iters = ITERS_PER_EPOCH * args.total_epochs
    lr = args.lr
    if args.scheduler == 'multistep':
        milestones = [int(total_iters * milestone / args.total_epochs) for milestone in args.milestones]
        for milestone in milestones:
            lr *= 0.1 if iters >= milestone else 1.0
    elif args.scheduler == 'cos':
        lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    elif args.scheduler == 'warmcos':
        warmup_total_iters = ITERS_PER_EPOCH * args.warmup_epochs
        if iters <= warmup_total_iters:
            warmup_lr = 1e-6
            lr = (lr - warmup_lr) * iters / float(warmup_total_iters) + warmup_lr
        else:
            lr *= 0.5 * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters)))
    else:
        raise ValueError('Scheduler of CLS {} is not available'.format(args.scheduler_cls))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def run_train(args, model, classifier, optimizer, train_loader, eval_loader, logger, writer):
    best_top1 = 0
    _best_top5 = 0
    best_top1_epoch = 0

    criterion = nn.CrossEntropyLoss()

    model.eval()
    classifier.train()

    ITERS_PER_EPOCH = len(train_loader)
    prefetcher = DataPrefetcher(train_loader, single_aug=True)

    iter_time_meter = AvgMeter()
    losses = AvgMeter()
    top1 = AvgMeter()
    top5 = AvgMeter()

    for epoch in range(args.start_epoch, args.total_epochs):

        if args.rank == 0:
            logger.info('Epoch: [{}/{}]'.format(epoch+1, args.total_epochs))

        if prefetcher.next_input is None:
            if args.world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            prefetcher = DataPrefetcher(train_loader, single_aug=True)

        for i in range(ITERS_PER_EPOCH):
            iter_time = time.time()
            data_time = time.time()
            inps, targets = prefetcher.next()
            data_time = time.time() - data_time

            bs_gpu = targets.size(0)

            # forward
            with torch.no_grad():
                feat = model(inps, res5=True).detach()
                if not (args.model == 'deitS' or args.model == 'swinT'):
                    feat = torch.flatten(model.avgpool(feat), 1)
            logits = classifier(feat)
            loss = criterion(logits, targets)
            top1_train, top5_train = accuracy(logits, targets, (1, 5))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_count = epoch * ITERS_PER_EPOCH + i + 1
            lr = adjust_learning_rate(args, optimizer, iter_count, ITERS_PER_EPOCH)

            reduced_loss = reduce_tensor(loss.data)
            reduced_top1_train = reduce_tensor(top1_train)
            reduced_top5_train = reduce_tensor(top5_train)
            losses.update(to_python_float(reduced_loss), bs_gpu)
            top1.update(to_python_float(reduced_top1_train), bs_gpu)
            top5.update(to_python_float(reduced_top5_train), bs_gpu)

            synchronize()
            iter_time = time.time() - iter_time
            iter_time_meter.update(iter_time)

            remain_time = cal_remain_time(args, iter_count, iter_time_meter, ITERS_PER_EPOCH)
            if (i + 1) % args.print_interval == 0 and args.rank == 0:
                iter_speed = 1 / iter_time_meter.avg
                logger.info('\tIter: [{}/{}], Remain-Time: {}, {:.2f}it/s, Data-Time: {:.3f}, LR: {:.4f},'
                            ' Loss: {:.4f}, Top-1: {:.2f}, Top-5: {:.2f}'.format(
                    i+1, ITERS_PER_EPOCH, remain_time, iter_speed, data_time, lr, losses.avg, top1.avg, top5.avg
                ))

        if args.rank == 0:
            logger.info('\tTrain-Epoch: [{}/{}], LR: {:.4f}, Loss: {:.4f}, '
                        'Top-1: {:.2f}, Top-5: {:.2f}'.format(epoch+1, args.total_epochs, lr,
                                                              losses.avg, top1.avg, top5.avg))
            writer.add_scalar('Train/Loss', losses.avg, global_step=epoch+1)
            writer.add_scalar('Train/Top1', top1.avg, global_step=epoch+1)
            writer.add_scalar('Train/Top5', top5.avg, global_step=epoch+1)

            losses.reset()
            top1.reset()
            top5.reset()

            if (epoch + 1) % args.eval_interval == 0:
                eval_loss, eval_top1, eval_top5 = run_eval(args, model, classifier, eval_loader, criterion, logger)
                model.eval()
                classifier.train()
                logger.info('\tEval-Epoch: [{}/{}], Loss: {:.4f}, Top-1: {:.2f},'
                            ' Top-5: {:.2f}'.format(epoch+1, args.total_epochs, eval_loss, eval_top1, eval_top5))

                writer.add_scalars('Eval/Loss', {'Train': losses.avg, 'Eval': eval_loss}, global_step=epoch+1)
                writer.add_scalars('Eval/Top1', {'Train': top1.avg, 'Eval': eval_top1}, global_step=epoch+1)
                writer.add_scalars('Eval/Top5', {'Train': top5.avg, 'Eval': eval_top5}, global_step=epoch+1)

                if eval_top1 > best_top1:
                    is_best = True
                    best_top1 = eval_top1
                    _best_top5 = eval_top5
                    best_top1_epoch = epoch+1
                else:
                    is_best = False

                logger.info('\tBest Top-1 at epoch [{}/{}], Best Top-1: {:.2f},'
                            ' Top-5: {:.2f}'.format(best_top1_epoch, args.total_epochs, best_top1, _best_top5))

                save_checkpoint({
                    'start_epoch': epoch + 1,
                    'classifier': classifier.state_dict(),
                    'best_top1': best_top1,
                    '_best_top5': _best_top5,
                    'best_top1_epoch': best_top1_epoch,
                    'optimizer': optimizer.state_dict(),
                }, is_best, args.eval_file_name, 'linear_eval')
                logger.info('*' * 100)
            logger.info('')


def run_eval(args, model, classifier, eval_loader, criterion, logger, cls_fn=False):
    model.eval()
    classifier.eval()

    top1 = AvgMeter()
    top5 = AvgMeter()
    losses = AvgMeter()

    with torch.no_grad():
        pred_list, label_list = [], []
        for _, (inp, target) in enumerate(eval_loader):
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            feat = model(inp, res5=True)
            if not (args.model == 'deitS' or args.model == 'swinT'):
                feat = torch.flatten(model.avgpool(feat), 1)
            logits = classifier(feat)
            loss = criterion(logits, target)
            acc1, acc5 = accuracy(logits, target, (1, 5))
            pred_list.append(logits.argmax(dim=1).data.cpu())
            label_list.append(target.data.cpu())

            top1.update(acc1.item(), inp.size(0))
            top5.update(acc5.item(), inp.size(0))
            losses.update(loss.item(), inp.size(0))

        if cls_fn:
            avg_top1_list = []
            pred_list = torch.cat(pred_list, dim=0)
            label_list = torch.cat(label_list, dim=0)
            cls_count = 0
            for i in range(1000):
                mask = label_list.eq(i)
                cls_count += 1
                top1_cls = pred_list[mask].eq(label_list[mask])
                avg_top1_list.append(top1_cls.sum().float().div(mask.sum().float()).numpy())
            logger.info(str([float(i) for i in avg_top1_list]))

    return losses.avg, top1.avg, top5.avg


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # ------------ set environment variables for distributed training ------------------------------------- #
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.rank == -1:
        args.rank = args.gpu

    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = (args.num_machines-1) * ngpus_per_node + gpu

    init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    # make dir for experiment output
    args.pretrained_file_name = os.path.join(args.output_dir, args.experiment_name)
    if args.linear_eval_name is None:
        args.eval_file_name = os.path.join(args.pretrained_file_name, 'linear_eval')
    else:
        args.eval_file_name = os.path.join(args.pretrained_file_name, args.linear_eval_name)

    if args.rank == 0:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        if not args.resume:
            if os.path.exists(args.eval_file_name):
                raise ValueError('Experiment name conflicts.')
            else:
                os.mkdir(args.eval_file_name)
    synchronize()

    # setup the logger and writer
    logger = setup_logger(args.eval_file_name, distributed_rank=args.rank, filename='log.txt', mode='a')
    writer = setup_writer(args.eval_file_name, distributed_rank=args.rank)

    # Data loading code
    train_loader, eval_loader = get_local_dataloader(args)
    if args.print_interval is None:
        if len(train_loader) // 1000 == 1:
            args.print_interval = 200
        elif len(train_loader) // 1000 < 1:
            args.print_interval = 100
        else:
            args.print_interval = 1000

    args = modify_args(args)

    # print the argparser
    if args.rank == 0:
        logger.info('args: {}'.format(args))

    # model
    model = build_model(args)

    # load weights
    model = load_weights(args, model, logger)

    # classifier
    if args.cls_bn:
        classifier = nn.Sequential(
            nn.Linear(args.feat_dim, 1000),
            nn.BatchNorm1d(1000)
        )
    else:
        classifier = nn.Linear(args.feat_dim, 1000)

    # optimizer
    optimizer = build_optimizer(args, classifier)

    # To GPU
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    classifier = classifier.cuda(gpu)
    if ngpus_per_node > 1:
        classifier = DDP(classifier, device_ids=[gpu])

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        resume(args, classifier, optimizer, logger)

    # train
    run_train(args, model, classifier, optimizer, train_loader, eval_loader, logger, writer)
    if args.rank == 0:
        criterion = nn.CrossEntropyLoss()
        eval_loss, eval_top1, eval_top5 = run_eval(args, model, classifier, eval_loader, criterion, logger, True)
        logger.info('Linear evaluation is done.')
        logger.info('Experiment name: {}'.format(args.experiment_name))
        writer.close()


def main():
    args = parser.parse_args()

    # setup randomization
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! You may see unexpected behavior when restarting '
            'from checkpoints.'
        )

    # multi-processing
    args.multiprocessing_distributed = args.num_machines > 1

    if args.machine_rank == 0:
        master_ip = subprocess.check_output(['hostname', '--fqdn']).decode("utf-8")
        master_ip = str(master_ip).strip()
        args.dist_url = 'tcp://{}:23456'.format(master_ip)
        print('dist_url on Machine 0:', args.dist_url)

    ngpus_per_node = torch.cuda.device_count()

    if ngpus_per_node > 1:
        args.world_size = ngpus_per_node * args.num_machines
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        args.world_size = 1
        main_worker(0, ngpus_per_node, args)


if __name__ == '__main__':
    main()
