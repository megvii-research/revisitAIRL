import os
import sys
import random
import builtins
import warnings
import importlib
import subprocess

import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from data import DataPrefetcher
from utils import resume, get_state_dict, save_checkpoint
from utils.log import setup_logger, setup_writer
from utils.dist import synchronize
from utils.default_argparse import default_argument_parser


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


    # get the Exp file
    if not args.exp_file:
        print('Exp file missing.')
        sys.exit(1)
    else:
        sys.path.insert(0, os.path.dirname(args.exp_file))
        current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])
        trainer = current_exp.Trainer()

    # update some config if necessary
    updated_config = trainer.update(args.exp_options)

    # make dir for experiment output
    file_name = os.path.join(args.output_dir, args.experiment_name)
    if args.rank == 0:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        if not args.resume:
            if os.path.exists(file_name):
                raise ValueError('Experiment name conflicts.')
            else:
                os.mkdir(file_name)
    synchronize()

    # setup the logger and writer
    logger = setup_logger(file_name, distributed_rank=args.rank, filename='train_log.txt', mode='a')
    writer = setup_writer(file_name, distributed_rank=args.rank)

    # setup model, dataloader and optimizer
    trainer.build_dataloader(args)
    trainer.build_model()
    trainer.build_optimizer(args)

    if args.rank == 0:
        logger.info('args: {}'.format(args))
        hyper_param = []
        for k in trainer.__dict__:
            if 'model' not in k:
                hyper_param.append(str(k) + '=' + str(trainer.__dict__[k]))
        logger.info('Hyper-parameters: {}'.format(', '.join(hyper_param)))

        if updated_config:
            logger.opt(ansi=True).info("List of override configs:\n<blue>{}</blue>\n".format(updated_config))

    if args.rank == 0:
        logger.info('Model: ')
        logger.info(str(trainer.model))

    # put the model onto gpu
    torch.cuda.set_device(gpu)
    trainer.model.cuda(gpu)
    if trainer.CLS:
        trainer.classifier.cuda(gpu)
    if ngpus_per_node > 1:
        trainer.model = DDP(trainer.model, device_ids=[gpu])
        if trainer.CLS:
            trainer.classifier = DDP(trainer.classifier, device_ids=[gpu])

    cudnn.benchmark = True

    # resume
    if args.resume:
        resume(args, trainer)

    # ------------------------ start training ------------------------------------------------------------ #

    if args.rank == 0:
        logger.info('Start training from iteration {},'
                    ' and the total training iterations is {}'.format(trainer.ITERS_PER_EPOCH * args.start_epoch + 1,
                                                                      trainer.total_iters))

    trainer.prefetcher = DataPrefetcher(trainer.data_loader, args.single_aug)

    for epoch in range(args.start_epoch, args.total_epochs):
        # set epoch
        trainer.epoch = epoch

        if trainer.prefetcher.next_input is None:
            if args.world_size > 1:
                trainer.data_loader.sampler.set_epoch(epoch)
            trainer.prefetcher = DataPrefetcher(trainer.data_loader, args.single_aug)

        trainer.train(args, logger, writer)

        # save models
        if args.rank == 0:
            state_dict = get_state_dict(trainer)
            save_checkpoint(state_dict, False, file_name, 'last_epoch')

    if args.rank == 0:
        logger.info("Pre-training of experiment: {} is done.".format(args.experiment_name))
        writer.close()


def main():
    args = default_argument_parser().parse_args()

    # setup randomization
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    # multi-processing
    args.multiprocessing_distributed = args.num_machines > 1

    print('Total number of using machines: {}'.format(args.num_machines))
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


if __name__ == "__main__":
    main()
