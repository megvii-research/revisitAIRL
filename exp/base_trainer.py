import math
from abc import ABCMeta, abstractmethod

import torch

from data import InfiniteSampler


class BaseTrainer(metaclass=ABCMeta):
    """
    Basic class for any experiment.
    """

    def __init__(self):
        self.seed = None
        self.output_dir = "outputs"
        self.print_interval = 100

        self.num_workers = 6

        self.iter_count = 0

    @abstractmethod
    def build_model(self):
        pass

    def build_dataloader(self, args):
        if 'train_set' in self.__dict__:
            if args.world_size > 1:
                batch_size = args.batch_size // args.world_size
                sampler = InfiniteSampler(len(self.train_set), shuffle=True, seed=self.seed if self.seed is not None else 0,
                                          rank=args.rank, world_size=args.world_size)
            else:
                batch_size = args.batch_size
                sampler = None

            dataloader_kwargs = {"num_workers": self.num_workers, "pin_memory": False}
            dataloader_kwargs["sampler"] = sampler
            dataloader_kwargs["batch_size"] = batch_size
            dataloader_kwargs["shuffle"] = False
            dataloader_kwargs["drop_last"] = True
            train_loader = torch.utils.data.DataLoader(self.train_set, **dataloader_kwargs)
            self.data_loader = train_loader
            self.ITERS_PER_EPOCH = len(self.train_set) // args.batch_size
        else:
            raise ValueError('Error: train_set is not in self.__dict__.')
        self.total_iters = args.total_epochs * self.ITERS_PER_EPOCH
        self.prefetcher = None
        return self.data_loader

    @abstractmethod
    def build_optimizer(self, args):
        pass

    def adjust_learning_rate_iter(self, args):
        """Decay the learning rate based on schedule"""
        total_iters = self.ITERS_PER_EPOCH * args.total_epochs

        lr = self.lr
        if self.scheduler == 'cos':  # cosine lr schedule
            lr *= 0.5 * (1.0 + math.cos(math.pi * self.iter_count / total_iters))
        elif self.scheduler == 'warmcos':
            warmup_total_iters = self.ITERS_PER_EPOCH * self.warmup_epochs
            if self.iter_count <= warmup_total_iters:
                warmup_lr = 1e-6
                lr = (lr - warmup_lr) * self.iter_count / float(warmup_total_iters) + warmup_lr
            else:
                lr *= 0.5 * (
                            1.0 + math.cos(math.pi * (self.iter_count - warmup_total_iters) / (total_iters - warmup_total_iters)))
        elif self.scheduler == 'multistep':  # stepwise lr schedule
            milestones = [int(total_iters * milestone / 100) for milestone in self.milestones]
            for milestone in milestones:
                lr *= 0.1 if self.iter_count >= milestone else 1.0
        else:
            raise ValueError('Scheduler version {} is not available.'.format(self.scheduler))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def adjust_learning_rate_cls(self, args):
        total_iters = self.ITERS_PER_EPOCH * args.total_epochs
        lr = self.lr_cls
        if self.scheduler_cls == 'multistep':
            milestones = [int(total_iters * milestone / args.total_epochs) for milestone in self.milestones]
            for milestone in milestones:
                lr *= 0.1 if self.iter_count >= milestone else 1.0
        elif self.scheduler_cls == 'cos':
            lr *= 0.5 * (1.0 + math.cos(math.pi * self.iter_count / total_iters))
        else:
            raise ValueError('Scheduler of CLS {} is not available'.format(self.scheduler_cls))
        for param_group in self.optimizer_cls.param_groups:
            param_group['lr'] = lr
        return lr

    def train(self, args, logger, writer):
        pass

    def update(self, options: dict) -> str:
        if options is None:
            return None
        assert isinstance(options, dict)
        msg = ""
        for k, v in options.items():
            if k in self.__dict__:
                old_v = self.__getattribute__(k)
                old_v = self.__getattribute__(k)

                if isinstance(v, list) and v[0].startswith("{"):
                    v = eval("','".join(v))
                if not v == old_v:
                    self.__setattr__(k, v)
                    msg = "{}\n'{}' is overridden from '{}' to '{}'".format(msg, k, old_v, v)
            else:
                self.__setattr__(k, v)
                msg = "{}\n'{}' is set to '{}'".format(msg, k, v)
        return msg
