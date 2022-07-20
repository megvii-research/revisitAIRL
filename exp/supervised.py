import time

import torch
import torch.nn as nn

from exp.base_trainer import BaseTrainer
from utils.dist import reduce_tensor
from utils import accuracy, AvgMeter, to_python_float, cal_remain_time
from models.resnet import resnet50


class Trainer(BaseTrainer):
    def __init__(self):
        super(Trainer, self).__init__()
        # others
        self.num_workers = 4
        self.print_interval = 20 # iters
        self.eval_interval = 100 # epochs
        self.to_eval = True

        # optimization
        self.lr = 0.1
        self.scheduler = 'warmcos'
        self.milestones = [30, 60, 90]
        self.warmup_lr = 1e-6
        self.warmup_epochs = 10
        self.weight_decay = 1e-4
        self.weight_decay_exclude = 0.
        self.momentum = 0.9

        self.iter_count = 0
        self.total_iters = None

        # model config
        self.low_dim = 1000
        self.hidden_dim = 2048
        self.width = 1
        self.MLP = 'cls'
        self.bn = 'vanilla'
        self.CLS = False

        # criterion
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self):
        self.model = resnet50(low_dim=self.low_dim, MLP=self.MLP, bn=self.bn, width=self.width)

    def build_dataloader(self, args):
        if 'train_set' not in self.__dict__:
            from data.transforms import typical_imagenet_transform
            from data.datasets import build_dataset

            train_transform = typical_imagenet_transform(True)
            eval_transform = typical_imagenet_transform(False)
            self.train_set = build_dataset(train_transform, args.data_path, True, False)
            self.eval_set = build_dataset(eval_transform, args.data_path, False, False)

            self.eval_loader = torch.utils.data.DataLoader(self.eval_set, batch_size=200, shuffle=False, num_workers=2,
                                                           pin_memory=True)

        return super(Trainer, self).build_dataloader(args)

    def build_optimizer(self, args):
        self.lr = self.lr * (args.batch_size / 256)

        if 'warm' in self.scheduler:
            init_lr = self.warmup_lr
        else:
            init_lr = self.lr

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=init_lr, weight_decay=self.weight_decay,
            momentum=self.momentum
        )

    def train(self, args, logger, writer):
        losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()
        iter_time_meter = AvgMeter()

        if args.world_size > 1:
            self.model.module.train()
        else:
            self.model.train()

        epoch = self.epoch
        if args.world_size > 1:
            self.model.module.total_iters = args.total_epochs * self.ITERS_PER_EPOCH
        else:
            self.model.total_iters = args.total_epochs * self.ITERS_PER_EPOCH

        for i in range(self.ITERS_PER_EPOCH):

            iter_time = time.time()
            data_time = time.time()
            inps, targets = self.prefetcher.next()
            data_time = time.time() - data_time

            bs_gpu = targets.size(0)

            # forward
            logits = self.model(inps)
            loss = self.criterion(logits, targets)
            train_top1, train_top5 = accuracy(logits, targets, (1, 5))

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update iter count
            self.iter_count = epoch * self.ITERS_PER_EPOCH + i + 1

            # update lr
            lr = self.adjust_learning_rate_iter(args)

            # compute the statistics
            iter_time = time.time() - iter_time
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

            iter_time_meter.update(iter_time)
            reduced_loss = reduce_tensor(loss.data)
            reduced_top1 = reduce_tensor(train_top1)
            reduced_top5 = reduce_tensor(train_top5)
            losses.update(to_python_float(reduced_loss), bs_gpu)
            top1.update(to_python_float(reduced_top1), bs_gpu)
            top5.update(to_python_float(reduced_top5), bs_gpu)

            # tensorboard
            if i % self.print_interval == 0 and args.rank == 0:
                writer.add_scalar('Train/Loss_Iter', to_python_float(reduced_loss),
                                  global_step=self.iter_count)
                writer.add_scalar('Train/Top1_Iter', to_python_float(reduced_top1), global_step=self.iter_count)
                writer.add_scalar('Train/Top5_Iter', to_python_float(reduced_top5), global_step=self.iter_count)

            # logger print
            remain_time = cal_remain_time(args, self.iter_count, iter_time_meter, self.ITERS_PER_EPOCH)
            if (i + 1) % self.print_interval == 0 and args.rank == 0:
                iter_speed = 1 / iter_time_meter.avg
                logger.info(
                        'Epoch: [{}/{}], Iter: [{}/{}], Mem: {:.2f}, Remain-Time: {}, {:.2f}it/s, Data-Time: {:.3f},'
                    ' LR: {:.4f}, Loss: {:.2f}, Top-1: {:.2f}, Top-5: {:.2f}'.format(
                        epoch + 1, args.total_epochs, i + 1, self.ITERS_PER_EPOCH, max_mem_mb, remain_time, iter_speed,
                        data_time, lr, losses.avg, top1.avg, top5.avg))

        if args.rank == 0:
            logger.info(
                'Train-Epoch: [{}/{}], LR: {:.4f}, Loss: {:.2f}, Top-1: {:.2f}, Top-5: {:.2f}'.format(
                    epoch + 1, args.total_epochs, lr, losses.avg, top1.avg, top5.avg))
            logger.info('')

            writer.add_scalar('Train/Loss', losses.avg, global_step=epoch + 1)
            writer.add_scalar('Train/Top1', top1.avg, global_step=epoch + 1)
            writer.add_scalar('Train/Top5', top5.avg, global_step=epoch + 1)

        if self.to_eval and (epoch + 1) % self.eval_interval == 0 and args.rank == 0:
            self.eval(args, logger, writer)

    def eval(self, args, logger, writer):
        if args.world_size > 1:
            self.model.module.eval()
        else:
            self.model.eval()
        eval_top1 = AvgMeter()
        eval_top5 = AvgMeter()
        eval_loss = AvgMeter()

        with torch.no_grad():
            logger.info('')
            logger.info('Eval at epoch-{}'.format(self.epoch + 1))

            for i, (inps, targets) in enumerate(self.eval_loader):
                inps = inps.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)

                logits = self.model(inps)

                loss = self.criterion(logits, targets)
                batch_top1, batch_top5 = accuracy(logits, targets, (1, 5))

                eval_loss.update(loss.data, inps.size(0))
                eval_top1.update(batch_top1.item(), inps.size(0))
                eval_top5.update(batch_top5.item(), inps.size(0))

                if (i + 1) % 50 == 0:
                    logger.info('\tIter: [{}/{}], Loss: {:.2f}, Top-1: {:.2f}, Top-5: {:.2f}'.format(
                        i+1, len(self.eval_loader), eval_loss.avg, eval_top1.avg, eval_top5.avg)
                    )
        logger.info('\tEvaluation done! Loss: {:.2f}, Top-1: {:.2f}, Top-5: {:.2f}'.format(eval_loss.avg,
                                                                                           eval_top1.avg, eval_top5.avg))
        logger.info('')
        writer.add_scalar('Eval/Loss', eval_loss.avg, global_step=self.epoch + 1)
        writer.add_scalar('Eval/Top1', eval_top1.avg, global_step=self.epoch + 1)
        writer.add_scalar('Eval/Top5', eval_top5.avg, global_step=self.epoch + 1)
