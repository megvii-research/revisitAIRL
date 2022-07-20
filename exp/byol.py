import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.base_trainer import BaseTrainer
from models.siamese_network import SiameseNetwork
from utils.dist import reduce_tensor, synchronize
from utils import accuracy, AvgMeter, to_python_float, cal_remain_time
from solver.optimizer import LARS


class Normalized_L2Loss(nn.Module):
    def __init__(self):
        super(Normalized_L2Loss, self).__init__()

    def forward(self, q, k):
        q = F.normalize(q, dim=1, p=2)
        k = F.normalize(k.detach(), dim=1, p=2)

        return (2 - 2 * (q * k).sum(dim=-1, keepdim=True)).mean()


class BYOL_Network(SiameseNetwork):
    def __init__(self, low_dim, hidden_dim, width, MLP, predictor, bn, CLS):
        super(BYOL_Network, self).__init__(low_dim, hidden_dim, width, MLP, predictor, bn, CLS)

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.momentum_update(0)


class Trainer(BaseTrainer):
    def __init__(self):
        super(Trainer, self).__init__()
        # others
        self.num_workers = 6
        self.print_interval = 20
        self.CLS = True

        # optimization
        self.lr = 0.03
        self.scheduler = 'warmcos'
        self.warmup_lr = 1e-6
        self.warmup_epochs = 10
        self.weight_decay = 1e-4
        self.weight_decay_exclude = 0.
        self.momentum = 0.9
        self.lars = False

        self.lr_cls = 30
        self.scheduler_cls = 'multistep'
        self.milestones = [60, 80]

        self.iter_count = 0
        self.total_iters = None

        # model config
        self.low_dim = 256
        self.hidden_dim = 4096
        self.width = 1
        self.MLP = 'byol'
        self.bn = {'online': 'torchsync', 'target': 'torchsync'}
        self.init_param_momentum = 0.99
        self.predictor = {'online': True, 'target': False}

        # criterion
        self.criterion = Normalized_L2Loss()
        self.criterion_cls = nn.CrossEntropyLoss()

    def momentum_decay(self):
        self.param_momentum =  1.0 - (1.0 - self.init_param_momentum) * (
                (math.cos(math.pi * self.iter_count / self.total_iters) + 1) * 0.5
        )

    def build_dataloader(self, args):
        if 'train_set' not in self.__dict__:
            from data.transforms import byol_transform
            from data.datasets import build_dataset

            self.train_set = build_dataset(byol_transform(), args.data_path, True)
        return super(Trainer, self).build_dataloader(args)

    def build_model(self):
        if 'model' not in self.__dict__:
            self.model = BYOL_Network(self.low_dim, self.hidden_dim, self.width, self.MLP,
                                      predictor=self.predictor, bn=self.bn, CLS=self.CLS)
        if 'classifier' not in self.__dict__ and self.CLS:
            self.classifier = nn.Linear(2048, 1000)

    def build_optimizer(self, args):
        self.lr = self.lr * (args.batch_size / 256)

        if 'warm' in self.scheduler:
            init_lr = self.warmup_lr
        else:
            init_lr = self.lr

        if self.lars:
            params_lars = []
            params_exclude = []
            for m in self.model.online_encoder.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                    params_exclude.append(m.weight)
                    params_exclude.append(m.bias)
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        params_exclude.append(m.bias)
                    params_lars.append(m.weight)
                elif isinstance(m, nn.Conv2d):
                    params_lars.extend(list(m.parameters()))

            assert len(params_lars) + len(params_exclude) == len(list(self.model.online_encoder.parameters()))

            assert 'data_loader' in self.__dict__
            self.optimizer = LARS(
                [
                    {'params': params_lars},
                    {'params': params_exclude, 'weight_decay': self.weight_decay_exclude}
                ], lr=init_lr, weight_decay=self.weight_decay,
                momentum=self.momentum, max_epoch=args.total_epochs * self.ITERS_PER_EPOCH)
        else:
            self.optimizer = torch.optim.SGD(
                self.model.online_encoder.parameters(), lr=init_lr, weight_decay=self.weight_decay, momentum=self.momentum
            )

        if self.CLS:
            self.optimizer_cls = torch.optim.SGD(self.classifier.parameters(), lr=self.lr_cls, momentum=0.9, weight_decay=0.)

    def train(self, args, logger, writer):
        con_losses = AvgMeter()
        cls_losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()
        iter_time_meter = AvgMeter()

        if args.world_size > 1:
            self.model.module.online_encoder.train()
            self.model.module.partial_eval()
            if self.CLS:
                self.classifier.module.train()
        else:
            self.model.online_encoder.train()
            self.model.partial_eval()
            if self.CLS:
                self.classifier.train()

        epoch = self.epoch

        for i in range(self.ITERS_PER_EPOCH):
            # update iter count
            self.iter_count = epoch * self.ITERS_PER_EPOCH + i + 1

            iter_time = time.time()
            data_time = time.time()
            inps, targets = self.prefetcher.next()
            data_time = time.time() - data_time

            bs_gpu = targets.size(0)

            # forward
            output_dict = self.model(inps)
            con_loss = self.criterion(output_dict['online1'], output_dict['target2']) +\
                       self.criterion(output_dict['online2'], output_dict['target1'])

            # backward for con-loss
            self.optimizer.zero_grad()
            con_loss.backward()
            self.optimizer.step()

            # momentum update
            self.momentum_decay()
            if args.world_size > 1:
                self.model.module.momentum_update(self.param_momentum)
            else:
                self.model.momentum_update(self.param_momentum)

            # compute the statistics
            reduced_con_loss = reduce_tensor(con_loss.data)
            con_losses.update(to_python_float(reduced_con_loss), bs_gpu)

            # update lr
            lr = self.adjust_learning_rate_iter(args)

            if self.CLS:
                cls_logits = self.classifier(output_dict['gp_feat'].detach())
                cls_loss = self.criterion_cls(cls_logits, targets)
                train_top1, train_top5 = accuracy(cls_logits, targets, (1, 5))

                # backward for cls-loss
                self.optimizer_cls.zero_grad()
                cls_loss.backward()
                self.optimizer_cls.step()

                # compute the statistics
                reduced_cls_loss = reduce_tensor(cls_loss.data)
                reduced_top1 = reduce_tensor(train_top1)
                reduced_top5 = reduce_tensor(train_top5)

                cls_losses.update(to_python_float(reduced_cls_loss), bs_gpu)
                top1.update(to_python_float(reduced_top1), bs_gpu)
                top5.update(to_python_float(reduced_top5), bs_gpu)

                lr_cls = self.adjust_learning_rate_cls(args)
            else:
                reduced_cls_loss = torch.tensor([0.])
                reduced_top1 = torch.tensor([0.])
                reduced_top5 = torch.tensor([0.])
                lr_cls = 0

            iter_time = time.time() - iter_time
            iter_time_meter.update(iter_time)

            # tensorboard
            if i % self.print_interval == 0 and args.rank == 0:
                writer.add_scalar('Train/BYOL_Loss_Iter', to_python_float(reduced_con_loss),
                                  global_step=self.iter_count)
                writer.add_scalar('Train/CLS_Loss_Iter', to_python_float(reduced_cls_loss), global_step=self.iter_count)
                writer.add_scalar('Train/Top1_Iter', to_python_float(reduced_top1), global_step=self.iter_count)
                writer.add_scalar('Train/Top5_Iter', to_python_float(reduced_top5), global_step=self.iter_count)

            # logger print
            remain_time = cal_remain_time(args, self.iter_count, iter_time_meter, self.ITERS_PER_EPOCH)
            if (i + 1) % self.print_interval == 0 and args.rank == 0:

                iter_speed = 1 / iter_time_meter.avg
                logger.info(
                    'Epoch: [{}/{}], Iter: [{}/{}], Remain-Time: {}, {:.2f}it/s, Data-Time: {:.3f},'
                    ' LR: {:.4f}, LR-CLS: {:.4f}, Con-Loss: {:.2f}, CLS-Loss: {:.2f}, Top-1: {:.2f},'
                    ' Top-5: {:.2f}'.format(
                        epoch + 1, args.total_epochs, i + 1, self.ITERS_PER_EPOCH, remain_time, iter_speed,
                        data_time, lr, lr_cls, con_losses.avg, cls_losses.avg, top1.avg, top5.avg
                    ))

        if args.rank == 0:
            logger.info(
                'Train-Epoch: [{}/{}], LR: {:.4f}, LR-CLS: {:.4f}, Con-Loss: {:.2f}, CLS-Loss: {:.2f},'
                ' Top-1: {:.2f}, Top-5: {:.2f}'.format(epoch + 1, args.total_epochs, lr, lr_cls,
                                                       con_losses.avg, cls_losses.avg, top1.avg,
                                                       top5.avg))
            logger.info('')

            writer.add_scalar('Train/BYOL_Loss', con_losses.avg, global_step=epoch + 1)
            writer.add_scalar('Train/CLS_Loss', cls_losses.avg, global_step=epoch + 1)
            writer.add_scalar('Train/Top1', top1.avg, global_step=epoch + 1)
            writer.add_scalar('Train/Top5', top5.avg, global_step=epoch + 1)
