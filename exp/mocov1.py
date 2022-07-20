import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from exp.base_trainer import BaseTrainer
from models.siamese_network import SiameseNetwork
from utils.dist import reduce_tensor, dist_collect
from utils import accuracy, AvgMeter, to_python_float, cal_remain_time


class MoCo_Network(SiameseNetwork):
    def __init__(self, low_dim, hidden_dim, width, MLP, predictor, bn, param_momentum,
                 queue_size, temperature, CLS):
        super(MoCo_Network, self).__init__(low_dim, hidden_dim, width, MLP, predictor, bn, CLS)
        self.init_param_momentum = param_momentum
        self.param_momentum = param_momentum
        self.queue_size = queue_size
        self.temperature = temperature

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # sync the weights of the siamese network
        self.momentum_update(0)

        self.register_buffer('queue', torch.randn(queue_size, low_dim))
        self.queue = F.normalize(self.queue, dim=1, p=2)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def batch_shuffle_ddp(self, x):
        """
                Batch shuffle, for making use of BatchNorm.
                *** Only support DistributedDataParallel (DDP) model. ***
                """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = dist_collect(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def batch_unshffule_ddp(self, x, idx_unshuffle):
        """
                Undo batch shuffle.
                *** Only support DistributedDataParallel (DDP) model. ***
                """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = dist_collect(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def norm(self, x):
        return F.normalize(x, dim=1, p=2)

    def forward(self, inps):
        # forward with grad
        online_q, gp_feat = self.online_encoder(inps[0])
        online_q = self.norm(online_q)

        # forward without grad
        with torch.no_grad():
            shuffled_x2, shuffled_idx = self.batch_shuffle_ddp(inps[1])
            shuffled_target_k = self.target_encoder(shuffled_x2)
            target_k = self.batch_unshffule_ddp(shuffled_target_k, shuffled_idx)
            target_k = self.norm(target_k)

        pos = (online_q * target_k.detach()).sum(dim=-1, keepdim=True)
        neg = torch.mm(online_q, self.queue.clone().detach().t())
        logits = torch.cat((pos, neg), dim=1).div(self.temperature)
        labels = torch.zeros(logits.size(0)).long().cuda()
        loss = self.criterion(logits, labels)

        target_k_all = dist_collect(target_k)
        self.update_queue(target_k_all)
        return {'loss': loss, 'logits': logits, 'labels': labels, 'gp_feat': gp_feat}

    @torch.no_grad()
    def update_queue(self, target_k_all):
        total_batch_size = target_k_all.size(0)
        out_ids = torch.fmod(torch.arange(total_batch_size, dtype=torch.long).cuda() + self.queue_ptr,
                             self.queue_size)
        self.queue.index_copy_(0, out_ids, target_k_all)
        self.queue_ptr[0] = (self.queue_ptr + total_batch_size) % self.queue_size


class Trainer(BaseTrainer):
    def __init__(self):
        super(Trainer, self).__init__()
        # others
        self.num_workers = 6
        self.print_interval = 20
        self.CLS = True

        # data
        self.aug_plus = False

        # optimization
        self.lr = 0.03
        self.scheduler = 'multistep'
        self.warmup_lr = 1e-6
        self.warmup_epochs = 10
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.temperature = 0.07

        self.lr_cls = 30
        self.scheduler_cls = 'multistep'
        self.milestones = [60, 80]

        self.iter_count = 0
        self.total_iters = None

        # model config
        self.low_dim = 128
        self.hidden_dim = 2048
        self.width = 1
        self.MLP = 'cls'
        self.predictor = {'online': False, 'target': False}
        self.bn = {'online': 'vanilla', 'target': 'vanilla'}
        self.param_momentum = 0.999
        self.queue_size = 65536

        self.criterion_cls = nn.CrossEntropyLoss()

    def build_dataloader(self, args):
        if 'train_set' not in self.__dict__:
            from data.transforms import moco_transform
            from data.datasets import build_dataset

            self.train_set = build_dataset(moco_transform(aug_plus=self.aug_plus), args.data_path, True)
        return super(Trainer, self).build_dataloader(args)

    def build_model(self):
        if 'model' not in self.__dict__:
            self.model = MoCo_Network(low_dim=self.low_dim, hidden_dim=self.hidden_dim, width=self.width, MLP=self.MLP,
                                      predictor=self.predictor, bn=self.bn, param_momentum=self.param_momentum,
                                      queue_size=self.queue_size, temperature=self.temperature, CLS=self.CLS)
        if 'classifier' not in self.__dict__ and self.CLS:
            self.classifier = nn.Linear(2048, 1000)

    def build_optimizer(self, args):
        self.lr = self.lr * (args.batch_size / 256)

        if 'warm' in self.scheduler:
            init_lr = self.warmup_lr
        else:
            init_lr = self.lr

        self.optimizer = torch.optim.SGD(
            self.model.online_encoder.parameters(), lr=init_lr, weight_decay=self.weight_decay,
            momentum=self.momentum
        )

        if self.CLS:
            self.optimizer_cls = torch.optim.SGD(self.classifier.parameters(), lr=self.lr_cls, momentum=0.9, weight_decay=0.)

    def train(self, args, logger, writer):
        con_losses = AvgMeter()
        cls_losses = AvgMeter()
        top1 = AvgMeter()
        top5 = AvgMeter()
        inst_top1 = AvgMeter()
        inst_top5 = AvgMeter()
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
            con_loss = output_dict['loss']
            train_inst_top1, train_inst_top5 = accuracy(output_dict['logits'], output_dict['labels'], (1, 5))

            # backward for con-loss
            self.optimizer.zero_grad()
            con_loss.backward()
            self.optimizer.step()

            # update target encoder
            if args.world_size > 1:
                self.model.module.momentum_update(self.param_momentum)
            else:
                self.model.momentum_update(self.param_momentum)

            # update lr
            lr = self.adjust_learning_rate_iter(args)

            # compute the statistics
            reduced_con_loss = reduce_tensor(con_loss.data)
            con_losses.update(to_python_float(reduced_con_loss), bs_gpu)
            reduced_inst_top1 = reduce_tensor(train_inst_top1)
            reduced_inst_top5 = reduce_tensor(train_inst_top5)
            inst_top1.update(to_python_float(reduced_inst_top1), bs_gpu)
            inst_top5.update(to_python_float(reduced_inst_top5), bs_gpu)

            if self.CLS:
                cls_logits = self.classifier(output_dict['gp_feat'].detach())
                cls_loss = self.criterion_cls(cls_logits, targets)
                train_top1, train_top5 = accuracy(cls_logits, targets, (1, 5))

                # backward for cls-loss
                self.optimizer_cls.zero_grad()
                cls_loss.backward()
                self.optimizer_cls.step()

                lr_cls = self.adjust_learning_rate_cls(args)

                reduced_cls_loss = reduce_tensor(cls_loss.data)
                reduced_top1 = reduce_tensor(train_top1)
                reduced_top5 = reduce_tensor(train_top5)

                cls_losses.update(to_python_float(reduced_cls_loss), bs_gpu)
                top1.update(to_python_float(reduced_top1), bs_gpu)
                top5.update(to_python_float(reduced_top5), bs_gpu)

            else:
                reduced_cls_loss = torch.tensor([0.])
                reduced_top1 = torch.tensor([0.])
                reduced_top5 = torch.tensor([0.])
                lr_cls = 0

            iter_time = time.time() - iter_time
            iter_time_meter.update(iter_time)

            # tensorboard
            if i % self.print_interval == 0 and args.rank == 0:
                writer.add_scalar('Train/Contrastive_Loss_Iter', to_python_float(reduced_con_loss), global_step=self.iter_count)
                writer.add_scalar('Train/CLS_Loss_Iter', to_python_float(reduced_cls_loss), global_step=self.iter_count)
                writer.add_scalar('Train/Top1_Iter', to_python_float(reduced_top1), global_step=self.iter_count)
                writer.add_scalar('Train/Top5_Iter', to_python_float(reduced_top5), global_step=self.iter_count)
                writer.add_scalar('Train/Inst_Top1_Iter', to_python_float(reduced_inst_top1), global_step=self.iter_count)
                writer.add_scalar('Train/Inst_Top5_Iter', to_python_float(reduced_inst_top5), global_step=self.iter_count)

            # logger print
            remain_time = cal_remain_time(args, self.iter_count, iter_time_meter, self.ITERS_PER_EPOCH)
            if (i + 1) % self.print_interval == 0 and args.rank == 0:
                iter_speed = 1 / iter_time_meter.avg
                logger.info(
                    'Epoch: [{}/{}], Iter: [{}/{}], Remain-Time: {}, {:.2f}it/s, Data-Time: {:.3f},'
                    ' LR: {:.4f}, LR-CLS: {:.4f}, Con-Loss: {:.2f}, CLS-Loss: {:.2f}, Top-1: {:.2f},'
                    ' Top-5: {:.2f}, Inst-Top-1: {:.2f}, Inst-Top-5: {:.2f}'.format(
                        epoch + 1, args.total_epochs, i + 1, self.ITERS_PER_EPOCH, remain_time, iter_speed,
                        data_time, lr, lr_cls, con_losses.avg, cls_losses.avg, top1.avg, top5.avg,
                        inst_top1.avg, inst_top5.avg
                    ))

        if args.rank == 0:
            logger.info(
                'Train-Epoch: [{}/{}], LR: {:.4f}, LR-CLS: {:.4f}, Con-Loss: {:.2f},'
                ' CLS-Loss: {:.2f}, Top-1: {:.2f}, Top-5: {:.2f}, Inst-Top-1: {:.2f},'
                ' Inst-Top-5: {:.2f}'.format(epoch + 1, args.total_epochs, lr, lr_cls,
                                             con_losses.avg, cls_losses.avg, top1.avg,
                                             top5.avg, inst_top1.avg, inst_top5.avg))
            logger.info('')

            writer.add_scalar('Train/Contrastive_Loss', con_losses.avg, global_step=epoch + 1)
            writer.add_scalar('Train/CLS_Loss', cls_losses.avg, global_step=epoch + 1)
            writer.add_scalar('Train/Top1', top1.avg, global_step=epoch + 1)
            writer.add_scalar('Train/Top5', top5.avg, global_step=epoch + 1)
            writer.add_scalar('Train/Inst_Top1', inst_top1.avg, global_step=epoch + 1)
            writer.add_scalar('Train/Inst_Top5', inst_top5.avg, global_step=epoch + 1)
