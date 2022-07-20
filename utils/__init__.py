import os
import torch
import shutil


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def cal_remain_time(args, iter_count, batch_time_meter, ITERS_PER_EPOCH):
    remain_time = (ITERS_PER_EPOCH * args.total_epochs - iter_count) * batch_time_meter.avg
    t_m, t_s = divmod(remain_time, 60)
    t_h, t_m = divmod(t_m, 60)
    t_d, t_h = divmod(t_h, 24)
    remain_time = "{}d.{:02d}h.{:02d}m".format(int(t_d), int(t_h), int(t_m))
    return remain_time


def get_state_dict(trainer):
    state_dict = {
        'start_epoch': trainer.epoch + 1,
        'model': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
    }
    if trainer.CLS:
        state_dict['classifier'] = trainer.classifier.state_dict()
        state_dict['optimizer_cls'] = trainer.optimizer_cls.state_dict()
    return state_dict


def resume(args, trainer):
    file_name = os.path.join('outputs', args.experiment_name)
    checkpoint_tar = os.path.join(file_name, 'last_epoch_ckpt.pth.tar')
    if not os.path.isfile(checkpoint_tar) and args.rank == 0:
        print('No checkpoint found in {}'.format(file_name))

    if args.rank == 0:
        print('Loading checkpoint from {}'.format(file_name))

    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(checkpoint_tar, map_location=loc)

    args.start_epoch = checkpoint['start_epoch']
    trainer.model.load_state_dict(checkpoint['model'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])

    if trainer.CLS:
        trainer.classifier.load_state_dict(checkpoint['classifier'])
        trainer.optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])


def save_checkpoint(state, is_best, save, model_name=""):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, model_name + "_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, model_name + "_best_ckpt.pth.tar")
        shutil.copyfile(filename, best_filename)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_devices(n_gpus):
    if n_gpus == 1:
        return '0'
    else:
        parsed_ids = ','.join(map(lambda x: str(x), list(range(n_gpus))))
        return parsed_ids


class AvgMeter(object):

    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.cnt = 0
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
