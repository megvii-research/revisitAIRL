import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd.function import Function


__all__ = [
    'SimpleBatchNorm2d',
    'SimpleBatchNorm1d',
    'get_batchnorm_2d',
    'get_batchnorm_1d',
]


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class PlainSyncBatchNorm2d(nn.BatchNorm2d):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """

    def forward(self, input):
        if dist.get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias


class PlainSyncBatchNorm1d(nn.BatchNorm1d):
    """
    `torch.nn.SyncBatchNorm` has known unknown bugs.
    It produces significantly worse AP (and sometimes goes NaN)
    when the batch size on each worker is quite different
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    Use this implementation before `nn.SyncBatchNorm` is fixed.
    It is slower than `nn.SyncBatchNorm`.
    """

    def forward(self, input):
        if dist.get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]

        mean = torch.mean(input, dim=[0])
        meansqr = torch.mean(input * input, dim=[0])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1)
        bias = bias.reshape(1, -1)
        return input * scale + bias


class SimpleBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        if self.training:
            self.running_mean = self.running_mean.clone()
            self.running_var = self.running_var.clone()

        out = super(SimpleBatchNorm2d, self).forward(x)
        return out


class SimpleBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if self.train:
            self.running_mean = self.running_mean.clone()
            self.running_var = self.running_var.clone()
        out = super(SimpleBatchNorm1d, self).forward(x)
        return out


def get_batchnorm_2d(bn):
    if bn == 'customized':
        norm_layer = SimpleBatchNorm2d
    elif bn == 'vanilla':
        norm_layer = nn.BatchNorm2d
    elif bn == 'torchsync':
        norm_layer = nn.SyncBatchNorm
    elif bn == 'plainsync':
        norm_layer = PlainSyncBatchNorm2d
    else:
        raise ValueError('{} (bn) is not available'.format(bn))

    return norm_layer


def get_batchnorm_1d(bn):
    if bn == 'customized':
        norm1d_layer = SimpleBatchNorm1d
    elif bn == 'vanilla':
        norm1d_layer = nn.BatchNorm1d
    elif bn == 'torchsync':
        norm1d_layer = nn.SyncBatchNorm
    elif bn == 'plainsync':
        norm1d_layer = PlainSyncBatchNorm1d
    else:
        raise ValueError('{} (bn) is not available'.format(bn))

    return norm1d_layer
