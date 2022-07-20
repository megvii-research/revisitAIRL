import torch
from torch import distributed as dist


def reduce_tensor(tensor):
    rt = tensor.clone()
    try:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    except AssertionError:
        pass
    rt /= dist.get_world_size()
    return rt


def synchronize():
    """Helper function to synchronize (barrier) among all processes when using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    current_world_size = dist.get_world_size()
    if current_world_size == 1:
        return
    dist.barrier()


@torch.no_grad()
def dist_collect(x):
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0)


def dist_collect_grad(x):
    gpu_id = dist.get_rank()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype)
                for _ in range(dist.get_world_size())]
    dist.all_gather(out_list, x)
    out_list[gpu_id] = x
    return torch.cat(out_list, dim=0)
