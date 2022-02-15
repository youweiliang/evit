import math
import time
import torch
from torchprofile import profile_macs


def adjust_keep_rate(iters, epoch, warmup_epochs, total_epochs,
                       ITERS_PER_EPOCH, base_keep_rate=0.5, max_keep_rate=1):
    if epoch < warmup_epochs:
        return 1
    if epoch >= total_epochs:
        return base_keep_rate
    total_iters = ITERS_PER_EPOCH * (total_epochs - warmup_epochs)
    iters = iters - ITERS_PER_EPOCH * warmup_epochs
    keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) \
        * (math.cos(iters / total_iters * math.pi) + 1) * 0.5

    return keep_rate


def speed_test(model, ntest=100, batchsize=64, x=None, **kwargs):
    if x is None:
        img_size = model.img_size
        x = torch.rand(batchsize, 3, *img_size).cuda()
    else:
        batchsize = x.shape[0]
    model.eval()

    start = time.time()
    for i in range(ntest):
        model(x, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    elapse = end - start
    speed = batchsize * ntest / elapse
    # speed = torch.tensor(speed, device=x.device)
    # torch.distributed.broadcast(speed, src=0, async_op=False)
    # speed = speed.item()
    return speed


def get_macs(model, x=None):
    model.eval()
    if x is None:
        img_size = model.img_size
        x = torch.rand(1, 3, *img_size).cuda()
    macs = profile_macs(model, x)
    return macs


def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl
