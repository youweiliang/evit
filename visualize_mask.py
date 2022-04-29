import os
import torch
from torchvision.utils import save_image
from einops import rearrange


def mask(x, idx, patch_size):
    """
    Args:
        x: input image, shape: [B, 3, H, W]
        idx: indices of masks, shape: [B, T], value in range [0, h*w)
    Return:
        out_img: masked image with only patches from idx postions
    """
    h = x.size(2) // patch_size
    x = rearrange(x, 'b c (h p) (w q) -> b (c p q) (h w)', p=patch_size, q=patch_size)
    output = torch.zeros_like(x)
    idx1 = idx.unsqueeze(1).expand(-1, x.size(1), -1)
    extracted = torch.gather(x, dim=2, index=idx1)  # [b, c p q, T]
    scattered = torch.scatter(output, dim=2, index=idx1, src=extracted)
    out_img = rearrange(scattered, 'b (c p q) (h w) -> b c (h p) (w q)', p=patch_size, q=patch_size, h=h)
    return out_img


def get_deeper_idx(idx1, idx2):
    """
    Args:
        idx1: indices, shape: [B, T1]
        idx2: indices to gather from idx1, shape: [B, T2], T2 <= T1
    """
    return torch.gather(idx1, dim=1, index=idx2)


def get_real_idx(idxs, fuse_token):
    # nh = img_size // patch_size
    # npatch = nh ** 2

    # gather real idx
    for i in range(1, len(idxs)):
        tmp = idxs[i - 1]
        if fuse_token:
            B = tmp.size(0)
            tmp = torch.cat([tmp, torch.zeros(B, 1, dtype=tmp.dtype, device=tmp.device)], dim=1)
        idxs[i] = torch.gather(tmp, dim=1, index=idxs[i])
    return idxs


def save_img_batch(x, path, file_name='img{}', start_idx=0):
    for i, img in enumerate(x):
        save_image(img, os.path.join(path, file_name.format(start_idx + i)))
