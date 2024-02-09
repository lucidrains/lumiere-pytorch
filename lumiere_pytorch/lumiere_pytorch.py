import torch
from torch import nn, einsum
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange

from x_transformers.x_transformers import (
    Attention,
    RMSNorm
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

# main modules

class ConvolutionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        conv2d_kernel_size = 3,
        conv1d_kernel_size = 3,
        groups = 8
    ):
        super().__init__()
        assert is_odd(conv2d_kernel_size)
        assert is_odd(conv1d_kernel_size)

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, conv2d_kernel_size, padding = conv2d_kernel_size // 2),
            nn.GroupNorm(groups, num_channels = dim),
            nn.SiLU()
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(dim, dim, conv1d_kernel_size, padding = conv1d_kernel_size // 2),
            nn.GroupNorm(groups, num_channels = dim),
            nn.SiLU()
        )

        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(
        self,
        images,
        batch_size = None
    ):
        return images

class AttentionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        **attn_kwargs
    ):
        super().__init__()

        self.attns = ModuleList([])

        for _ in range(depth):
            attn = Attention(
                dim = dim,
                **attn_kwargs
            )

            self.attns.append(attn)

        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(
        self,
        images,
        batch_size = None
    ):
        return images
