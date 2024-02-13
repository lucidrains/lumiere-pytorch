"""
einstein notation
b - batch
t - time
c - channels
h - height
w - width
"""

from functools import wraps, partial

import torch
from torch import nn, einsum, Tensor, is_tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import List, Tuple, Optional, Type

from einops import rearrange, pack, unpack, repeat

from lumiere_pytorch.lumiere import (
    residualize,
    image_or_video_to_time,
    handle_maybe_channel_last,
    Sequential,
    Residual,
    Lumiere
)

from x_transformers.x_transformers import (
    Attention,
    RMSNorm
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def compact_values(d: dict):
    return {k: v for k, v in d.items() if exists(v)}

# temporal down and upsample

def init_bilinear_kernel_1d_(conv: Module):
    nn.init.zeros_(conv.weight)
    if exists(conv.bias):
        nn.init.zeros_(conv.bias)

    channels = conv.weight.shape[0]
    bilinear_kernel = Tensor([0.5, 1., 0.5])
    diag_mask = torch.eye(channels).bool()
    conv.weight.data[diag_mask] = bilinear_kernel

class MPTemporalDownsample(Module):
    def __init__(
        self,
        dim,
        channel_last = False,
        time_dim = None
    ):
        super().__init__()
        self.time_dim = time_dim
        self.channel_last = channel_last

        self.conv = nn.Conv1d(dim, dim, kernel_size = 3, stride = 2, padding = 1)
        init_bilinear_kernel_1d_(self.conv)

    @handle_maybe_channel_last
    @image_or_video_to_time
    def forward(
        self,
        x
    ):
        assert x.shape[-1] > 1, 'time dimension must be greater than 1 to be compressed'

        return self.conv(x)

class MPTemporalUpsample(Module):
    def __init__(
        self,
        dim,
        channel_last = False,
        time_dim = None
    ):
        super().__init__()
        self.time_dim = time_dim
        self.channel_last = channel_last

        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        init_bilinear_kernel_1d_(self.conv)

    @handle_maybe_channel_last
    @image_or_video_to_time
    def forward(
        self,
        x
    ):
        return self.conv(x)

# main modules

class MPConvolutionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        conv2d_kernel_size = 3,
        conv1d_kernel_size = 3,
        groups = 8,
        channel_last = False,
        time_dim = None
    ):
        super().__init__()
        assert is_odd(conv2d_kernel_size)
        assert is_odd(conv1d_kernel_size)

        self.time_dim = time_dim
        self.channel_last = channel_last

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

        self.proj_out = nn.Conv1d(dim, dim, 1)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    @residualize
    @handle_maybe_channel_last
    def forward(
        self,
        x,
        batch_size = None
    ):
        is_video = x.ndim == 5

        if is_video:
            batch_size = x.shape[0]
            x = rearrange(x, 'b c t h w -> (b t) c h w')

        x = self.spatial_conv(x)

        rearrange_kwargs = compact_values(dict(b = batch_size, t = self.time_dim))

        assert len(rearrange_kwargs) > 0, 'either batch_size is passed in on forward, or time_dim is set on init'
        x = rearrange(x, '(b t) c h w -> b h w c t', **rearrange_kwargs)

        x, ps = pack_one(x, '* c t')

        x = self.temporal_conv(x)
        x = self.proj_out(x)

        x = unpack_one(x, ps, '* c t')

        if is_video:
            x = rearrange(x, 'b h w c t -> b c t h w')
        else:
            x = rearrange(x, 'b h w c t -> (b t) c h w')

        return x

class MPAttentionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        depth = 1,
        prenorm = True,
        residual_attn = True,
        time_dim = None,
        channel_last = False,
        **attn_kwargs
    ):
        super().__init__()

        self.time_dim = time_dim
        self.channel_last = channel_last

        self.temporal_attns = ModuleList([])

        for _ in range(depth):
            attn = Sequential(
                RMSNorm(dim) if prenorm else None,
                Attention(
                    dim = dim,
                    **attn_kwargs
                )
            )

            if residual_attn:
                attn = Residual(attn)

            self.temporal_attns.append(attn)

        self.proj_out = nn.Linear(dim, dim)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    @residualize
    @handle_maybe_channel_last
    def forward(
        self,
        x,
        batch_size = None
    ):
        is_video = x.ndim == 5
        assert is_video ^ (exists(batch_size) or exists(self.time_dim)), 'either a tensor of shape (batch, channels, time, height, width) is passed in, or (batch * time, channels, height, width) along with `batch_size`'

        if self.channel_last:
            x = rearrange(x, 'b ... c -> b c ...')

        if is_video:
            batch_size = x.shape[0]
            x = rearrange(x, 'b c t h w -> b h w t c')
        else:
            assert exists(batch_size) or exists(self.time_dim)

            rearrange_kwargs = dict(b = batch_size, t = self.time_dim)
            x = rearrange(x, '(b t) c h w -> b h w t c', **compact_values(rearrange_kwargs))

        x, ps = pack_one(x, '* t c')

        for attn in self.temporal_attns:
            x = attn(x)

        x = self.proj_out(x)

        x = unpack_one(x, ps, '* t c')

        if is_video:
            x = rearrange(x, 'b h w t c -> b c t h w')
        else:
            x = rearrange(x, 'b h w t c -> (b t) c h w')

        if self.channel_last:
            x = rearrange(x, 'b c ... -> b ... c')

        return x

# mp lumiere is just lumiere with the four mp temporal modules

MPLumiere = partial(
    Lumiere,
    conv_klass = MPConvolutionInflationBlock,
    attn_klass = MPAttentionInflationBlock,
    downsample_klass = MPTemporalDownsample,
    upsample_klass = MPTemporalUpsample
)
