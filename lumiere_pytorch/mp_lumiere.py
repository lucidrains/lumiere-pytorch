"""
the magnitude-preserving modules proposed in https://arxiv.org/abs/2312.02696 by Karras et al.
"""

from math import sqrt
from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import List, Tuple, Optional

from einops import rearrange, pack, unpack, repeat

from lumiere_pytorch.lumiere import (
    image_or_video_to_time,
    handle_maybe_channel_last,
    Lumiere
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

def compact_values(d: dict):
    return {k: v for k, v in d.items() if exists(v)}

# in paper, they use eps 1e-4 for pixelnorm

def l2norm(t, dim = -1, eps = 1e-12):
    return F.normalize(t, dim = dim, eps = eps)

def interpolate_1d(x, length, mode = 'bilinear'):
    x = rearrange(x, 'b c t -> b c t 1')
    x = F.interpolate(x, (length, 1), mode = mode)
    return rearrange(x, 'b c t 1 -> b c t')

# mp activations
# section 2.5

class MPSiLU(Module):
    def forward(self, x):
        return F.silu(x) / 0.596

# gain - layer scaling

class Gain(Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return x * self.gain

# mp linear

class Linear(Module):
    def __init__(self, dim_in, dim_out, eps = 1e-4):
        super().__init__()
        weight = torch.randn(dim_out, dim_in)
        self.weight = nn.Parameter(weight)
        self.eps = eps
        self.fan_in = dim_in

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                normed_weight = l2norm(self.weight, eps = self.eps)
                self.weight.copy_(normed_weight)

        weight = l2norm(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.linear(x, weight)

# forced weight normed conv2d and linear
# algorithm 1 in paper

class Conv2d(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size ** 2

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                weight, ps = pack_one(self.weight, 'o *')
                normed_weight = l2norm(weight, eps = self.eps)
                normed_weight = unpack_one(normed_weight, ps, 'o *')
                self.weight.copy_(normed_weight)

        weight = l2norm(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.conv2d(x, weight, padding = 'same')

class Conv1d(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        eps = 1e-4,
        init_dirac = False
    ):
        super().__init__()
        weight = torch.randn(dim_out, dim_in, kernel_size)
        self.weight = nn.Parameter(weight)

        if init_dirac:
            nn.init.dirac_(self.weight)

        self.eps = eps
        self.fan_in = dim_in * kernel_size

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                weight, ps = pack_one(self.weight, 'o *')
                normed_weight = l2norm(weight, eps = self.eps)
                normed_weight = unpack_one(normed_weight, ps, 'o *')
                self.weight.copy_(normed_weight)

        weight = l2norm(self.weight, eps = self.eps) / sqrt(self.fan_in)
        return F.conv1d(x, weight, padding = 'same')

# pixelnorm
# equation (30)

class PixelNorm(Module):
    def __init__(self, dim, eps = 1e-4):
        super().__init__()
        # high epsilon for the pixel norm in the paper
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        dim = self.dim
        return l2norm(x, dim = dim, eps = self.eps) * sqrt(x.shape[dim])

# magnitude preserving sum
# equation (88)
# empirically, they found t=0.3 for encoder / decoder / attention residuals
# and for embedding, t=0.5

class MPAdd(Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x, res):
        a, b, t = x, res, self.t
        num = a * (1. - t) + b * t
        den = sqrt((1 - t) ** 2 + t ** 2)
        return num / den

# mp attention

class MPAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 64,
        num_mem_kv = 4,
        mp_add_t = 0.3,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.pixel_norm = PixelNorm(dim = -1)

        self.dropout = nn.Dropout(dropout)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = Linear(dim, hidden_dim * 3)
        self.to_out = Linear(hidden_dim, dim)

        self.mp_add = MPAdd(t = mp_add_t)

    def forward(self, x):
        res, b = x, x.shape[0]

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        q, k, v = map(self.pixel_norm, (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return self.mp_add(out, res)

# temporal down and upsample

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
        self.conv = Conv1d(dim, dim, 3, init_dirac = True)

    @handle_maybe_channel_last
    @image_or_video_to_time
    def forward(
        self,
        x
    ):
        t = x.shape[-1]
        assert t > 1, 'time dimension must be greater than 1 to be compressed'

        x = interpolate_1d(x, t // 2)
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
        self.conv = Conv1d(dim, dim, 3, init_dirac = True)

    @handle_maybe_channel_last
    @image_or_video_to_time
    def forward(
        self,
        x
    ):
        t = x.shape[-1]
        x = interpolate_1d(x, t * 2)
        return self.conv(x)

# main modules

class MPConvolutionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        conv2d_kernel_size = 3,
        conv1d_kernel_size = 3,
        channel_last = False,
        time_dim = None,
        mp_add_t = 0.3,
        dropout = 0.
    ):
        super().__init__()
        self.time_dim = time_dim
        self.channel_last = channel_last

        self.spatial_conv = nn.Sequential(
            Conv2d(dim, dim, conv2d_kernel_size, 3),
            MPSiLU()
        )

        self.temporal_conv = nn.Sequential(
            Conv1d(dim, dim, conv1d_kernel_size, 3),
            MPSiLU(),
            nn.Dropout(dropout)
        )

        self.proj_out = nn.Sequential(
            Conv1d(dim, dim, 1),
            Gain()
        )

        self.residual_mp_add = MPAdd(t = mp_add_t)

    @handle_maybe_channel_last
    def forward(
        self,
        x,
        batch_size = None
    ):
        residual = x

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

        return self.residual_mp_add(x, residual)

class MPAttentionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        depth = 1,
        time_dim = None,
        channel_last = False,
        mp_add_t = 0.3,
        dropout = 0.,
        **attn_kwargs
    ):
        super().__init__()

        self.time_dim = time_dim
        self.channel_last = channel_last

        self.temporal_attns = ModuleList([])

        for _ in range(depth):
            attn = MPAttention(
                dim = dim,
                dropout = dropout,
                **attn_kwargs
            )

            self.temporal_attns.append(attn)

        self.proj_out = nn.Sequential(
            Linear(dim, dim),
            Gain()
        )

        self.residual_mp_add = MPAdd(t = mp_add_t)

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

        residual = x

        for attn in self.temporal_attns:
            x = attn(x)

        x = self.proj_out(x)

        x = self.residual_mp_add(x, residual)

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
