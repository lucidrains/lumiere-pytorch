"""
einstein notation
b - batch
t - time
c - channels
h - height
w - width
"""

from copy import deepcopy
from functools import wraps

import torch
from torch import nn, einsum, Tensor, is_tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import List, Tuple, Optional

from einops import rearrange, pack, unpack, repeat

from optree import tree_flatten, tree_unflatten

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

# extract dimensions using hooks

@beartype
def extract_output_shapes(
    modules: List[Module],
    model: Module,
    model_input,
    model_kwargs: dict = dict()
):
    shapes = []
    hooks = []

    def hook_fn(_, input, output):
        return shapes.append(output.shape)

    for module in modules:
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

    with torch.no_grad():
        model(model_input, **model_kwargs)

    for hook in hooks:
        hook.remove()

    return shapes

# freezing text-to-image, and only learning temporal parameters

@beartype
def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

# function that takes in the entire text-to-video network, and sets the time dimension

def set_time_dim_(model: Module, time_dim: int):
    for model in model.modules():
        if isinstance(model, (AttentionInflationBlock, ConvolutionInflationBlock, TemporalUpsample, TemporalDownsample)):
            model.time_dim = time_dim

# decorator for converting an input tensor from either image or video format to 1d time

def image_or_video_to_time(fn):

    @wraps(fn)
    def inner(
        self,
        x,
        batch_size = None,
        **kwargs
    ):

        is_video = x.ndim == 5

        if is_video:
            batch_size = x.shape[0]
            x = rearrange(x, 'b c t h w -> b h w c t')
        else:
            assert exists(batch_size) or exists(self.time_dim)
            rearrange_kwargs = dict(b = batch_size, t = self.time_dim)
            x = rearrange(x, '(b t) c h w -> b h w c t', **compact_values(rearrange_kwargs))

        x, ps = pack_one(x, '* c t')

        x = fn(self, x, **kwargs)

        x = unpack_one(x, ps, '* c t')

        if is_video:
            x = rearrange(x, 'b h w c t -> b c t h w')
        else:
            x = rearrange(x, 'b h w c t -> (b t) c h w')

        return x

    return inner

# helpers

def Sequential(*modules):
    modules = list(filter(exists, modules))
    return nn.Sequential(*modules)

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, t, *args, **kwargs):
        return self.fn(t, *args, **kwargs) + t

# temporal down and upsample

def init_bilinear_kernel_1d_(conv: Module):
    nn.init.zeros_(conv.weight)
    if exists(conv.bias):
        nn.init.zeros_(conv.bias)

    channels = conv.weight.shape[0]
    bilinear_kernel = Tensor([0.5, 1., 0.5])
    diag_mask = torch.eye(channels).bool()
    conv.weight.data[diag_mask] = bilinear_kernel

class TemporalDownsample(Module):
    def __init__(
        self,
        dim,
        time_dim = None
    ):
        super().__init__()
        self.time_dim = time_dim

        self.conv = nn.Conv1d(dim, dim, kernel_size = 3, stride = 2, padding = 1)
        init_bilinear_kernel_1d_(self.conv)

    @image_or_video_to_time
    def forward(
        self,
        x
    ):
        assert x.shape[-1] > 1, 'time dimension must be greater than 1 to be compressed'

        return self.conv(x)

class TemporalUpsample(Module):
    def __init__(
        self,
        dim,
        time_dim = None
    ):
        super().__init__()
        self.time_dim = time_dim

        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        init_bilinear_kernel_1d_(self.conv)

    @image_or_video_to_time
    def forward(
        self,
        x
    ):
        return self.conv(x)

# main modules

class ConvolutionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        conv2d_kernel_size = 3,
        conv1d_kernel_size = 3,
        groups = 8,
        time_dim = None
    ):
        super().__init__()
        assert is_odd(conv2d_kernel_size)
        assert is_odd(conv1d_kernel_size)

        self.time_dim = time_dim

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

        return x + residual

class AttentionInflationBlock(Module):
    def __init__(
        self,
        *,
        dim,
        depth = 1,
        prenorm = True,
        residual_attn = True,
        time_dim = None,
        **attn_kwargs
    ):
        super().__init__()

        self.time_dim = time_dim

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

    def forward(
        self,
        x,
        batch_size = None
    ):
        is_video = x.ndim == 5
        assert is_video ^ (exists(batch_size) or exists(self.time_dim)), 'either a tensor of shape (batch, channels, time, height, width) is passed in, or (batch * time, channels, height, width) along with `batch_size`'

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

        x = x + residual

        x = unpack_one(x, ps, '* t c')

        if is_video:
            x = rearrange(x, 'b h w t c -> b c t h w')
        else:
            x = rearrange(x, 'b h w t c -> (b t) c h w')

        return x

# post module hook wrapper

class PostModuleHookWrapper(Module):
    def __init__(self, temporal_module: Module):
        super().__init__()
        self.temporal_module = temporal_module

    def forward(self, _, input, output):
        output = self.temporal_module(output)
        return output

def insert_temporal_modules_(modules: List[Module], temporal_modules: ModuleList):
    assert len(modules) == len(temporal_modules)

    for module, temporal_module in zip(modules, temporal_modules):
        module.register_forward_hook(PostModuleHookWrapper(temporal_module))

# main wrapper model around text-to-image

class Lumiere(Module):

    @beartype
    def __init__(
        self,
        model: Module,
        *,
        image_size: int,
        unet_time_kwarg: str,
        conv_module_names: List[str],
        attn_module_names: List[str] = [],
        downsample_module_names: List[str] = [],
        upsample_module_names: List[str] = [],
        channels: int = 3,
        conv_inflation_kwargs: dict = dict(),
        attn_inflation_kwargs: dict = dict()
    ):
        super().__init__()

        model = deepcopy(model)
        freeze_all_layers_(model)
        self.model = model

        # all modules in text-to-images model indexed by name

        modules_dict = {name: module for name, module in model.named_modules() if not isinstance(module, ModuleList)}

        all_module_names = conv_module_names + attn_module_names + downsample_module_names + upsample_module_names

        for name in all_module_names:
            assert name in modules_dict, f'module name {name} not found in the unet {modules_dict.keys()}'

        assert len(downsample_module_names) == len(upsample_module_names), 'one must have the same number of temporal downsample modules as temporal upsample modules'

        # get all modules

        conv_modules = [modules_dict[name] for name in conv_module_names]
        attn_modules = [modules_dict[name] for name in attn_module_names]
        downsample_modules = [modules_dict[name] for name in downsample_module_names]
        upsample_modules = [modules_dict[name] for name in upsample_module_names]

        # mock input, to auto-derive dimensions

        self.channels = channels
        self.image_size = image_size

        mock_images = torch.randn(1, channels, image_size, image_size)
        mock_time = torch.ones((1,))
        unet_kwarg = {unet_time_kwarg: mock_time}

        # get all dimensions

        conv_shapes = extract_output_shapes(conv_modules, self.model, mock_images, unet_kwarg)
        attn_shapes = extract_output_shapes(attn_modules, self.model, mock_images, unet_kwarg)
        downsample_shapes = extract_output_shapes(downsample_modules, self.model, mock_images, unet_kwarg)
        upsample_shapes = extract_output_shapes(upsample_modules, self.model, mock_images, unet_kwarg)

        # all modules

        self.convs = ModuleList([ConvolutionInflationBlock(dim = shape[1], **conv_inflation_kwargs) for shape in conv_shapes])
        self.attns = ModuleList([AttentionInflationBlock(dim = shape[1], **attn_inflation_kwargs) for shape in attn_shapes])
        self.downsamples = ModuleList([TemporalDownsample(dim = shape[1]) for shape in downsample_shapes])
        self.upsamples = ModuleList([TemporalUpsample(dim = shape[1]) for shape in upsample_shapes])

        # insert all the temporal modules with hooks

        insert_temporal_modules_(conv_modules, self.convs)
        insert_temporal_modules_(attn_modules, self.attns)
        insert_temporal_modules_(downsample_modules, self.downsamples)
        insert_temporal_modules_(upsample_modules, self.upsamples)

        # validate modules

        assert len(self.parameters()) > 0, 'no temporal parameters to be learned'

    @property
    def downsample_factor(self):
        return 2 ** len(self.downsamples)

    def parameters(self):
        return [
            *self.convs.parameters(),
            *self.attns.parameters(),
            *self.downsamples.parameters(),
            *self.upsamples.parameters(),
        ]

    @beartype
    def forward(
        self,
        video: Tensor,
        *args,
        **kwargs
    ) -> Tensor:

        assert video.ndim == 5
        batch, channels, time, height, width = video.shape

        assert channels == self.channels
        assert (height, width) == (self.image_size, self.image_size)

        assert divisible_by(time, self.downsample_factor)

        # find all arguments that are Tensors

        all_args = (args, kwargs)
        all_args, pytree_spec = tree_flatten(all_args)

        # and repeat across for each frame of the video
        # so one conditions all images of each video with same input

        for ind, arg in enumerate(all_args):
            if is_tensor(arg):
                all_args[ind] = repeat(arg, 'b ... -> (b t) ...', t = time)

        # turn video into a stack of images

        images = rearrange(video, 'b c t h w -> (b t) c h w')

        # unflatten arguments to be passed into text-to-image model

        all_args = tree_unflatten(pytree_spec, all_args)
        args, kwargs = all_args

        # set the correct time dimension for all temporal layers

        set_time_dim_(self, time)

        # forward all images into text-to-image model

        images = self.model(images, *args, **kwargs)

        # reshape back to denoised video

        return rearrange(images, '(b t) c h w -> b c t h w', b = batch)
