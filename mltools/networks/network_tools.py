import torch
import torch.nn as nn
import numpy as np


@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        torch.nn.init.zeros_(p.data)
    return module


def get_conv(in_channels, out_channels, **kwargs):
    def_params = {
        "dim": 2,
        "kernel_size": 3,
        "padding": 1,
        "stride": 1,
        "padding_mode": "zeros",
        "dilation": 1,
        "groups": 1,
        "init": lambda x: x,
        "transposed": False,
    }
    def_params.update(kwargs)
    dim = def_params.pop("dim")
    transposed = def_params.pop("transposed")
    init = def_params.pop("init")
    if dim == 2:
        conv = nn.ConvTranspose2d if transposed else nn.Conv2d
        return init(conv(in_channels, out_channels, **def_params))
    elif dim == 3:
        conv = nn.ConvTranspose3d if transposed else nn.Conv3d
        return init(conv(in_channels, out_channels, **def_params))


def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    T=1000,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    timesteps *= T
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1 or timesteps.ndim == 2
    assert embedding_dim % 2 == 0
    num_timescales = embedding_dim // 2
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        base=10.0,
        device=timesteps.device,
    )
    if timesteps.ndim == 1:
        emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
        return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)
    else:
        emb = timesteps.to(dtype)[:,:,None] * inv_timescales[None,None, :] # (B, T, D/2)
        return torch.cat([emb.sin(), emb.cos()], dim=2) # (B, T, D)
