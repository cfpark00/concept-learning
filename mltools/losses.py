import torch
import torch.nn as nn


class MultiScaleMSE(nn.Module):
    def __init__(self, N, dim, k_func=lambda k: k**1.0, dtype=torch.float32):
        super().__init__()
        self.N = N
        self.dim = dim
        self.dtype = dtype
        k_arr = torch.fft.fftfreq(N, 1 / N).to(dtype=dtype)
        k_p_dims = torch.meshgrid(*(k_arr for _ in range(dim)), indexing="ij")
        ksqs = torch.stack([k_p_dim**2 for k_p_dim in k_p_dims]).sum(dim=0)
        ks = torch.sqrt(ksqs)
        self.register_buffer("weights", k_func(ks))
        self.weights /= self.weights.sum()
        self.weights *= self.N**self.dim

    def forward(self, x_in, x_tar):
        assert x_in.shape == x_tar.shape
        assert x_in.shape[-self.dim :] == (self.N,) * self.dim
        assert x_in.ndim == (self.dim + 2)
        x_in_k = torch.fft.fftn(x_in, dim=tuple(range(-self.dim, 0)))
        x_tar_k = torch.fft.fftn(x_tar, dim=tuple(range(-self.dim, 0)))
        loss = (
            (x_in_k - x_tar_k).abs().pow(2)
            * self.weights[None, None]
            / (self.N**self.dim)
        )
        # return mean except batch dim
        return loss.mean(dim=tuple(range(1, self.dim + 2)))
