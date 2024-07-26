from torch import nn, Tensor, LongTensor
import torch
from typing import Tuple


class StandardSampler(nn.Module):
    def __init__(self, denoiser: nn.Module, t: Tensor, sigmas: Tensor, betas: Tensor, shape: int):
        super(StandardSampler, self).__init__()
        assert sigmas.shape == (T,)
        assert betas.shape == (T,)

        self.denoiser = denoiser
        self.sampling_distribution = torch.distributions.MultivariateNormal(torch.zeros(shape ** 2), torch.eye(shape ** 2))
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('alphas_t', 1 - betas)
        self.register_buffer('alphas_t_bar', torch.cumprod(self.alphas_t, dim=0))
        self.register_buffer('t', t)
        self.register_buffer('shape', torch.tensor(shape))
    
    def get_z(self, t: int, b_size: int) -> Tensor:
        z = torch.zeros(b_size, self.shape, self.shape)
        if t > 0:
            z = self.sampling_distribution.sample((b_size,)).view(b_size, self.shape, self.shape)
        return z
    
    def denoise_step(self, t: float, x: Tensor, z: Tensor) -> Tensor:
        epsilon = self.denoiser(x, t)
        epsilon_scale = (1 - self.alphas_t[t]) / (1 - self.alphas_t_bar[t]).sqrt()
        delta = x - epsilon_scale * epsilon
        delta_scale = 1 / self.alphas_t[t].sqrt()
        x = delta_scale * delta + self.sigmas[t] * z
        return x
    
    def forward(self, b_size: int) -> Tensor:
        x = self.sampling_distribution.sample((b_size,)).view(b_size, self.shape, self.shape)
        for t in self.t:
            z = self.get_z(t, b_size)
            x = self.denoise_step(t, x, z)
        return x
        