from torch import nn, Tensor, LongTensor
import torch
from typing import Tuple
from .utils import get_alphas_bar, get_alphas


class StandardSampler(nn.Module):
    def __init__(self, denoiser: nn.Module, T: int, sigmas: Tensor, betas: Tensor, shape: int):
        super(StandardSampler, self).__init__()
        assert sigmas.shape == (T,)
        assert betas.shape == (T,)

        self.denoiser = denoiser
        self.sampling_distribution = torch.distributions.MultivariateNormal(torch.zeros(shape ** 2), torch.eye(shape ** 2))
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('alphas_t', get_alphas(betas))
        self.register_buffer('alphas_t_bar', get_alphas_bar(self.alphas_t))
        self.register_buffer('T', torch.tensor(T))
        self.register_buffer('shape', torch.tensor(shape))
    
    def get_z(self, t: int, b_size: int) -> Tensor:
        z = torch.zeros(b_size, 1, self.shape, self.shape)
        if t > 0:
            z = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape).to(self.alphas_t_bar.device)
        return z
    
    def denoise_step(self, t: int, x: Tensor, z: Tensor) -> Tensor:
        epsilon = self.denoiser(x, t)
        epsilon_scale = (1 - self.alphas_t[t]) / (1 - self.alphas_t_bar[t]).sqrt()
        delta = x - epsilon_scale * epsilon
        delta_scale = 1 / self.alphas_t[t].sqrt()
        x = delta_scale * delta + self.sigmas[t] * z
        return x
    
    def forward(self, b_size: int) -> Tensor:
        x = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape).to(self.alphas_t_bar.device)
        for t in range(self.T):
            z = self.get_z(t, b_size)
            x = self.denoise_step(t, x, z)
        return x
        