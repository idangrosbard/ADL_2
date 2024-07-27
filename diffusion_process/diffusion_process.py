import torch
from torch import nn, Tensor, LongTensor
from typing import Tuple
from .utils import get_alphas_bar, get_alphas


class DiffusionProcess(nn.Module):
    def __init__(self, betas_t: Tensor, dim: int):
        self.register_buffer('input_dim', dim)
        self.register_buffer('betas_t', betas_t)

        # From the definitions above equation 4
        self.register_buffer('alphas_t', get_alphas(self.betas_t))
        self.register_buffer('alpha_bar_t', get_alphas_bar(self.alphas_t))

        # From equations 9-10
        self.noise_distribution = torch.distributions.MultivariateNormal(torch.zeros(dim ** 2), torch.eye(dim ** 2))
        

    def sample(self, x_0: Tensor, t: LongTensor) -> Tuple[Tensor, Tensor]:
        # x_0 of shape (batch_size, dim, dim)
        # t of shape (batch_size,)
        # implement the diffusion process according to equations 11-12
        epsilon = self.noise_distribution.sample((x_0.shape[0],)).reshape(x_0.shape[0], self.input_dim, self.input_dim)
        
        x_t = x_0 * self.alpha_bar_t[t].sqrt() + (epsilon * (1 - self.alpha_bar_t[t]).sqrt())
        return x_t, epsilon
