from torch import nn, Tensor
import torch
from typing import Callable


class HeunSampler(nn.Module):
    def __init__(self, denoiser: nn.Module, t: Tensor, sigma: nn.Module, sigma_dot: nn.Module, shape: int):
        super(HeunSampler, self).__init__()
        
        self.denoiser = denoiser
        self.sampling_distribution = torch.distributions.MultivariateNormal(torch.zeros(shape ** 2), torch.eye(shape ** 2))
        self.sigma = sigma
        self.sigma_dot = sigma_dot
        self.register_buffer('t', torch.tensor(t))
        self.register_buffer('shape', torch.tensor(shape))
    
    def forward(self, b_size: int) -> Tensor:
        x_0 = self.sampling_distribution.sample((b_size,)).view(b_size, self.shape, self.shape)
        x_0 *= self.sigma(self.t[0])

        x_i = x_0

        for i in range(self.t.shape[0]):
            t_i = self.t[i]
            d_i = self.sigma_dot(t_i) / self.sigma(t_i) * (x_i - self.denoiser(x_i, t_i))
            x_i_1 = x_i - d_i
        return x