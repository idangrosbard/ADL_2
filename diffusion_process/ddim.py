from torch import nn, Tensor, LongTensor
import torch
from typing import Tuple


def get_sigmas(alphas: Tensor, etas: Tensor) -> Tensor:
    # Taken from equation 16
    sigmas = etas * (1 - alphas[:-1]) / (1 - alphas[1:]).sqrt() * ((1 - alphas[1:]) / alphas[:-1]).sqrt()
    return sigmas


class DDIMSampler(nn.Module):
    def __init__(self, denoiser: nn.Module, alphas: Tensor, taus: Tensor, etas: Tensor, dim: int) -> None:
        super(DDIMSampler, self).__init__()
        self.denoiser = denoiser
        self.register_buffer('alphas', alphas)
        self.register_buffer('taus', taus)
        sigmas = get_sigmas(alphas, etas)
        self.register_buffer('sigmas', sigmas)
        self.shape = dim
        self.sampling_distribution = torch.distributions.MultivariateNormal(torch.zeros(dim ** 2), torch.eye(dim ** 2))
    
    def get_z(self, t: int, b_size: int) -> Tensor:
        z = torch.zeros(b_size, 1, self.shape, self.shape)
        if t > 0:
            z = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape)
        return z.to(self.alphas.device)
    
    def expand_to_x(self, parameter: Tensor, x: Tensor) -> Tensor:
        for dim_size in x.shape[1:]:
            parameter = parameter.unsqueeze(-1).repeat_interleave(dim_size, -1)
        return parameter
        
    def denoise_step(self, i: int, t: Tensor, x: Tensor, z: Tensor) -> Tensor:
        # From equation 13:
        t_1 = self.taus[i + 1].repeat(t.shape[0])
        epsilon = self.denoiser(x, t)
        epsilon_scale = ((1 - self.alphas[t_1]) / (self.alphas[t_1])).sqrt() - ((1 - self.alphas[t]) / (self.alphas[t])).sqrt()
        
        resid_scale = 1 / self.alphas[t].sqrt()
        total_scaling = self.alphas[t_1].sqrt()
        epsilon_scale = self.expand_to_x(epsilon_scale, x)
        resid_scale = self.expand_to_x(resid_scale, x)
        total_scaling = self.expand_to_x(total_scaling, x)
        sigma = self.expand_to_x(self.sigmas[t], x)
        # Adding noise per equation 12
        x = total_scaling * (resid_scale * x - epsilon_scale * epsilon + sigma * z)

        return x
    
    
    def forward(self, b_size: int) -> Tensor:
        x = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape).to(self.alphas.device)
        for i in range(self.taus.shape[0] - 1):
            z = self.get_z(self.taus[i], b_size)
            t_batch = self.taus[i].repeat(b_size)
            x = self.denoise_step(i, t_batch, x, z)
        return x
        