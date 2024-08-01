from torch import nn, Tensor, LongTensor
import torch
from typing import Tuple


def get_sigmas(alphas: Tensor, etas: Tensor) -> Tensor:
    # Taken from equation 16
    sigmas = etas * (1 - alphas[:-1]) / (1 - alphas[1:]).sqrt() * ((1 - alphas[1:]) / alphas[:-1]).sqrt()
    return sigmas


class DDIMSampler(nn.Module):
    def __init__(self, denoiser: nn.Module, alphas: Tensor, taus: Tensor, etas: Tensor):
        super(DDIMSampler, self).__init__()
        self.denoiser = denoiser
        self.register_buffer('alphas', alphas)
        self.register_buffer('taus', taus)
        sigmas = get_sigmas(alphas, etas)
        self.register_buffer('sigmas', sigmas)
    
    def get_z(self, t: int, b_size: int) -> Tensor:
        z = torch.zeros(b_size, 1, self.shape, self.shape)
        if t > 0:
            z = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape)
        return z.to(self.alphas_t_bar.device)
        
    def denoise_step(self, i: int, t: Tensor, x: Tensor, z: Tensor) -> Tensor:
        # From equation 13:
        t_1 = self.taus[i + 1].repeat(t.shape[0])
        epsilon = self.denoiser(x, t)
        epsilon_scale = ((1 - self.alphas[t_1]) / (self.alphas[t_1])).sqrt() - ((1 - self.alphas[t]) / (self.alphas[t])).sqrt()
        epsilon_scale.unsqueeze(-1).repeat(1,1,1)

        for s in x.shape[1:]:
            epsilon_scale = epsilon_scale.unsqueeze(-1).repeat_interleave(s, -1)

        resid_scale = 1 / self.alphas[t].sqrt()
        total_scaling = self.alphas[t_1].sqrt()
        # Adding noise per equation 12
        x = total_scaling * (resid_scale * x - epsilon_scale * epsilon + self.sigmas[t] * z)

        return x
    
    
    def forward(self, b_size: int) -> Tensor:
        x = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape).to(self.alphas_t_bar.device)
        for i in range(1, self.taus.shape[0]):
            z = self.get_z(i, b_size)
            t_batch = self.taus[i].repeat(b_size)
            x = self.denoise_step(i, t_batch, x, z)
        return x
        