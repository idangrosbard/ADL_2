from torch import nn, Tensor, LongTensor
import torch
from typing import Tuple

from src.models.abstract_diffusion_model import AbstractDiffusionModel
from src.samplers.abstract_sampler import AbstractSampler


def get_sigmas(alphas: Tensor, etas: Tensor) -> Tensor:
    # Taken from equation 16
    sigmas = etas * (1 - alphas[:-1]) / (1 - alphas[1:]).sqrt() * ((1 - alphas[1:]) / alphas[:-1]).sqrt()
    return sigmas


class DDIMSampler(AbstractSampler):
    def __init__(self, alphas: Tensor, taus: Tensor, etas: Tensor, dim: int) -> None:
        super().__init__()
        self.register_buffer('alphas', alphas)
        self.register_buffer('taus', taus)
        sigmas = torch.cat([get_sigmas(alphas, etas), torch.tensor([0.0])])
        self.register_buffer('sigmas', sigmas)
        self.shape = dim
        self.sampling_distribution = torch.distributions.MultivariateNormal(torch.zeros(dim ** 2), torch.eye(dim ** 2))

    def get_z(self, t: int, b_size: int) -> Tensor:
        z = torch.zeros(b_size, 1, self.shape, self.shape)
        if t > 0:
            z = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape)
        return z.to(self.alphas.device)

    def denoise_step(self, denoiser: AbstractDiffusionModel, s: int, x: Tensor, z: Tensor) -> Tensor:
        # From equation 13:
        t = self.taus[s].item()
        t_batch = self.taus[s].repeat(x.shape[0])
        t_1 = self.taus[s - 1].item()
        epsilon = denoiser(x, t_batch)
        epsilon_scale = (((1 - self.alphas[t_1]) / (self.alphas[t_1])).sqrt() - ((1 - self.alphas[t]) / (self.alphas[t])).sqrt()).item()

        resid_scale = (1 / self.alphas[t].sqrt()).item()
        total_scaling = self.alphas[t_1].sqrt().item()

        sigma = self.sigmas[t].item()
        # Adding noise per equation 12
        x = total_scaling * (resid_scale * x - epsilon_scale * epsilon + sigma * z)

        return x


    def forward(self, b_size: int) -> Tensor:
        x = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape).to(self.alphas.device)
        for s in range(self.taus.shape[0] - 1, 0, -1):
            z = self.get_z(self.taus[s], b_size)
            x = self.denoise_step(s, x, z)
        return x
