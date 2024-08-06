from torch import nn, Tensor, LongTensor
import torch
from typing import Tuple

from src.models.abstract_diffusion_model import AbstractDiffusionModel
from src.samplers.abstract_sampler import AbstractSampler
from src.types import TimeStep
from .utils import get_alphas_bar, get_alphas, get_sigmas


class StandardSampler(AbstractSampler):
    def __init__(self, T: TimeStep, betas: Tensor, shape: int, deterministic: bool = False):
        super().__init__()
        assert betas.shape == (T,)

        self.sampling_distribution = torch.distributions.MultivariateNormal(torch.zeros(shape ** 2), torch.eye(shape ** 2))
        self.register_buffer('sigmas', get_sigmas(T, betas, deterministic))
        self.register_buffer('alphas_t', get_alphas(betas))
        self.register_buffer('alphas_t_bar', get_alphas_bar(self.alphas_t))
        self.register_buffer('T', torch.tensor(T))
        self.register_buffer('shape', torch.tensor(shape))
        self.deterministic = deterministic
    
    def get_z(self, t: TimeStep, b_size: int) -> Tensor:
        z = torch.zeros(b_size, 1, self.shape, self.shape)
        if (t > 0) and (not self.deterministic):
            z = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape)
        return z.to(self.alphas_t_bar.device)
    
    def denoise_step(self, denoiser: AbstractDiffusionModel, t: TimeStep, t_batch: Tensor, x: Tensor, z: Tensor) -> Tensor:
        # Algorithm step taken from "Algorithm 2" in DDPM paper
        epsilon = denoiser(x, t_batch)
        epsilon_scale = ((1 - self.alphas_t[t]) / (1 - self.alphas_t_bar[t]).sqrt()).item()

        delta = x - epsilon_scale * epsilon
        delta_scale = (1 / self.alphas_t[t].sqrt()).item()
        sigmas = self.sigmas[t].item()

        x = delta_scale * delta + torch.clamp(sigmas * z, -1, 1)
        return x
    
    def forward(self, b_size: int) -> Tensor:
        # Sample random noise
        x = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape).to(self.alphas_t_bar.device)

        # Iterate for t steps...
        for t in range(self.T - 1, -1, -1):
            t_batch = torch.tensor([t for _ in range(b_size)], device=x.device)
            z = self.get_z(t, b_size)
            x = self.denoise_step(t, t_batch, x, z)
        return x
