from torch import nn, Tensor
import torch
import numpy as np
import math
from .utils import get_alphas_bar, get_alphas



def stirling_approx(t: float, delta_beta: float, beta_0: float) -> float:
    # Perform stirling 
    beta_hat = (1 - beta_0) / delta_beta 
    return t * delta_beta + (beta_hat + 1/2) * math.log(beta_hat) - (beta_hat - t + 1/2) * math.log(beta_hat - t) - t * + 1/12 * (1 / beta_hat - 1 / (beta_hat - t))


def init_approx_t(alpha_bar: Tensor, eta_s: Tensor) -> float:
    # Find the t range according to alpha_bar
    for t in range(1, len(eta_s) - 1):
        if eta_s < alpha_bar[t + 1]:
            return t

            
def approximate_t(eta_s: float, alpha_bar: Tensor, delta_beta: float, beta_0: float, n_steps: int = 20) -> float:
    # Init variables for binary search
    t_min = init_approx_t(alpha_bar, eta_s)
    t_max = t_min + 1
    t = (t_min + t_max) / 2
    
    # Perform binary search
    for t in range(n_steps):
        if stirling_approx(t, delta_beta, beta_0) > 2 * math.log(eta_s):
            t_max = t
            t = (t_min + t) / 2
        
        elif stirling_approx(t, delta_beta, beta_0) < 2 * math.log(eta_s):
            t_min = t
            t = (t_max + t) / 2
        
        else: # stirling_approx(t, delta_beta, beta_0) == eta_s:
            break
    return t
            

class FastDPM(nn.Module):
    def __init__(self, denoiser: nn.Module, shape: int, alpha_bar: Tensor, delta_beta: float, beta_0: float, tau: Tensor | None = None, eta: Tensor | None = None) -> None:
        super(FastDPM, self).__init__()
        self.register_buffer('alpha_bar_t', alpha_bar)
        if (tau is None) and (eta is not None):
            tau = torch.tensor([approximate_t(eta[s], alpha_bar, delta_beta, beta_0) for s in eta.shape[0]])
            
        elif (tau is not None) and (eta is None):
            alpha_bar_s = alpha_bar[tau]
            eta = 1 - (alpha_bar_s[1:] / alpha_bar_s[:-1])

        else:
            raise ValueError('Expected exactly one of ETA and TAU to be none, not both, not neither')
        
        self.register_buffer('tau', tau)
        self.register_buffer('eta', eta)
        self.register_buffer('gamma', 1 - self.eta)
        self.register_buffer('gamma_bar', self.gamma.cumprod(dim=0))
        
        eta_tilde = self.eta * (1 - self.gamma_bar[:-1]) / (1 - self.gamma_bar[1:])
        eta_tilde0 = (self.eta[0] * eta_tilde[-1]).unsqueeze(0)
        eta_tilde = torch.cat([eta_tilde0, eta_tilde])
        self.register_buffer('eta_tilde', eta_tilde)
        self.shape = shape

        self.denoiser = denoiser
        self.sampling_distribution = torch.distributions.MultivariateNormal(torch.zeros(shape ** 2), torch.eye(shape ** 2))
    
        
    def get_z(self, t: int, b_size: int) -> Tensor:
        z = torch.zeros(b_size, 1, self.shape, self.shape)
        if t > 0:
            z = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape)
        return z.to(self.alphas_t_bar.device)
    
    def denoise_step(self, s: int, x_s: Tensor, eps: Tensor) -> Tensor:
        epsilon_hat = self.denoiser(x_s, s)
        epsilon_scale = self.eta[s] / (1 - self.gamma_bar[s]).sqrt()
        noise_scale = self.eta_tilde[s].sqrt()
        overall_scale = 1 / self.gamma[s].sqrt()
        x_s_1 = overall_scale * (x_s - epsilon_scale * epsilon_hat) + noise_scale * eps
        return x_s_1
    
    def forward(self, b_size: int) -> Tensor:
        x = self.sampling_distribution.sample((b_size,)).view(b_size, 1, self.shape, self.shape).to(self.alphas_t_bar.device)
        for t in self.tau:
            z = self.get_z(t, b_size)
            x = self.denoise_step(t, x, z)
        return x