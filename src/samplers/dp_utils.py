import torch


def get_betas(T: int, beta_1: float = 1e-4, beta_T: float = 0.02) -> torch.Tensor:
    betas = torch.linspace(beta_1, beta_T, T)
    return betas


def get_alphas(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1 - betas
    return alphas


def get_alphas_bar(alphas: torch.Tensor) -> torch.Tensor:
    alphas_bar = torch.cumprod(alphas, dim=0)
    return alphas_bar


def get_sigmas(T: int, betas: torch.Tensor, x_0_fixed: bool = False) -> torch.Tensor:
    if not x_0_fixed:
        sigmas = betas.sqrt()
    else:
        alpha_bar = get_alphas_bar(get_alphas(betas))
        sigmas_square = (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:]) * betas[1:]
        sigmas = sigmas_square.sqrt()
        sigmas = torch.cat([torch.tensor([0]), sigmas])
    return sigmas
