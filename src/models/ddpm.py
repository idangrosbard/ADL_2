import torch
from torch import Tensor

from ddpm import PositionalEncoding
from denoiser_backbones import get_unet
from src.config_types import DDPMConfig
from src.config_types import FashionMNISTConfig
from src.models.abstract_diffusion_model import AbstractDiffusionModel


class DDPMModel(AbstractDiffusionModel):
    def __init__(self, config: DDPMConfig, dataset_config: FashionMNISTConfig) -> None:
        super().__init__()
        assert (
                (
                        (torch.log(torch.tensor(dataset_config["dim"])) / torch.log(torch.tensor(2)))
                        // config['unet']['depth']
                ) >= 1
        ), f'Cannot perform more downsampling than input size allows, input_dim={dataset_config["dim"]}, model_depth={config["unet"]["depth"]}'

        self.unet = get_unet(config['unet'])
        self.pe = PositionalEncoding(
            d_model=dataset_config['dim'],  # TODO: kept for backward compatability, but looks weird
            max_T=dataset_config['T'],
            length=config['length'],
            dim=dataset_config['dim'],
        )

    def _forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_t_emb = self.pe(x, t)
        sigma_hat = self.unet(x_t_emb)
        return sigma_hat
