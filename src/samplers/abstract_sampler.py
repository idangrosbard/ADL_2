from abc import ABC
from abc import abstractmethod

from torch import Tensor
from torch import nn

from src.models.abstract_diffusion_model import AbstractDiffusionModel


class AbstractSampler(nn.Module, ABC):
    @abstractmethod
    def forward(self, diffusion_model: AbstractDiffusionModel, b_size: int) -> Tensor:
        pass
