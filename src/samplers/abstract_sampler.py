from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import torch
from torch import Tensor
from torch import nn
from typing_extensions import final

from src.models.abstract_diffusion_model import AbstractDiffusionModel


class AbstractSampler(nn.Module, ABC):
    @abstractmethod
    def forward(self, diffusion_model: AbstractDiffusionModel, b_size: int) -> Tensor:
        pass
