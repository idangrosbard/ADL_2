from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import torch
from torch import Tensor
from torch import nn
from typing_extensions import final


class AbstractSampler(nn.Module, ABC):
    @abstractmethod
    def forward(self, b_size: int) -> Tensor:
        pass
