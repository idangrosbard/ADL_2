from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import torch
from torch import nn
from typing_extensions import final


class AbstractDiffusionModel(nn.Module, ABC):
    @abstractmethod
    def _forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    @final
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._forward(x, t)

    def count_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
