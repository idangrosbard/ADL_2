from abc import ABC
from abc import abstractmethod

import torch
from torch import nn


class AbstractDiffusionModel(nn.Module, ABC):
    @abstractmethod
    def _forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._forward(x, t)

    def count_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
