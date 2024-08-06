from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import torch
from torch import nn
from typing_extensions import final


class AbstractDiffusionModel(nn.Module, ABC):
    @abstractmethod
    def forward_sequence_model(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    @final
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence_model(x, t)

    def count_params(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())
