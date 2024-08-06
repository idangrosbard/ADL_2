from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Tuple

import torch
from PIL.Image import Image
from torch import distributions
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing_extensions import assert_never

from src.consts import PATHS
from src.types import SPLIT
from src.types import T_SAMPLER
from src.types import TimeStep


class DiffusionDataset(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            T: TimeStep,
            t_sampler: T_SAMPLER
    ):
        self.dataset = dataset
        self.T = T
        if t_sampler == T_SAMPLER.UNIFORM:
            self.t_sampler = distributions.Uniform(1, T)
        elif t_sampler == T_SAMPLER.CONSTANT:
            self.t_sampler = distributions.Uniform(T - 1, T)
        else:
            assert_never(t_sampler)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[Image, torch.Tensor]:
        img = self.dataset[idx][0]
        t = self.t_sampler.sample((1,)).long()
        return img, t


class DiffusionDatasetFactory(ABC):
    def __init__(self, T: TimeStep, t_sampler: T_SAMPLER):
        self.T = T
        self.t_sampler = t_sampler

    def get_dataset(self, split: SPLIT) -> DiffusionDataset:
        dataset = DiffusionDataset(
            dataset=self.load_dataset(split),
            T=self.T,
            t_sampler=self.t_sampler
        )
        return dataset

    @abstractmethod
    def load_dataset(self, split: SPLIT) -> Dataset:
        pass

    @staticmethod
    @abstractmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        pass
