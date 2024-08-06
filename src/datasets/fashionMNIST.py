import torch
from torch.utils.data import Dataset

from src.config_types import FashionMNISTConfig
from src.consts import C_DATASETS
from src.datasets.base_diffusion_dataset import DiffusionDatasetFactory
from src.types import SPLIT
import torchvision


class FashionMNISTDatasetFactory(DiffusionDatasetFactory):
    def __init__(self, config: FashionMNISTConfig) -> None:
        super().__init__(T=config['T'], t_sampler=config['t_sampler'])
        self.dim = config['dim']

    def load_dataset(self, split: SPLIT) -> Dataset:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.dim, self.dim)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((C_DATASETS.NORMALIZE_MEAN,), (C_DATASETS.NORMALIZE_STD,))
        ])

        return torchvision.datasets.FashionMNIST(
            root=C_DATASETS.FASHION_MNIST_DIR,
            download=True,
            train=split == SPLIT.TRAIN,
            transform=transform
        )


    @staticmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        return x * C_DATASETS.NORMALIZE_STD + C_DATASETS.NORMALIZE_MEAN
