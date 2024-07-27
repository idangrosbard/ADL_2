import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from torch import Tensor, distributions
from PIL import Image


class DiffusionDataset(Dataset):
    def __init__(self, data: Dataset, T: int = 200, t_sampler: distributions.Distribution | str = 'uniform') -> None:
        self.data = data
        self.T = T
        if t_sampler == 'uniform':
            t_sampler = distributions.Uniform(1, T)
        elif t_sampler == 'constant':
            t_sampler = distributions.Uniform(T - 1, T)

        self.t_sampler = t_sampler

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Tensor]:
        img = self.data[idx][0]
        t = self.t_sampler.sample((1,)).long()
        return img, t


def get_dataloaders(batch_size: int = 1, T: int = 200, dim: int = 32) -> Tuple[DataLoader, DataLoader]:
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Create datasets
    train_ds = torchvision.datasets.FashionMNIST('./fashionMNIST', download=True, train=True, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST('./fashionMNIST', download=True, train=False, transform=transform)

    # Add sampling of time step
    train_ds = DiffusionDataset(train_ds, T, 'uniform')
    test_ds = DiffusionDataset(test_ds, T, 'constant')

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4)

    return train_loader, test_loader



def denormalize(x: Tensor) -> Tensor:
    return x * 0.3081 + 0.1307