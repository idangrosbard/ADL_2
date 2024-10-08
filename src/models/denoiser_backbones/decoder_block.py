import torch
from torch import Tensor
from torch import nn

from .resblock import ResBlock


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5, upsample_width_factor: int = 2,
                 n_convs: int = 1, resblock: bool = False) -> None:
        super().__init__()

        # self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 3, stride=2, padding=1)
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(in_channels, in_channels // upsample_width_factor, 3, 1, 1))

        layers = [nn.Conv2d(2 * in_channels // upsample_width_factor, out_channels, 3, 1, 1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU()]

        if resblock:
            for _ in range(n_convs):
                layers += [ResBlock(out_channels, dropout, 3)]
        else:
            for _ in range(n_convs):
                layers += [
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.ReLU(),
                    nn.Dropout(dropout)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, x_encoded: Tensor) -> Tensor:
        x = self.upsample(x)
        return self.layers(torch.cat([x_encoded, x], dim=1))
