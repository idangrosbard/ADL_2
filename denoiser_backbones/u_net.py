from torch import nn, Tensor
import torch
from typing import List


class UNet(nn.Module):
    def __init__(self, encoder_blocks: List[nn.Module], bridge: nn.Module, decoder_blocks: List[nn.Module]) -> None:
        super(UNet, self).__init__()
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.maxpool = nn.MaxPool2d(2)
        self.bridge = bridge
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x: Tensor) -> Tensor:
        history = []

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            history.append(x)
            x = self.maxpool(x)
        
        x = self.bridge(x)

        for decoder_block in self.decoder_blocks:
            x_encoded = history.pop()
            x = decoder_block(x, x_encoded)

        return x