from torch import nn, Tensor
import torch
from typing import List


class UNet(nn.Module):
    def __init__(self, encoder_blocks: List[nn.Module], decoder_blocks: List[nn.Module]) -> None:
        super(UNet, self).__init__()
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x: Tensor) -> Tensor:
        history = []

        for encoder_block in self.encoder_blocks:
            history.append(x)
            x = encoder_block(x)
            print('encoded', x.shape)
        
        for h in history:
            print(h.shape)
        
        for decoder_block in self.decoder_blocks:
            x_encoded = history.pop()
            x = decoder_block(x, x_encoded)
            print('decoded', x.shape)

        return x