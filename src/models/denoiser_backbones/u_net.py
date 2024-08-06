from typing import List

from torch import Tensor
from torch import nn

from src.models.denoiser_backbones.decoder_block import DecoderBlock
from src.models.denoiser_backbones.encoder_block import EncoderBlock


class UNet(nn.Module):
    def __init__(
            self,
            encoder_blocks: List[EncoderBlock],
            bridge: EncoderBlock,
            decoder_blocks: List[DecoderBlock],
            kernel_size: int,
    ) -> None:
        super().__init__()
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.maxpool = nn.MaxPool2d(kernel_size)
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
