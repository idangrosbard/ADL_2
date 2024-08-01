from .decoder_block import DecoderBlock
from .encoder_block import EncoderBlock
from .u_net import UNet
from torch import nn, Tensor


def get_unet(depth: int, n_channels: int = 1, p_dropout: float = 0.1) -> UNet:
    encoder_blocks = []
    decoder_blocks = []
    
    in_channels = 64
    encoder_blocks.append(EncoderBlock(n_channels, in_channels, p_dropout))
    
    for i in range(1, depth - 1):
        encoder_blocks.append(EncoderBlock(in_channels, in_channels * 2, p_dropout))
        in_channels *= 2

    bridge = EncoderBlock(in_channels, in_channels * 2, p_dropout)
    in_channels *= 2

    for i in range(depth - 2):
        decoder_blocks.append(DecoderBlock(in_channels, in_channels // 2, p_dropout))
        in_channels //= 2
    
    decoder_blocks.append(DecoderBlock(in_channels, n_channels, p_dropout))
    return UNet(encoder_blocks, bridge, decoder_blocks)