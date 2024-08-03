from .decoder_block import DecoderBlock
from .encoder_block import EncoderBlock
from .u_net import UNet
from torch import nn, Tensor


def get_unet(depth: int, n_channels: int = 1, p_dropout: float = 0.1, init_width: int = 64, width_expansion_factor: int = 2, n_convs: int = 2) -> UNet:
    encoder_blocks = []
    decoder_blocks = []
    
    in_channels = init_width
    encoder_blocks.append(EncoderBlock(n_channels, in_channels, p_dropout))
    
    for i in range(1, depth - 1):
        encoder_blocks.append(EncoderBlock(in_channels, in_channels * width_expansion_factor, p_dropout, n_convs))
        in_channels *= width_expansion_factor

    bridge = EncoderBlock(in_channels, in_channels * width_expansion_factor, p_dropout, n_convs)
    in_channels *= width_expansion_factor

    for i in range(depth - 2):
        decoder_blocks.append(DecoderBlock(in_channels, in_channels // width_expansion_factor, p_dropout, width_expansion_factor, n_convs))
        in_channels //= width_expansion_factor
    
    decoder_blocks.append(DecoderBlock(in_channels, n_channels, p_dropout, width_expansion_factor, n_convs))
    return UNet(encoder_blocks, bridge, decoder_blocks)