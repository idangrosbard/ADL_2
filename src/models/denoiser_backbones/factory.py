from src.config_types import UNetConfig
from .decoder_block import DecoderBlock
from .encoder_block import EncoderBlock
from .u_net import UNet


def get_unet(unet_config: UNetConfig) -> UNet:
    encoder_blocks = []
    decoder_blocks = []

    in_channels = unet_config['init_width']
    encoder_blocks.append(
        EncoderBlock(
            in_channels=unet_config['n_channels'],
            out_channels=in_channels,
            dropout=unet_config['p_dropout'],
            n_convs=unet_config['n_convs'],
            resblock=unet_config['resblock']
        ),
    )

    for i in range(1, unet_config['depth'] - 1):
        encoder_blocks.append(EncoderBlock(
            in_channels=in_channels,
            out_channels=in_channels * unet_config['width_expansion_factor'],
            dropout=unet_config['p_dropout'],
            n_convs=unet_config['n_convs'],
            resblock=unet_config['resblock']),
        )
        in_channels *= unet_config['width_expansion_factor']

    bridge = EncoderBlock(
        in_channels=in_channels,
        out_channels=in_channels * unet_config['width_expansion_factor'],
        dropout=unet_config['p_dropout'],
        n_convs=unet_config['n_convs'],
        resblock=unet_config['resblock']
    )
    in_channels *= unet_config['width_expansion_factor']

    for i in range(unet_config['depth'] - 2):
        decoder_blocks.append(
            DecoderBlock(
                in_channels=in_channels,
                out_channels=in_channels // unet_config['width_expansion_factor'],
                dropout=unet_config['p_dropout'],
                upsample_width_factor=unet_config['width_expansion_factor'],
                n_convs=unet_config['n_convs'],
                resblock=unet_config['resblock']
            ),
        )
        in_channels //= unet_config['width_expansion_factor']

    decoder_blocks.append(DecoderBlock(
        in_channels=in_channels,
        out_channels=unet_config['n_channels'],
        dropout=unet_config['p_dropout'],
        upsample_width_factor=unet_config['width_expansion_factor'],
        n_convs=unet_config['n_convs'],
        resblock=unet_config['resblock']
    ))

    return UNet(encoder_blocks, bridge, decoder_blocks, unet_config['kernel_size'])
