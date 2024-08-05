from torch import nn, Tensor
from .resblock import ResBlock


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5, n_convs: int = 1, resblock: bool = False) -> None:
        super(EncoderBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                  nn.LayerNorm(), nn.ReLU()]
        
        if resblock:
            for _ in range(n_convs):
                layers = layers + [ResBlock(out_channels, dropout, 3)]

        else:
            for _ in range(n_convs):
                layers = layers + [
                        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                        nn.ReLU(),
                        nn.Dropout(dropout)]
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)