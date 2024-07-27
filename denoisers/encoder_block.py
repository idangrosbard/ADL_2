from torch import nn, Tensor

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool, dropout: float = 0.5) -> None:
        super(EncoderBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(),
                  nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                  nn.ReLU(),
                  nn.Dropout(dropout)]
        # if downsample:
        #     layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)