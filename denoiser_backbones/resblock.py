from torch import nn, Tensor


class ResBlock(nn.Module):
    def __init__(self, n_channels: int, dropout: float = 0.5, n_convs: int = 3) -> None:
        super(ResBlock, self).__init__()
        
        for _ in range(n_convs):
            layers = layers + [
                    nn.Conv2d(n_channels, n_channels, 3, 1, 1),
                    nn.ReLU(),
                    nn.Dropout(dropout)]
        
        self.layers = nn.Sequential(*layers)
    

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)
        