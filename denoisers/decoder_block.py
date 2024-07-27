from torch import nn, Tensor
import torch

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5) -> None:
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, 3, stride=2, padding=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(),
                  nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                  nn.ReLU(),
                  nn.Dropout(dropout)]
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: Tensor, x_encoded: Tensor) -> Tensor:
        x = self.upsample(x)
        return self.layers(torch.cat([x_encoded, x], dim=1))