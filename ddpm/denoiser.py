from torch import nn, Tensor



class EDMModel(nn.Module):
    def __init__(self, unet: nn.Module, pe: nn.Module) -> None:
        super(EDMModel, self).__init__()
        self.unet = unet
        self.pe = pe

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.pe(x, t)
        x = self.unet(x)
        return x
