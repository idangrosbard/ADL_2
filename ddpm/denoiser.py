from torch import nn, Tensor



class DDPMModel(nn.Module):
    def __init__(self, unet: nn.Module, pe: nn.Module) -> None:
        super(DDPMModel, self).__init__()
        self.unet = unet
        self.pe = pe

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_t_emb = self.pe(x, t)
        sigma_hat = self.unet(x_t_emb)
        return sigma_hat
