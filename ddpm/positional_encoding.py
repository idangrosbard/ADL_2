import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_T: int, length: int, dim: int):
        super().__init__()
        self.d_model = d_model
        self.max_T = max_T
        self.length = length
        self.dim = dim ** 2

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        pe = torch.zeros(x.shape[0], self.d_model ** 2, device=x.device)

        denom = self.length ** (torch.arange(0, self.dim, device=x.device) / self.dim)
        # Set the positional encoding to be sinus for even indices
        mat = t.unsqueeze(-1) / denom
        mat = mat.squeeze(1)

        pe[:, 0::2] = torch.sin(mat[:, 0::2])
        # cosine for odd indices
        pe[:, 1::2] = torch.cos(mat[:, 1::2])

        pe = pe.reshape(-1, 1, self.d_model, self.d_model)
        return x + pe
