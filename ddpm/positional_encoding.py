import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_T: int = 200):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_T, d_model ** 2)
        position = torch.arange(0, max_T).unsqueeze(1).float()
        # Set the positional encoding to be sinus for even indices 
        pe[:, 0::2] = torch.sin(position / (2 * max_T) ** (torch.arange(0, d_model ** 2, 2).float() / d_model ** 2))
        # cosine for odd indices
        pe[:, 1::2] = torch.cos(position / (2 * max_T) ** (torch.arange(1, d_model ** 2, 2).float() / d_model ** 2))
    
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor, t: int) -> Tensor:
        pe = self.pe[:, t]
        pe.reshape(-1, 1, self.d_model, self.d_model)
        print(pe.shape)
        print(x.shape)
        return x + pe

