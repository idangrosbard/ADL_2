import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_T: int = 200):
        super().__init__()
        self.d_model = d_model
        self.max_T = max_T
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        pe = torch.zeros(x.shape[0], self.d_model ** 2, device=x.device)
        
        # Set the positional encoding to be sinus for even indices 
        pe[:, 0::2] = torch.sin(t.unsqueeze(-1) / (2 * self.max_T) ** (torch.arange(0, self.d_model ** 2, 2, device=t.device).float() / self.d_model ** 2)).squeeze(1)
        # cosine for odd indices
        pe[:, 1::2] = torch.cos(t.unsqueeze(-1) / (2 * self.max_T) ** (torch.arange(1, self.d_model ** 2, 2, device=t.device).float() / self.d_model ** 2)).squeeze(1)

        pe = pe.reshape(-1, 1, self.d_model, self.d_model)
        return x + pe

