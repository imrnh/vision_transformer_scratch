import torch
import torch.nn as nn
from mhsa import MultiHeadedSelfAttention


class Encoder(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(Encoder, self).__init__()

        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio

        self.norm_1 = nn.LayerNorm(self.hidden_d)
        self.norm_2 = nn.LayerNorm(self.hidden_d)
        self.mhsa = MultiHeadedSelfAttention(self.hidden_d, self.n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_d, out_features=hidden_d * mlp_ratio),
            nn.GELU(),
            nn.Linear(in_features=hidden_d * mlp_ratio, out_features=hidden_d),
        )

    def forward(self, x):
        rn = self.mhsa(self.norm_1(x))
        out = x + self.mhsa(self.norm_1(x))
        out = out + self.mlp(self.norm_2(out))
        return out
