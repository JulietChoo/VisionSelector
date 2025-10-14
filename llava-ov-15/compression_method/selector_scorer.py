import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TransformerScorer(nn.Module):
    """
    Lightweight Transformer Scorer using simplified attention for importance scoring.
    Initializes outputs close to zero to minimally interfere with the original attention_sum.
    """
    def __init__(self, in_features: int, hidden_dim: int = 1792, init_scale: float = 0.0001):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # Lightweight projection layers for Key and Query
        self.k_proj = nn.Linear(in_features, hidden_dim)
        self.q_proj = nn.Linear(in_features, hidden_dim)

        # Initialize all weights to produce near-zero output
        self._init_near_zero(init_scale)

    def _init_near_zero(self, scale: float = 0.0001):
        """Initialize parameters with small values to ensure near-zero output."""
        # Initialize k_proj and q_proj weights with small scale
        nn.init.normal_(self.k_proj.weight, std=scale)
        nn.init.zeros_(self.k_proj.bias)

        nn.init.normal_(self.q_proj.weight, std=scale)
        nn.init.zeros_(self.q_proj.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates importance scores via simplified self-attention.

        Args:
            x (torch.Tensor): Visual tokens of shape [B, N, D]
                             (B: batch size, N: token count, D: embedding dim)
        Returns:
            torch.Tensor: Learned importance scores, shape [B, N]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Generate Key and Query representations
        k = self.k_proj(x)  # [B, N, hidden_dim]
        q = self.q_proj(x)  # [B, N, hidden_dim]

        # Simplified self-attention: compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # [B, N, N]
        # Scores are the mean of attention weights across the attention dimension
        scores = attn_weights.mean(dim=-1)

        return scores


if __name__ == '__main__':
    model = TransformerScorer(in_features=3584)
    x = torch.randn(1,220,3584)
    y = model(x)
    print(y.shape)
    print(y)