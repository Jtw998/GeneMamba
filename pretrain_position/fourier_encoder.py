"""
Fourier Position Encoder.

Encodes genomic coordinates (chromosome, bp position) into a fixed-dimensional
embedding using Fourier basis functions (high-frequency sin/cos pairs).

φ(x) = [sin(2π·x·f₁), cos(2π·x·f₁), ..., sin(2π·x·fₖ), cos(2π·x·fₖ)]

fₖ = f₀ · σ^(k-1)  (geometric progression of frequencies)

Key properties:
  - No parameters → no overfitting risk
  - Low frequencies capture global (TAD-level) structure
  - High frequencies capture local (kb-level) precision
  - Learnable projection → adaptive frequency selection
"""

import math
import torch
import torch.nn as nn


class FourierPositionEncoder(nn.Module):
    """
    Fourier feature position encoder.

    Input:  [batch, input_dim]  (input_dim=3: chrom_idx, tss_norm, log_tss_norm)
    Output: [batch, embed_dim]
    """

    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = 256,
        num_frequencies: int = 16,
        f0: float = 1.0,
        sigma: float = 2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Geometric frequency progression: f_k = f0 * sigma^k
        frequencies = [f0 * (sigma ** k) for k in range(num_frequencies)]
        self.register_buffer(
            "frequencies", torch.tensor(frequencies, dtype=torch.float32)
        )

        # Fourier features dim = input_dim * num_frequencies * 2 (sin + cos)
        fourier_dim = input_dim * num_frequencies * 2

        # Learnable projection: Fourier → target embedding dim
        self.projection = nn.Sequential(
            nn.Linear(fourier_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def fourier_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, input_dim]
        returns: [batch, input_dim * num_freq * 2]
        """
        batch = x.shape[0]
        x = x.unsqueeze(-1)                              # [B, D, 1]
        angles = 2 * math.pi * x * self.frequencies     # [B, D, num_freq]
        sin_f = torch.sin(angles)
        cos_f = torch.cos(angles)
        features = torch.cat([sin_f, cos_f], dim=-1)   # [B, D, num_freq*2]
        return features.reshape(batch, -1)               # [B, D * num_freq * 2]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: [batch, input_dim]  genomic coordinates
        returns: [batch, embed_dim] position embeddings
        """
        fourier_features = self.fourier_transform(coords)
        return self.projection(fourier_features)
