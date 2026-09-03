"""Sinusoidal time embeddings and projection MLP for continuous diffusion timesteps."""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbeddings(nn.Module):
    """
    Computes sinusoidal time step embeddings and projects them through a 2-layer MLP.
    Formula:
        PE(t, 2i)   = sin(t / 10000^(2i/dim))
        PE(t, 2i+1) = cos(t / 10000^(2i/dim))
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: 1D Tensor of shape (batch_size,) with values in [0, T-1].
        Returns:
            Tensor of shape (batch_size, emb_dim).
        """
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)

        device = timesteps.device
        half_dim = self.emb_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.emb_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros(embedding.shape[0], 1, device=device)], dim=-1
            )

        return self.mlp(embedding)
