"""Critic (Discriminator) network for WGAN-GP baseline."""

import torch
import torch.nn as nn

from genai_trainer.config import GANModelConfig


class WGANCritic(nn.Module):
    """
    Convolutional Critic for WGAN-GP.
    Uses LayerNorm/InstanceNorm instead of BatchNorm to maintain gradient penalty validity.
    Outputs an unbounded real scalar (Wasserstein distance estimate).
    """

    def __init__(self, config: GANModelConfig) -> None:
        super().__init__()
        df = config.discriminator_features
        c = config.channels

        # 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1
        self.net = nn.Sequential(
            # 32x32 -> 16x16 (No norm in first layer per Gulrajani et al.)
            nn.Conv2d(c, df, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(df, df * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, df * 2),  # Equivalent to LayerNorm for 2D maps
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(df * 2, df * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, df * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 -> 1
            nn.Conv2d(df * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, 32, 32) image tensor
        Returns:
            (B,) real scalar scores
        """
        score = self.net(x)
        return score.view(-1)
