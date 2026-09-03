"""Generator network for WGAN-GP baseline."""

import torch
import torch.nn as nn

from genai_trainer.config import GANModelConfig


class WGANGenerator(nn.Module):
    """
    Convolutional Generator mapping latent vector z ~ N(0, I) to image in [-1, 1].
    Outputs 32x32 images by default.
    """

    def __init__(self, config: GANModelConfig) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.channels = config.channels
        gf = config.generator_features

        # Map latent vector z to 4x4 feature map
        self.initial_dense = nn.Sequential(
            nn.Linear(config.latent_dim, gf * 4 * 4 * 4),
            nn.BatchNorm1d(gf * 4 * 4 * 4),
            nn.ReLU(True),
        )

        self.gf = gf

        # Upsample blocks: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(gf * 4, gf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gf * 2),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(gf * 2, gf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gf),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(gf, gf // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(gf // 2),
            nn.ReLU(True),
            # Final projection to channels with Tanh -> [-1, 1]
            nn.Conv2d(gf // 2, config.channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent vectors
        Returns:
            (B, channels, 32, 32) generated images in [-1, 1]
        """
        h = self.initial_dense(z)
        h = h.view(-1, self.gf * 4, 4, 4)
        return self.blocks(h)
