"""Time-conditioned Residual U-Net architecture for score/noise estimation."""

import torch
import torch.nn as nn

from genai_trainer.config import DiffusionModelConfig
from genai_trainer.models.diffusion.embeddings import SinusoidalTimeEmbeddings


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with continuous time-embedding injection and GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        # Inject time embedding
        time_h = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_h
        h = self.block2(h)
        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class DiffusionUNet(nn.Module):
    """
    Lightweight, modular time-conditioned UNet for epsilon prediction.
    """

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__()
        self.config = config

        self.time_embed = SinusoidalTimeEmbeddings(config.time_emb_dim)

        self.init_conv = nn.Conv2d(
            config.in_channels, config.base_channels, kernel_size=3, padding=1
        )

        channels = [config.base_channels * m for m in config.channel_mults]

        # Encoder (Downsampling path)
        self.downs = nn.ModuleList()
        in_c = config.base_channels
        self.down_channels: list[int] = [in_c]

        for i, out_c in enumerate(channels):
            for _ in range(config.num_res_blocks):
                self.downs.append(
                    ResidualBlock(
                        in_channels=in_c,
                        out_channels=out_c,
                        time_emb_dim=config.time_emb_dim,
                        dropout=config.dropout,
                    )
                )
                in_c = out_c
                self.down_channels.append(in_c)

            if i < len(channels) - 1:
                self.downs.append(Downsample(in_c))
                self.down_channels.append(in_c)

        # Mid block (Bottleneck)
        self.mid1 = ResidualBlock(in_c, in_c, config.time_emb_dim, dropout=config.dropout)
        self.mid2 = ResidualBlock(in_c, in_c, config.time_emb_dim, dropout=config.dropout)

        # Decoder (Upsampling path)
        self.ups = nn.ModuleList()
        for i, out_c in reversed(list(enumerate(channels))):
            for _ in range(config.num_res_blocks + 1):
                skip_c = self.down_channels.pop()
                self.ups.append(
                    ResidualBlock(
                        in_channels=in_c + skip_c,
                        out_channels=out_c,
                        time_emb_dim=config.time_emb_dim,
                        dropout=config.dropout,
                    )
                )
                in_c = out_c

            if i > 0:
                self.ups.append(Upsample(in_c))

        # Final projection to out_channels (epsilon)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, in_c), in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, config.out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting added noise epsilon.
        Args:
            x: (B, C, H, W) noisy image batch.
            timesteps: (B,) integer or continuous timesteps.
        Returns:
            (B, C, H, W) predicted noise tensor.
        """
        t_emb = self.time_embed(timesteps)

        h = self.init_conv(x)
        skips = [h]

        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
                skips.append(h)
            else:  # Downsample
                h = layer(h)
                skips.append(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)
            else:  # Upsample
                h = layer(h)

        return self.out_conv(h)
