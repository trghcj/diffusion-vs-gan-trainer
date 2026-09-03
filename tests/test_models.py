"""Tests for network architectures, tensor dimensions, and gradient backpropagation."""

import torch

from genai_trainer.config import DiffusionModelConfig, GANModelConfig
from genai_trainer.models.diffusion.embeddings import SinusoidalTimeEmbeddings
from genai_trainer.models.diffusion.unet import DiffusionUNet
from genai_trainer.models.gan.critic import WGANCritic
from genai_trainer.models.gan.generator import WGANGenerator


def test_sinusoidal_time_embeddings():
    """Verify time embedding dimensions and output variance."""
    emb_module = SinusoidalTimeEmbeddings(emb_dim=64)
    t = torch.tensor([0, 50, 100, 500])
    emb = emb_module(t)
    assert emb.shape == (4, 64)
    assert not torch.isnan(emb).any()


def test_diffusion_unet_forward():
    """Verify Diffusion UNet preserves spatial resolution (B, C, H, W)."""
    cfg = DiffusionModelConfig(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_mults=[1, 2],
        num_res_blocks=1,
        time_emb_dim=32,
    )
    unet = DiffusionUNet(cfg)
    x = torch.randn(2, 1, 32, 32)
    t = torch.tensor([10, 20])
    out = unet(x, t)

    assert out.shape == (2, 1, 32, 32)

    # Test backward pass gradient propagation
    loss = out.sum()
    loss.backward()
    for name, param in unet.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"


def test_wgan_generator_and_critic():
    """Verify WGAN Generator and Critic shapes and gradient flow."""
    cfg = GANModelConfig(
        latent_dim=64, channels=1, generator_features=16, discriminator_features=16
    )
    generator = WGANGenerator(cfg)
    critic = WGANCritic(cfg)

    z = torch.randn(3, 64)
    fakes = generator(z)
    assert fakes.shape == (3, 1, 32, 32)
    assert fakes.min() >= -1.0
    assert fakes.max() <= 1.0

    scores = critic(fakes)
    assert scores.shape == (3,)

    loss = scores.sum()
    loss.backward()
    assert z.grad is None  # Input was not tracked
    for name, param in critic.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for {name}"
