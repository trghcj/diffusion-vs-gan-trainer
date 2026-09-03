"""WGAN-GP models: Generator and Critic."""

from genai_trainer.models.gan.critic import WGANCritic
from genai_trainer.models.gan.generator import WGANGenerator

__all__ = ["WGANCritic", "WGANGenerator"]
