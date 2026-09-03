"""Diffusion noise scheduling and forward sampling math."""

import math
from typing import Literal

import torch
import torch.nn as nn


class NoiseSchedule(nn.Module):
    """
    Computes and caches Gaussian diffusion variance schedules:
    beta_t, alpha_t, alpha_cumprod_t, and closed-form forward sampling q(x_t | x_0).
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule_type: Literal["linear", "cosine"] = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif schedule_type == "cosine":
            # Cosine schedule proposed by Nichol & Dhariwal (2021)
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, min=1e-5, max=0.999)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        # Calculations for diffusion q(x_t | x_0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # Register non-parameter buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _extract(self, tensor: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract values from 1D tensor at indices t, and reshape to match x_shape for broadcasting.
        """
        batch_size = t.shape[0]
        out = tensor.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Closed-form forward diffuse: q(x_t | x_0)
        Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 given x_t and predicted noise:
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        """
        sqrt_recip_alpha = self._extract(1.0 / self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod, t, x_t.shape
        )
        return sqrt_recip_alpha * x_t - sqrt_recipm1_alpha * noise

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance for DDPM step: q(x_{t-1} | x_t, x_0)
        """
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_var

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute Signal-to-Noise Ratio (SNR) at timestep t:
        SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
        """
        alpha_bar = self.alphas_cumprod[t]
        return alpha_bar / (1.0 - alpha_bar)
