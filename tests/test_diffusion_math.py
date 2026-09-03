"""Mathematical tests for diffusion noise schedules and closed-form properties."""

import pytest
import torch

from genai_trainer.models.diffusion.schedule import NoiseSchedule


@pytest.mark.parametrize("schedule_type", ["linear", "cosine"])
def test_schedule_boundary_conditions(schedule_type: str):
    """Test boundary conditions: alpha_bar_0 ~ 1, alpha_bar_T -> 0."""
    schedule = NoiseSchedule(timesteps=1000, schedule_type=schedule_type)

    assert schedule.alphas_cumprod[0] > 0.95
    assert schedule.alphas_cumprod[-1] < 0.05
    assert schedule.betas.min() > 0.0
    assert schedule.betas.max() < 1.0


def test_snr_monotonicity():
    """Verify that Signal-to-Noise Ratio (SNR) strictly decreases over time."""
    schedule = NoiseSchedule(timesteps=100, schedule_type="linear")
    t = torch.arange(100)
    snr = schedule.get_snr(t)

    # Check strictly monotonic decrease
    diffs = snr[1:] - snr[:-1]
    assert (diffs <= 0).all()


def test_q_sample_exact_reconstruction():
    """Verify that predict_start_from_noise inverts q_sample."""
    schedule = NoiseSchedule(timesteps=100)
    x_0 = torch.randn(4, 1, 32, 32)
    t = torch.tensor([10, 25, 50, 75])
    noise = torch.randn_like(x_0)

    x_noisy, _ = schedule.q_sample(x_0, t, noise=noise)
    x_0_pred = schedule.predict_start_from_noise(x_noisy, t, noise)

    assert torch.allclose(x_0, x_0_pred, atol=1e-5)


def test_q_posterior_mean_variance_shapes():
    """Test shape conservation in posterior mean and variance computation."""
    schedule = NoiseSchedule(timesteps=50)
    x_0 = torch.randn(2, 1, 32, 32)
    x_t = torch.randn(2, 1, 32, 32)
    t = torch.tensor([5, 15])

    mean, var, log_var = schedule.q_posterior_mean_variance(x_0, x_t, t)
    assert mean.shape == x_0.shape
    assert var.shape == (2, 1, 1, 1)
    assert log_var.shape == (2, 1, 1, 1)
