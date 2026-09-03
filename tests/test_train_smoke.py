"""Integration smoke tests for training steps, gradient penalties, and checkpoints."""

import tempfile

import torch

from genai_trainer.config import (
    DatasetConfig,
    DiffusionModelConfig,
    DiffusionPipelineConfig,
    DiffusionScheduleConfig,
    DiffusionTrainingConfig,
    GANModelConfig,
    GANPipelineConfig,
    GANTrainingConfig,
    WGANConfig,
)
from genai_trainer.training.checkpoint import CheckpointManager
from genai_trainer.training.diffusion_trainer import DiffusionTrainer
from genai_trainer.training.gan_trainer import WGANGPTrainer


def test_diffusion_train_step_smoke():
    """Verify a real 1-step forward/backward optimization runs without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = DiffusionPipelineConfig(
            dataset=DatasetConfig(name="synthetic", image_size=32, channels=1, batch_size=2),
            model=DiffusionModelConfig(
                in_channels=1,
                out_channels=1,
                base_channels=16,
                channel_mults=[1, 2],
                num_res_blocks=1,
                time_emb_dim=32,
            ),
            diffusion=DiffusionScheduleConfig(timesteps=50, sample_timesteps=5),
            training=DiffusionTrainingConfig(
                epochs=1,
                checkpoint_dir=f"{tmpdir}/ckpt",
                samples_dir=f"{tmpdir}/samples",
                device="cpu",
            ),
        )

        trainer = DiffusionTrainer(cfg, device=torch.device("cpu"))
        batch = torch.randn(2, 1, 32, 32)
        loss = trainer.train_step(batch)

        assert isinstance(loss, float)
        assert loss > 0.0

        # Fast DDIM sample check
        samples = trainer.sample_ddim(num_samples=2, steps=5)
        assert samples.shape == (2, 1, 32, 32)


def test_wgan_train_step_smoke():
    """Verify 1-step critic and generator optimization with gradient penalty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = GANPipelineConfig(
            dataset=DatasetConfig(name="synthetic", image_size=32, channels=1, batch_size=2),
            model=GANModelConfig(
                latent_dim=32, channels=1, generator_features=16, discriminator_features=16
            ),
            wgan_gp=WGANConfig(lambda_gp=10.0, n_critic=1),
            training=GANTrainingConfig(
                epochs=1,
                checkpoint_dir=f"{tmpdir}/ckpt",
                samples_dir=f"{tmpdir}/samples",
                device="cpu",
            ),
        )

        trainer = WGANGPTrainer(cfg, device=torch.device("cpu"))
        real_batch = torch.randn(2, 1, 32, 32)

        loss_d = trainer.train_critic_step(real_batch)
        assert isinstance(loss_d, float)

        loss_g = trainer.train_generator_step(batch_size=2)
        assert isinstance(loss_g, float)

        samples = trainer.sample(num_samples=2)
        assert samples.shape == (2, 1, 32, 32)


def test_checkpoint_manager_roundtrip():
    """Verify atomic save and load of model weights and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        lin = torch.nn.Linear(4, 2)
        saved_path = mgr.save("model_test", {"model": lin}, epoch=3, metrics={"loss": 0.42})
        assert saved_path.is_file()

        loaded = mgr.load(saved_path)
        assert loaded["epoch"] == 3
        assert loaded["metrics"]["loss"] == 0.42
        assert "model" in loaded
