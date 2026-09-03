"""Type-safe configuration schemas using Pydantic v2."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: Literal["fashion_mnist", "cifar10", "synthetic"] = "fashion_mnist"
    image_size: int = Field(default=32, ge=8, le=256)
    channels: int = Field(default=1, ge=1, le=4)
    batch_size: int = Field(default=64, ge=1)
    num_workers: int = Field(default=0, ge=0)
    download: bool = True
    data_dir: str = "data"


class DiffusionModelConfig(BaseModel):
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = Field(default=32, ge=8)
    channel_mults: list[int] = Field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: int = Field(default=2, ge=1)
    time_emb_dim: int = Field(default=128, ge=16)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)


class DiffusionScheduleConfig(BaseModel):
    timesteps: int = Field(default=1000, ge=10)
    schedule: Literal["linear", "cosine"] = "linear"
    beta_start: float = Field(default=0.0001, gt=0.0)
    beta_end: float = Field(default=0.02, gt=0.0)
    sample_timesteps: int = Field(default=50, ge=1)


class DiffusionTrainingConfig(BaseModel):
    epochs: int = Field(default=5, ge=1)
    lr: float = Field(default=2e-4, gt=0.0)
    grad_clip_norm: float = Field(default=1.0, ge=0.0)
    device: str = "auto"
    seed: int = 42
    checkpoint_interval: int = Field(default=1, ge=1)
    checkpoint_dir: str = "outputs/checkpoints/diffusion"
    samples_dir: str = "outputs/samples/diffusion"
    num_sample_images: int = Field(default=16, ge=1)


class DiffusionPipelineConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: DiffusionModelConfig = Field(default_factory=DiffusionModelConfig)
    diffusion: DiffusionScheduleConfig = Field(default_factory=DiffusionScheduleConfig)
    training: DiffusionTrainingConfig = Field(default_factory=DiffusionTrainingConfig)


class GANModelConfig(BaseModel):
    latent_dim: int = Field(default=128, ge=8)
    channels: int = Field(default=1, ge=1, le=4)
    generator_features: int = Field(default=32, ge=8)
    discriminator_features: int = Field(default=32, ge=8)


class WGANConfig(BaseModel):
    lambda_gp: float = Field(default=10.0, ge=0.0)
    n_critic: int = Field(default=5, ge=1)


class GANTrainingConfig(BaseModel):
    epochs: int = Field(default=5, ge=1)
    lr_g: float = Field(default=2e-4, gt=0.0)
    lr_d: float = Field(default=2e-4, gt=0.0)
    beta1: float = Field(default=0.0, ge=0.0, le=1.0)
    beta2: float = Field(default=0.9, ge=0.0, le=1.0)
    device: str = "auto"
    seed: int = 42
    checkpoint_interval: int = Field(default=1, ge=1)
    checkpoint_dir: str = "outputs/checkpoints/gan"
    samples_dir: str = "outputs/samples/gan"
    num_sample_images: int = Field(default=16, ge=1)


class GANPipelineConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: GANModelConfig = Field(default_factory=GANModelConfig)
    wgan_gp: WGANConfig = Field(default_factory=WGANConfig)
    training: GANTrainingConfig = Field(default_factory=GANTrainingConfig)


class EvalSectionConfig(BaseModel):
    num_eval_samples: int = Field(default=100, ge=10)
    batch_size: int = Field(default=32, ge=1)
    device: str = "auto"
    seed: int = 42
    feature_extractor: Literal["efficient_cnn", "inception_v3"] = "efficient_cnn"
    metrics: list[str] = Field(default_factory=lambda: ["fid", "kid"])
    diffusion_checkpoint: str = "outputs/checkpoints/diffusion/best_model.pt"
    gan_checkpoint: str = "outputs/checkpoints/gan/best_model.pt"
    output_report_json: str = "outputs/reports/metrics_summary.json"
    output_report_md: str = "outputs/reports/comparison_report.md"
    output_grid_png: str = "outputs/reports/comparison_grid.png"


class EvalPipelineConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvalSectionConfig = Field(default_factory=EvalSectionConfig)


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_diffusion_config(file_path: str | Path) -> DiffusionPipelineConfig:
    raw = load_yaml(file_path)
    return DiffusionPipelineConfig.model_validate(raw)


def load_gan_config(file_path: str | Path) -> GANPipelineConfig:
    raw = load_yaml(file_path)
    return GANPipelineConfig.model_validate(raw)


def load_eval_config(file_path: str | Path) -> EvalPipelineConfig:
    raw = load_yaml(file_path)
    return EvalPipelineConfig.model_validate(raw)
