"""Tests for dataset loading, normalization, and transforms."""

import torch

from genai_trainer.config import DatasetConfig
from genai_trainer.data.dataset import SyntheticShapeDataset, get_dataloader
from genai_trainer.data.transforms import (
    normalize_to_neg_one_to_one,
    tensor_to_pil,
    unnormalize_to_zero_one,
)


def test_normalization_inversion():
    """Verify that unnormalize(normalize(x)) reconstructs original [0, 1] tensor."""
    x = torch.rand(4, 1, 32, 32)
    normed = normalize_to_neg_one_to_one(x)
    assert normed.min() >= -1.0
    assert normed.max() <= 1.0

    recon = unnormalize_to_zero_one(normed)
    assert torch.allclose(x, recon, atol=1e-5)


def test_synthetic_shape_dataset():
    """Test the zero-network synthetic dataset generates valid normalized tensors."""
    ds = SyntheticShapeDataset(length=10, image_size=32, channels=1, seed=42)
    assert len(ds) == 10

    sample = ds[0]
    assert sample.shape == (1, 32, 32)
    assert sample.min() >= -1.0
    assert sample.max() <= 1.0


def test_synthetic_dataloader_batch():
    """Test dataloader yields correctly shaped batches."""
    cfg = DatasetConfig(name="synthetic", image_size=32, channels=1, batch_size=4)
    loader = get_dataloader(cfg, split="train")

    batch = next(iter(loader))
    assert batch.shape == (4, 1, 32, 32)
    assert batch.dtype == torch.float32


def test_tensor_to_pil():
    """Test converting [-1, 1] tensor to PIL image."""
    tensor = torch.zeros(1, 32, 32)
    img = tensor_to_pil(tensor)
    assert img.size == (32, 32)
    assert img.mode == "L"
