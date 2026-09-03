"""Tests for FID, KID calculation, comparison visualizer, and report generation."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from genai_trainer.evaluation.metrics import (
    EfficientFeatureExtractor,
    calculate_fid,
    calculate_kid,
)
from genai_trainer.evaluation.reporter import EvaluationReporter
from genai_trainer.evaluation.visualizer import create_comparison_grid


def test_efficient_feature_extractor():
    """Verify feature extractor output dimensionality."""
    extractor = EfficientFeatureExtractor(in_channels=1, feature_dim=64)
    x = torch.randn(4, 1, 32, 32)
    feats = extractor(x)
    assert feats.shape == (4, 64)


def test_fid_kid_identical_distributions():
    """Identical or same-distribution samples should have near-zero FID and low KID."""
    rng = np.random.default_rng(42)
    feats = rng.standard_normal((100, 32))

    fid = calculate_fid(feats, feats)
    assert fid < 1e-4

    f1 = rng.standard_normal((200, 32))
    f2 = rng.standard_normal((200, 32))
    kid = calculate_kid(f1, f2, subsets=50, subset_size=50)
    assert abs(kid) < 0.05


def test_fid_different_distributions():
    """Divergent distributions must produce positive non-zero FID."""
    rng = np.random.default_rng(42)
    feats_a = rng.standard_normal((50, 32))
    feats_b = rng.standard_normal((50, 32)) + 5.0

    fid = calculate_fid(feats_a, feats_b)
    assert fid > 10.0


def test_comparison_grid_rendering():
    """Verify clean comparison grid image generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        real = torch.randn(4, 1, 32, 32)
        diff = torch.randn(4, 1, 32, 32)
        gan = torch.randn(4, 1, 32, 32)
        out_path = Path(tmpdir) / "test_grid.png"

        grid_file = create_comparison_grid(real, diff, gan, out_path, num_columns=4)
        assert grid_file.is_file()
        assert grid_file.stat().st_size > 500


def test_reporter_json_and_markdown():
    """Verify EvaluationReporter generates valid JSON and Markdown artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = EvaluationReporter(output_dir=tmpdir)

        # JSON
        json_path = reporter.write_json_summary({"fid": 12.5, "kid": 0.002})
        assert json_path.is_file()

        # Markdown
        md_path = reporter.write_markdown_report(
            dataset_name="synthetic",
            num_samples=50,
            feature_extractor="efficient_cnn",
            diffusion_metrics={"fid": 15.2, "kid": 0.003},
            gan_metrics={"fid": 22.4, "kid": 0.007},
            model_stats={"diffusion_params": 125000, "gan_params": 120000},
        )
        assert md_path.is_file()
        content = md_path.read_text(encoding="utf-8")
        assert "Quantitative Benchmark Results" in content
        assert "Diffusion (DDPM/DDIM)" in content
