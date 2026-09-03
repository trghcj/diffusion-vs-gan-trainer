"""Quantitative evaluation reporting in Markdown and JSON formats."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EvaluationReporter:
    """
    Generates structured quantitative evaluation reports for CI artifacts and PR comments.
    """

    def __init__(self, output_dir: str | Path = "outputs/reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json_summary(
        self,
        metrics: dict[str, Any],
        filename: str = "metrics_summary.json",
    ) -> Path:
        """
        Saves machine-readable JSON metrics.
        """
        out_path = self.output_dir / filename
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return out_path

    def write_markdown_report(
        self,
        dataset_name: str,
        num_samples: int,
        feature_extractor: str,
        diffusion_metrics: dict[str, float],
        gan_metrics: dict[str, float],
        model_stats: dict[str, Any] | None = None,
        grid_image_rel_path: str = "comparison_grid.png",
        filename: str = "comparison_report.md",
    ) -> Path:
        """
        Renders a clean, professional Markdown report table.
        """
        out_path = self.output_dir / filename
        model_stats = model_stats or {}

        # Determine winner for each metric (lower is better for FID and KID)
        diff_fid = diffusion_metrics.get("fid", 0.0)
        gan_fid = gan_metrics.get("fid", 0.0)
        diff_kid = diffusion_metrics.get("kid", 0.0)
        gan_kid = gan_metrics.get("kid", 0.0)

        fid_winner = "Diffusion" if diff_fid < gan_fid else "WGAN-GP"
        kid_winner = "Diffusion" if diff_kid < gan_kid else "WGAN-GP"

        md_content = f"""# Generative Evaluation Report: Diffusion vs WGAN-GP

- **Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}
- **Dataset:** `{dataset_name}`
- **Evaluation Sample Size:** `{num_samples}` images
- **Feature Extractor:** `{feature_extractor}`

---

## 1. Quantitative Benchmark Results

| Model Architecture | Parameter Count | FID (lower is better) | KID (x10^3, lower is better) |
| :--- | :--- | :--- | :--- |
| **Diffusion (DDPM/DDIM)** | {model_stats.get("diffusion_params", "N/A"):,} | **{diff_fid:.4f}** | **{diff_kid * 1000:.4f}** |
| **WGAN-GP Baseline** | {model_stats.get("gan_params", "N/A"):,} | **{gan_fid:.4f}** | **{gan_kid * 1000:.4f}** |

> **Metric Summary:**
> - Lowest FID achieved by: **{fid_winner}**
> - Lowest KID achieved by: **{kid_winner}**

---

## 2. Qualitative Visual Comparison

The comparison grid below displays side-by-side generations across Ground Truth, Diffusion, and WGAN-GP:

![Qualitative Comparison Grid]({grid_image_rel_path})

---

## 3. Engineering & Architectural Tradeoffs

1. **Sample Diversity vs Generation Speed**:
   - **Diffusion (DDIM)** demonstrates mode coverage with stable loss convergence, requiring 20-50 iterative steps during sampling.
   - **WGAN-GP** produces samples in a single forward pass ($O(1)$ sampling latency), but requires careful 1-Lipschitz critic regularisation via gradient penalties to prevent mode collapse.

2. **Continuous Integration Validation**:
   - Automated checks verify that the forward noise schedule preserves variance $\\bar{{\\alpha}}_T \\to 0$.
   - 1-step smoke tests guarantee that backward optimization gradients propagate without NaN or shape regressions.
"""

        with out_path.open("w", encoding="utf-8") as f:
            f.write(md_content)

        return out_path
