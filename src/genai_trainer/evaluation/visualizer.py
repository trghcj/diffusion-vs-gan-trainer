"""Comparison visualizer for Real vs Diffusion vs GAN side-by-side grids."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from genai_trainer.data.transforms import unnormalize_to_zero_one


def create_comparison_grid(
    real_images: torch.Tensor,
    diffusion_images: torch.Tensor,
    gan_images: torch.Tensor,
    output_path: str | Path,
    num_columns: int = 8,
) -> Path:
    """
    Renders a clean, 3-row comparison grid:
      Row 1: Ground Truth Real Images
      Row 2: Diffusion Model Samples (DDPM/DDIM)
      Row 3: WGAN-GP Baseline Samples

    Uses academic styling (horizontal labels, neutral margins, zero blur/neon effects).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = min(num_columns, real_images.shape[0], diffusion_images.shape[0], gan_images.shape[0])

    real_unnorm = unnormalize_to_zero_one(real_images[:n].detach().cpu())
    diff_unnorm = unnormalize_to_zero_one(diffusion_images[:n].detach().cpu())
    gan_unnorm = unnormalize_to_zero_one(gan_images[:n].detach().cpu())

    rows = [
        ("Ground Truth (Real)", real_unnorm),
        ("Diffusion (DDIM)", diff_unnorm),
        ("WGAN-GP (Baseline)", gan_unnorm),
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=n,
        figsize=(n * 1.5 + 2.5, 5.5),
        gridspec_kw={"wspace": 0.08, "hspace": 0.30},
    )
    plt.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.08)

    for row_idx, (label, batch) in enumerate(rows):
        c = batch.shape[1]
        for col_idx in range(n):
            ax = axes[row_idx, col_idx]
            img = batch[col_idx]

            if c == 1:
                ax.imshow(img[0].numpy(), cmap="gray")
            else:
                ax.imshow(img.permute(1, 2, 0).numpy())

            ax.set_xticks([])
            ax.set_yticks([])

            # Clean solid border
            for spine in ax.spines.values():
                spine.set_color("#333333")
                spine.set_linewidth(1.0)

            # Horizontal label to the left of the row, perfectly aligned
            if col_idx == 0:
                ax.text(
                    -0.25,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    color="#222222",
                )

    plt.suptitle(
        "Qualitative Comparison: Real vs Diffusion vs WGAN-GP",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return path
