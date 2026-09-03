"""Image and tensor transformations."""

import numpy as np
import torch
from PIL import Image


def normalize_to_neg_one_to_one(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor with values in [0, 1] to [-1, 1]."""
    return tensor * 2.0 - 1.0


def unnormalize_to_zero_one(tensor: torch.Tensor) -> torch.Tensor:
    """Unnormalize a tensor with values in [-1, 1] to [0, 1]."""
    return torch.clamp((tensor + 1.0) / 2.0, min=0.0, max=1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a normalized tensor (C, H, W) in [-1, 1] to a PIL Image.
    """
    unnorm = unnormalize_to_zero_one(tensor.detach().cpu())
    if unnorm.ndim == 4:
        unnorm = unnorm[0]

    c, h, w = unnorm.shape
    if c == 1:
        arr = (unnorm.squeeze(0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="L")
    elif c == 3:
        arr = (unnorm.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(f"Unsupported channel count: {c}")
