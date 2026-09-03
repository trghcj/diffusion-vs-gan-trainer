"""Checkpoint manager for atomic saving and loading of model weights and optimizer state."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages saving, loading, and tracking the best/latest checkpoints.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        state_dict_or_models: dict[str, Any],
        epoch: int,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """
        Atomically saves training state to a .pt file.
        """
        payload: dict[str, Any] = {
            "epoch": epoch,
            "metrics": metrics or {},
        }

        for key, val in state_dict_or_models.items():
            if isinstance(val, (nn.Module, torch.optim.Optimizer)):
                payload[key] = val.state_dict()
            elif isinstance(val, dict):
                payload[key] = val
            else:
                payload[key] = val

        out_path = self.dir / f"{name}.pt"
        tmp_path = self.dir / f"{name}.tmp.pt"

        torch.save(payload, tmp_path)
        if tmp_path.exists():
            tmp_path.replace(out_path)

        return out_path

    def load(self, file_path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
        """
        Loads checkpoint payload.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return torch.load(path, map_location=device, weights_only=False)
