import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader

from genai_trainer.config import DiffusionPipelineConfig
from genai_trainer.data.transforms import unnormalize_to_zero_one
from genai_trainer.models.diffusion.schedule import NoiseSchedule
from genai_trainer.models.diffusion.unet import DiffusionUNet
from genai_trainer.training.checkpoint import CheckpointManager

console = Console()


class DiffusionTrainer:
    """
    Manages end-to-end training and sampling for the Diffusion model.
    """

    def __init__(
        self,
        config: DiffusionPipelineConfig,
        model: DiffusionUNet | None = None,
        schedule: NoiseSchedule | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config

        if device is not None:
            self.device = device
        elif config.training.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.training.device)

        self.model = model or DiffusionUNet(config.model)
        self.model.to(self.device)

        self.schedule = schedule or NoiseSchedule(
            timesteps=config.diffusion.timesteps,
            schedule_type=config.diffusion.schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
        )
        self.schedule.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.training.lr, weight_decay=1e-4
        )
        self.criterion = nn.MSELoss()
        self.ckpt_manager = CheckpointManager(config.training.checkpoint_dir)
        self.samples_dir = Path(config.training.samples_dir)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, x_0: torch.Tensor) -> float:
        """
        Executes a single forward-backward optimization step.
        """
        self.model.train()
        self.optimizer.zero_grad()

        x_0 = x_0.to(self.device)
        b = x_0.shape[0]

        # Sample uniform timesteps t in [0, T-1]
        t = torch.randint(0, self.schedule.timesteps, (b,), device=self.device).long()

        # Add Gaussian noise
        x_noisy, noise = self.schedule.q_sample(x_0, t)

        # Predict added noise
        pred_noise = self.model(x_noisy, t)
        loss = self.criterion(pred_noise, noise)

        loss.backward()
        if self.config.training.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
        self.optimizer.step()

        return float(loss.item())

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Runs one full training epoch.
        """
        total_loss = 0.0
        steps = 0

        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
            steps += 1

        avg_loss = total_loss / max(1, steps)
        return avg_loss

    def train(self, dataloader: DataLoader) -> list[float]:
        """
        Full training loop.
        """
        history: list[float] = []
        best_loss = float("inf")

        console.print(
            f"[bold green][INFO][/bold green] Starting Diffusion training on [bold cyan]{self.device}[/bold cyan] "
            f"for {self.config.training.epochs} epochs..."
        )

        for epoch in range(1, self.config.training.epochs + 1):
            avg_loss = self.train_epoch(dataloader, epoch)
            history.append(avg_loss)

            console.print(
                f"[bold blue][TRAIN][/bold blue] Epoch {epoch:03d}/{self.config.training.epochs:03d} - "
                f"Loss: {avg_loss:.5f}"
            )

            # Checkpoint & sample generation
            if (
                epoch % self.config.training.checkpoint_interval == 0
                or epoch == self.config.training.epochs
            ):
                self.ckpt_manager.save(
                    f"checkpoint_epoch_{epoch:03d}",
                    {"model": self.model, "optimizer": self.optimizer},
                    epoch=epoch,
                    metrics={"loss": avg_loss},
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.ckpt_manager.save(
                        "best_model",
                        {"model": self.model, "optimizer": self.optimizer},
                        epoch=epoch,
                        metrics={"loss": avg_loss},
                    )

                # Generate sample preview
                self.sample_and_save_grid(
                    num_samples=self.config.training.num_sample_images,
                    filename=f"sample_epoch_{epoch:03d}.png",
                    use_ddim=True,
                )

        return history

    @torch.no_grad()
    def sample_ddpm(
        self,
        num_samples: int,
        shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """
        Ancestral DDPM sampling (full timesteps T).
        Returns normalized tensor in [-1, 1].
        """
        self.model.eval()
        c = shape[0] if shape else self.config.dataset.channels
        h = shape[1] if shape else self.config.dataset.image_size
        w = shape[2] if shape else self.config.dataset.image_size

        img = torch.randn((num_samples, c, h, w), device=self.device)

        for step in reversed(range(self.schedule.timesteps)):
            t = torch.full((num_samples,), step, device=self.device, dtype=torch.long)
            pred_noise = self.model(img, t)
            x_recon = self.schedule.predict_start_from_noise(img, t, pred_noise)
            mean, var, log_var = self.schedule.q_posterior_mean_variance(x_recon, img, t)

            if step > 0:
                noise = torch.randn_like(img)
                img = mean + torch.exp(0.5 * log_var) * noise
            else:
                img = mean

        return img

    @torch.no_grad()
    def sample_ddim(
        self,
        num_samples: int,
        steps: int = 50,
        eta: float = 0.0,
        shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """
        Accelerated DDIM sampling (Song et al. 2020) in fewer steps.
        eta = 0.0 is deterministic sampling.
        Returns normalized tensor in [-1, 1].
        """
        self.model.eval()
        c = shape[0] if shape else self.config.dataset.channels
        h = shape[1] if shape else self.config.dataset.image_size
        w = shape[2] if shape else self.config.dataset.image_size

        total_timesteps = self.schedule.timesteps
        time_seq = np.linspace(0, total_timesteps - 1, steps, dtype=int)
        time_seq_prev = np.concatenate([[0], time_seq[:-1]])

        img = torch.randn((num_samples, c, h, w), device=self.device)

        for i in reversed(range(steps)):
            t_curr = time_seq[i]
            t_prev = time_seq_prev[i]

            t_tensor = torch.full((num_samples,), t_curr, device=self.device, dtype=torch.long)
            pred_noise = self.model(img, t_tensor)

            alpha_bar_t = self.schedule.alphas_cumprod[t_curr]
            alpha_bar_prev = (
                self.schedule.alphas_cumprod[t_prev]
                if i > 0
                else torch.tensor(1.0, device=self.device)
            )

            # Predict x_0
            x_0_pred = (img - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            sigma_t = eta * torch.sqrt(
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev)
            )

            # Direction pointing to x_t
            dir_xt = (
                torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * pred_noise
            )

            img = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt
            if eta > 0.0 and i > 0:
                noise = torch.randn_like(img)
                img = img + sigma_t * noise

        return img

    def sample_and_save_grid(
        self,
        num_samples: int = 16,
        filename: str = "sample.png",
        use_ddim: bool = True,
    ) -> Path:
        """
        Samples images and renders a clean, neutral grid saved to disk.
        """
        if use_ddim:
            samples = self.sample_ddim(
                num_samples=num_samples, steps=self.config.diffusion.sample_timesteps
            )
        else:
            samples = self.sample_ddpm(num_samples=num_samples)

        unnorm = unnormalize_to_zero_one(samples.detach().cpu())  # [0, 1]
        grid_rows = int(math.isqrt(num_samples))
        grid_cols = math.ceil(num_samples / grid_rows)

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 1.5, grid_rows * 1.5))
        axes = np.atleast_1d(axes).flatten()

        c = self.config.dataset.channels
        for idx in range(len(axes)):
            ax = axes[idx]
            if idx < num_samples:
                img_t = unnorm[idx]
                if c == 1:
                    ax.imshow(img_t[0].numpy(), cmap="gray")
                else:
                    ax.imshow(img_t.permute(1, 2, 0).numpy())
            ax.axis("off")

        plt.tight_layout(pad=0.5)
        out_path = self.samples_dir / filename
        plt.savefig(out_path, dpi=120, facecolor="white")
        plt.close(fig)

        return out_path
