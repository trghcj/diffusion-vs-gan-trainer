"""WGAN-GP baseline trainer with gradient penalty and alternating optimization."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
from rich.console import Console
from torch.utils.data import DataLoader

from genai_trainer.config import GANPipelineConfig
from genai_trainer.data.transforms import unnormalize_to_zero_one
from genai_trainer.models.gan.critic import WGANCritic
from genai_trainer.models.gan.generator import WGANGenerator
from genai_trainer.training.checkpoint import CheckpointManager

console = Console()


class WGANGPTrainer:
    """
    Manages WGAN-GP training with gradient penalty computation.
    """

    def __init__(
        self,
        config: GANPipelineConfig,
        generator: WGANGenerator | None = None,
        critic: WGANCritic | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config

        if device is not None:
            self.device = device
        elif config.training.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.training.device)

        self.generator = generator or WGANGenerator(config.model)
        self.critic = critic or WGANCritic(config.model)
        self.generator.to(self.device)
        self.critic.to(self.device)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.training.lr_g,
            betas=(config.training.beta1, config.training.beta2),
        )
        self.opt_d = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.training.lr_d,
            betas=(config.training.beta1, config.training.beta2),
        )

        self.ckpt_manager = CheckpointManager(config.training.checkpoint_dir)
        self.samples_dir = Path(config.training.samples_dir)
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def compute_gradient_penalty(
        self, real_samples: torch.Tensor, fake_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the 1-Lipschitz gradient penalty on interpolated points.
        GP = ((||grad(D(x_hat))||_2 - 1)^2).mean()
        """
        batch_size = real_samples.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

        d_interpolates = self.critic(interpolates)
        fake_grad_outputs = torch.ones_like(d_interpolates, requires_grad=False)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake_grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((grad_norm - 1.0) ** 2)
        return gradient_penalty

    def train_critic_step(self, real_samples: torch.Tensor) -> float:
        """
        Single step of critic optimization with gradient penalty.
        """
        self.critic.train()
        self.generator.eval()
        self.opt_d.zero_grad()

        b = real_samples.shape[0]
        real_samples = real_samples.to(self.device)

        z = torch.randn(b, self.config.model.latent_dim, device=self.device)
        fake_samples = self.generator(z)

        d_real = self.critic(real_samples)
        d_fake = self.critic(fake_samples.detach())

        gp = self.compute_gradient_penalty(real_samples, fake_samples.detach())
        loss_d = d_fake.mean() - d_real.mean() + self.config.wgan_gp.lambda_gp * gp

        loss_d.backward()
        self.opt_d.step()

        return float(loss_d.item())

    def train_generator_step(self, batch_size: int) -> float:
        """
        Single step of generator optimization.
        """
        self.generator.train()
        self.critic.eval()
        self.opt_g.zero_grad()

        z = torch.randn(batch_size, self.config.model.latent_dim, device=self.device)
        fake_samples = self.generator(z)
        d_fake = self.critic(fake_samples)

        loss_g = -d_fake.mean()
        loss_g.backward()
        self.opt_g.step()

        return float(loss_g.item())

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> tuple[float, float]:
        """
        Runs one full epoch of WGAN-GP training.
        """
        d_losses: list[float] = []
        g_losses: list[float] = []
        step = 0

        for real_batch in dataloader:
            b = real_batch.shape[0]
            loss_d = self.train_critic_step(real_batch)
            d_losses.append(loss_d)
            step += 1

            if step % self.config.wgan_gp.n_critic == 0:
                loss_g = self.train_generator_step(b)
                g_losses.append(loss_g)

        # Ensure generator was updated at least once
        if not g_losses:
            loss_g = self.train_generator_step(dataloader.batch_size or 16)
            g_losses.append(loss_g)

        avg_d = sum(d_losses) / max(1, len(d_losses))
        avg_g = sum(g_losses) / max(1, len(g_losses))
        return avg_d, avg_g

    def train(self, dataloader: DataLoader) -> dict[str, list[float]]:
        """
        Full training loop for WGAN-GP.
        """
        history: dict[str, list[float]] = {"loss_d": [], "loss_g": []}

        console.print(
            f"[bold green][INFO][/bold green] Starting WGAN-GP training on [bold cyan]{self.device}[/bold cyan] "
            f"for {self.config.training.epochs} epochs..."
        )

        for epoch in range(1, self.config.training.epochs + 1):
            avg_d, avg_g = self.train_epoch(dataloader, epoch)
            history["loss_d"].append(avg_d)
            history["loss_g"].append(avg_g)

            console.print(
                f"[bold magenta][GAN-TRAIN][/bold magenta] Epoch {epoch:03d}/{self.config.training.epochs:03d} - "
                f"Loss D: {avg_d:.5f} | Loss G: {avg_g:.5f}"
            )

            if (
                epoch % self.config.training.checkpoint_interval == 0
                or epoch == self.config.training.epochs
            ):
                self.ckpt_manager.save(
                    f"checkpoint_epoch_{epoch:03d}",
                    {
                        "generator": self.generator,
                        "critic": self.critic,
                        "opt_g": self.opt_g,
                        "opt_d": self.opt_d,
                    },
                    epoch=epoch,
                    metrics={"loss_d": avg_d, "loss_g": avg_g},
                )
                self.ckpt_manager.save(
                    "best_model",
                    {
                        "generator": self.generator,
                        "critic": self.critic,
                        "opt_g": self.opt_g,
                        "opt_d": self.opt_d,
                    },
                    epoch=epoch,
                    metrics={"loss_d": avg_d, "loss_g": avg_g},
                )

                self.sample_and_save_grid(
                    num_samples=self.config.training.num_sample_images,
                    filename=f"gan_sample_epoch_{epoch:03d}.png",
                )

        return history

    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generates images from random normal latent vectors.
        Returns normalized tensor in [-1, 1].
        """
        self.generator.eval()
        z = torch.randn(num_samples, self.config.model.latent_dim, device=self.device)
        return self.generator(z)

    def sample_and_save_grid(self, num_samples: int = 16, filename: str = "gan_sample.png") -> Path:
        """
        Samples images and renders a clean, neutral grid saved to disk.
        """
        samples = self.sample(num_samples=num_samples)
        unnorm = unnormalize_to_zero_one(samples.detach().cpu())

        grid_rows = int(math.isqrt(num_samples))
        grid_cols = math.ceil(num_samples / grid_rows)

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 1.5, grid_rows * 1.5))
        axes = np.atleast_1d(axes).flatten()

        c = self.config.model.channels
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
