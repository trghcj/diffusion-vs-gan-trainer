"""Command Line Interface for Diffusion-vs-GAN Trainer."""

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from genai_trainer.config import (
    load_diffusion_config,
    load_eval_config,
    load_gan_config,
)
from genai_trainer.data.dataset import get_dataloader
from genai_trainer.evaluation.metrics import (
    calculate_fid,
    calculate_kid,
    extract_features,
    get_feature_extractor,
)
from genai_trainer.evaluation.reporter import EvaluationReporter
from genai_trainer.evaluation.visualizer import create_comparison_grid
from genai_trainer.models.diffusion.unet import DiffusionUNet
from genai_trainer.models.gan.generator import WGANGenerator
from genai_trainer.training.diffusion_trainer import DiffusionTrainer
from genai_trainer.training.gan_trainer import WGANGPTrainer

console = Console()


def train_diffusion_cmd(args: argparse.Namespace) -> None:
    config = load_diffusion_config(args.config)
    if args.epochs:
        config.training.epochs = args.epochs
    if args.dataset:
        config.dataset.name = args.dataset

    console.print(
        f"[bold cyan][DIFFUSION][/bold cyan] Loading dataset: [bold]{config.dataset.name}[/bold]"
    )
    dataloader = get_dataloader(config.dataset, split="train")

    trainer = DiffusionTrainer(config)
    trainer.train(dataloader)
    console.print("[bold green][DONE][/bold green] Diffusion training complete.")


def train_gan_cmd(args: argparse.Namespace) -> None:
    config = load_gan_config(args.config)
    if args.epochs:
        config.training.epochs = args.epochs
    if args.dataset:
        config.dataset.name = args.dataset

    console.print(
        f"[bold magenta][GAN][/bold magenta] Loading dataset: [bold]{config.dataset.name}[/bold]"
    )
    dataloader = get_dataloader(config.dataset, split="train")

    trainer = WGANGPTrainer(config)
    trainer.train(dataloader)
    console.print("[bold green][DONE][/bold green] WGAN-GP training complete.")


def evaluate_cmd(args: argparse.Namespace) -> None:
    eval_config = load_eval_config(args.config)
    if args.num_samples:
        eval_config.evaluation.num_eval_samples = args.num_samples
    if args.dataset:
        eval_config.dataset.name = args.dataset

    dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_str)

    console.print(
        f"[bold green][EVAL][/bold green] Initializing evaluation on [bold]{dev_str}[/bold]..."
    )

    # Load ground truth samples
    eval_loader = get_dataloader(
        eval_config.dataset,
        split="test",
        shuffle=False,
        batch_size=eval_config.evaluation.batch_size,
    )

    real_images_list = []
    total_loaded = 0
    for b in eval_loader:
        real_images_list.append(b)
        total_loaded += b.shape[0]
        if total_loaded >= eval_config.evaluation.num_eval_samples:
            break
    real_images = torch.cat(real_images_list, dim=0)[: eval_config.evaluation.num_eval_samples]

    # Load / sample diffusion
    diff_cfg = load_diffusion_config(args.diffusion_config or "configs/diffusion_config.yaml")
    diffusion_model = DiffusionUNet(diff_cfg.model)
    diff_ckpt_path = Path(eval_config.evaluation.diffusion_checkpoint)
    if diff_ckpt_path.is_file():
        ckpt = torch.load(diff_ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        diffusion_model.load_state_dict(state_dict)
        console.print(
            f"[bold blue][INFO][/bold blue] Loaded Diffusion checkpoint from {diff_ckpt_path}"
        )
    else:
        console.print(
            f"[yellow][WARN][/yellow] Checkpoint {diff_ckpt_path} not found. Running initialized model for evaluation."
        )

    diff_trainer = DiffusionTrainer(diff_cfg, model=diffusion_model, device=device)
    diff_samples = diff_trainer.sample_ddim(
        num_samples=eval_config.evaluation.num_eval_samples,
        steps=min(20, diff_cfg.diffusion.sample_timesteps),
    )

    # Load / sample GAN
    gan_cfg = load_gan_config(args.gan_config or "configs/gan_config.yaml")
    gan_gen = WGANGenerator(gan_cfg.model)
    gan_ckpt_path = Path(eval_config.evaluation.gan_checkpoint)
    if gan_ckpt_path.is_file():
        ckpt = torch.load(gan_ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["generator"] if "generator" in ckpt else ckpt
        gan_gen.load_state_dict(state_dict)
        console.print(f"[bold blue][INFO][/bold blue] Loaded GAN checkpoint from {gan_ckpt_path}")
    else:
        console.print(
            f"[yellow][WARN][/yellow] Checkpoint {gan_ckpt_path} not found. Running initialized model for evaluation."
        )

    gan_trainer = WGANGPTrainer(gan_cfg, generator=gan_gen, device=device)
    gan_samples = gan_trainer.sample(num_samples=eval_config.evaluation.num_eval_samples)

    # Feature extraction & Metrics
    extractor = get_feature_extractor(
        eval_config.evaluation.feature_extractor,
        in_channels=eval_config.dataset.channels,
        device=device,
    )

    real_feats = extract_features(extractor, real_images, device=device)
    diff_feats = extract_features(extractor, diff_samples, device=device)
    gan_feats = extract_features(extractor, gan_samples, device=device)

    diff_fid = calculate_fid(real_feats, diff_feats)
    diff_kid = calculate_kid(real_feats, diff_feats)
    gan_fid = calculate_fid(real_feats, gan_feats)
    gan_kid = calculate_kid(real_feats, gan_feats)

    # Display clean CLI table
    table = Table(title="Generative Benchmark Summary")
    table.add_column("Model Architecture", style="cyan", justify="left")
    table.add_column("FID (lower is better)", style="bold green", justify="right")
    table.add_column("KID (x10^3, lower is better)", style="bold yellow", justify="right")
    table.add_row("Diffusion (DDIM)", f"{diff_fid:.4f}", f"{diff_kid * 1000:.4f}")
    table.add_row("WGAN-GP Baseline", f"{gan_fid:.4f}", f"{gan_kid * 1000:.4f}")
    console.print(table)

    # Visualizer
    grid_path = create_comparison_grid(
        real_images=real_images,
        diffusion_images=diff_samples,
        gan_images=gan_samples,
        output_path=eval_config.evaluation.output_grid_png,
    )
    console.print(f"[bold green][SAVED][/bold green] Comparison grid saved: {grid_path}")

    # Reporter
    diff_params = sum(p.numel() for p in diffusion_model.parameters())
    gan_params = sum(p.numel() for p in gan_gen.parameters())

    reporter = EvaluationReporter(output_dir=Path(eval_config.evaluation.output_report_json).parent)
    reporter.write_json_summary(
        {
            "dataset": eval_config.dataset.name,
            "num_samples": eval_config.evaluation.num_eval_samples,
            "diffusion": {"fid": diff_fid, "kid": diff_kid, "params": diff_params},
            "gan": {"fid": gan_fid, "kid": gan_kid, "params": gan_params},
        },
        filename=Path(eval_config.evaluation.output_report_json).name,
    )

    report_md = reporter.write_markdown_report(
        dataset_name=eval_config.dataset.name,
        num_samples=eval_config.evaluation.num_eval_samples,
        feature_extractor=eval_config.evaluation.feature_extractor,
        diffusion_metrics={"fid": diff_fid, "kid": diff_kid},
        gan_metrics={"fid": gan_fid, "kid": gan_kid},
        model_stats={"diffusion_params": diff_params, "gan_params": gan_params},
        grid_image_rel_path=Path(eval_config.evaluation.output_grid_png).name,
        filename=Path(eval_config.evaluation.output_report_md).name,
    )
    console.print(f"[bold green][SAVED][/bold green] Evaluation report written: {report_md}")


def run_pipeline_cmd(args: argparse.Namespace) -> None:
    """
    Executes the full end-to-end pipeline:
    1. Train tiny Diffusion
    2. Train tiny WGAN-GP
    3. Evaluate both and export report + comparison grid
    """
    console.print(
        "[bold green]=== Starting End-to-End Generative Training & Evaluation Pipeline ===[/bold green]"
    )

    # 1. Diffusion
    diff_args = argparse.Namespace(
        config=args.diffusion_config, epochs=args.epochs, dataset=args.dataset
    )
    train_diffusion_cmd(diff_args)

    # 2. GAN
    gan_args = argparse.Namespace(config=args.gan_config, epochs=args.epochs, dataset=args.dataset)
    train_gan_cmd(gan_args)

    # 3. Evaluation
    eval_args = argparse.Namespace(
        config=args.eval_config,
        diffusion_config=args.diffusion_config,
        gan_config=args.gan_config,
        num_samples=args.num_samples,
        dataset=args.dataset,
    )
    evaluate_cmd(eval_args)
    console.print("[bold green]=== Pipeline Execution Finished Successfully ===[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diffusion-vs-GAN Trainer with CI and Quantitative Evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train-diffusion
    p_diff = subparsers.add_parser("train-diffusion", help="Train tiny diffusion model")
    p_diff.add_argument(
        "--config", default="configs/diffusion_config.yaml", help="Path to YAML config"
    )
    p_diff.add_argument("--epochs", type=int, default=None, help="Override epochs")
    p_diff.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    p_diff.set_defaults(func=train_diffusion_cmd)

    # train-gan
    p_gan = subparsers.add_parser("train-gan", help="Train WGAN-GP baseline")
    p_gan.add_argument("--config", default="configs/gan_config.yaml", help="Path to YAML config")
    p_gan.add_argument("--epochs", type=int, default=None, help="Override epochs")
    p_gan.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    p_gan.set_defaults(func=train_gan_cmd)

    # evaluate
    p_eval = subparsers.add_parser(
        "evaluate", help="Run quantitative evaluation (FID/KID) & comparison"
    )
    p_eval.add_argument(
        "--config", default="configs/eval_config.yaml", help="Evaluation YAML config"
    )
    p_eval.add_argument("--diffusion-config", default="configs/diffusion_config.yaml")
    p_eval.add_argument("--gan-config", default="configs/gan_config.yaml")
    p_eval.add_argument(
        "--num-samples", type=int, default=None, help="Override number of evaluation samples"
    )
    p_eval.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    p_eval.set_defaults(func=evaluate_cmd)

    # run-pipeline
    p_pipe = subparsers.add_parser("run-pipeline", help="Run complete train & evaluation pipeline")
    p_pipe.add_argument("--diffusion-config", default="configs/diffusion_config.yaml")
    p_pipe.add_argument("--gan-config", default="configs/gan_config.yaml")
    p_pipe.add_argument("--eval-config", default="configs/eval_config.yaml")
    p_pipe.add_argument("--epochs", type=int, default=None)
    p_pipe.add_argument("--num-samples", type=int, default=None)
    p_pipe.add_argument("--dataset", type=str, default=None)
    p_pipe.set_defaults(func=run_pipeline_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
