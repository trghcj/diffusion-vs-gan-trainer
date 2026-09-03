"""Quantitative metrics calculation: FID (Fréchet Inception Distance) and KID (Kernel Inception Distance)."""

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EfficientFeatureExtractor(nn.Module):
    """
    A lightweight, deterministic convolutional feature extractor.
    Enables sub-second FID/KID computation in CI pipelines and CPU environments
    without requiring heavy 100MB+ pre-trained model downloads.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, feature_dim)

        # Initialize with deterministic orthogonal weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > 1 and self.conv[0].in_channels == 1:
            x = x.mean(dim=1, keepdim=True)
        elif x.shape[1] == 1 and self.conv[0].in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


def get_feature_extractor(
    name: str = "efficient_cnn", in_channels: int = 1, device: torch.device | str = "cpu"
) -> nn.Module:
    """
    Returns feature extractor network.
    """
    dev = torch.device(device)
    if name == "efficient_cnn":
        extractor = EfficientFeatureExtractor(in_channels=in_channels)
    elif name == "inception_v3":
        try:
            import torchvision.models as models

            weights = models.Inception_V3_Weights.DEFAULT
            model = models.inception_v3(weights=weights)
            model.fc = nn.Identity()  # 2048-dim features
            extractor = model
        except Exception:
            # Fallback to efficient extractor if internet is offline or download fails
            extractor = EfficientFeatureExtractor(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown feature extractor: {name}")

    extractor.to(dev)
    extractor.eval()
    return extractor


@torch.no_grad()
def extract_features(
    model: nn.Module,
    tensor_or_loader: torch.Tensor | DataLoader,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """
    Extracts deep features from a batch of images or a DataLoader.
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    features_list: list[np.ndarray] = []

    if isinstance(tensor_or_loader, torch.Tensor):
        b = tensor_or_loader.shape[0]
        batch_size = 32
        for i in range(0, b, batch_size):
            chunk = tensor_or_loader[i : i + batch_size].to(dev)
            feats = model(chunk).detach().cpu().numpy()
            features_list.append(feats)
    else:
        for batch in tensor_or_loader:
            batch = batch.to(dev)
            feats = model(batch).detach().cpu().numpy()
            features_list.append(feats)

    return np.concatenate(features_list, axis=0)


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray, eps: float = 1e-6) -> float:
    """
    Calculates the Fréchet Inception Distance between two feature sets:
    FID = ||mu_1 - mu_2||^2 + Tr(sigma_1 + sigma_2 - 2 * sqrt(sigma_1 * sigma_2))
    """
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)

    mu2 = np.mean(fake_features, axis=0)
    sigma2 = np.cov(fake_features, rowvar=False)

    # Ensure 2D arrays even if single feature dimension
    if sigma1.ndim == 0:
        sigma1 = np.array([[sigma1]])
    if sigma2.ndim == 0:
        sigma2 = np.array([[sigma2]])

    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps

    # Matrix square root of product
    covmean, _ = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

    if not np.isfinite(covmean).all():
        covmean = scipy.linalg.sqrtm(
            (sigma1 + np.eye(sigma1.shape[0]) * 1e-3).dot(sigma2 + np.eye(sigma2.shape[0]) * 1e-3),
            disp=False,
        )[0]

    # Handle numerical imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    return max(0.0, fid)


def calculate_kid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    subsets: int = 50,
    subset_size: int = 50,
) -> float:
    """
    Calculates Kernel Inception Distance (KID) via unbiased polynomial kernel MMD:
    k(x, y) = ((1/d) * x^T y + 1)^3
    """
    n_real = real_features.shape[0]
    n_fake = fake_features.shape[0]
    m = min(n_real, n_fake, subset_size)
    d = real_features.shape[1]

    if m < 2:
        return 0.0

    mmd_estimates: list[float] = []
    rng = np.random.default_rng(42)

    for _ in range(subsets):
        idx_r = rng.choice(n_real, size=m, replace=False)
        idx_f = rng.choice(n_fake, size=m, replace=False)

        x = real_features[idx_r]
        y = fake_features[idx_f]

        # Polynomial kernel: (1/d * X Y^T + 1)^3
        k_xx = (x.dot(x.T) / d + 1.0) ** 3
        k_yy = (y.dot(y.T) / d + 1.0) ** 3
        k_xy = (x.dot(y.T) / d + 1.0) ** 3

        # Unbiased estimator: zero diagonal
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)

        mmd = k_xx.sum() / (m * (m - 1)) + k_yy.sum() / (m * (m - 1)) - 2.0 * k_xy.sum() / (m * m)
        mmd_estimates.append(float(mmd))

    return float(np.mean(mmd_estimates))
