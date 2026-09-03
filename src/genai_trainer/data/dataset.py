"""Dataset definitions, pre-processing, and dataloader creation."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from genai_trainer.config import DatasetConfig
from genai_trainer.data.transforms import normalize_to_neg_one_to_one


class SyntheticShapeDataset(Dataset):
    """
    A lightweight, synthetic dataset generating circles and rectangles.
    Guarantees 100% offline, zero-network, sub-second execution for CI/testing.
    """

    def __init__(
        self,
        length: int = 500,
        image_size: int = 32,
        channels: int = 1,
        seed: int = 42,
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.channels = channels
        self.seed = seed

        rng = np.random.default_rng(seed)
        self.data: list[torch.Tensor] = []

        y, x = np.ogrid[:image_size, :image_size]
        center = image_size // 2

        for _ in range(length):
            img = np.zeros((image_size, image_size), dtype=np.float32)
            shape_type = rng.integers(0, 2)
            radius = rng.integers(image_size // 6, image_size // 3)
            offset_x = rng.integers(-image_size // 6, image_size // 6 + 1)
            offset_y = rng.integers(-image_size // 6, image_size // 6 + 1)

            cx = center + offset_x
            cy = center + offset_y

            if shape_type == 0:  # Circle
                mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
                img[mask] = 1.0
            else:  # Square
                r = radius
                x1 = max(0, cx - r)
                x2 = min(image_size, cx + r)
                y1 = max(0, cy - r)
                y2 = min(image_size, cy + r)
                img[y1:y2, x1:x2] = 1.0

            tensor_img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W) in [0, 1]
            if channels == 3:
                tensor_img = tensor_img.repeat(3, 1, 1)

            # Normalize to [-1, 1]
            normalized = normalize_to_neg_one_to_one(tensor_img)
            self.data.append(normalized)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def get_dataset(config: DatasetConfig, split: str = "train") -> Dataset:
    """
    Instantiate the specified dataset with normalization to [-1, 1].
    """
    is_train = split == "train"

    if config.name == "synthetic":
        return SyntheticShapeDataset(
            length=500 if is_train else 100,
            image_size=config.image_size,
            channels=config.channels,
            seed=42 if is_train else 1337,
        )

    # Lazy import of torchvision
    import torchvision
    import torchvision.transforms as T

    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    transform_list: list[torch.nn.Module] = []
    if config.channels == 3 and config.name == "fashion_mnist":
        transform_list.append(T.Grayscale(num_output_channels=3))
    elif config.channels == 1 and config.name == "cifar10":
        transform_list.append(T.Grayscale(num_output_channels=1))

    if config.image_size != 28 and config.name == "fashion_mnist":
        transform_list.append(T.Resize((config.image_size, config.image_size)))
    elif config.image_size != 32 and config.name == "cifar10":
        transform_list.append(T.Resize((config.image_size, config.image_size)))

    transform_list.extend(
        [
            T.ToTensor(),  # [0, 1]
            T.Normalize([0.5] * config.channels, [0.5] * config.channels),  # [-1, 1]
        ]
    )

    transform = T.Compose(transform_list)

    if config.name == "fashion_mnist":
        dataset = torchvision.datasets.FashionMNIST(
            root=str(data_dir),
            train=is_train,
            download=config.download,
            transform=transform,
        )
    elif config.name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=str(data_dir),
            train=is_train,
            download=config.download,
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.name}")

    # Dataset wrapper that yields only image tensor
    class ImageOnlyWrapper(Dataset):
        def __init__(self, ds: Dataset) -> None:
            self.ds = ds

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, idx: int) -> torch.Tensor:
            img, _ = self.ds[idx]
            return img

    return ImageOnlyWrapper(dataset)


def get_dataloader(
    config: DatasetConfig,
    split: str = "train",
    shuffle: bool | None = None,
    batch_size: int | None = None,
) -> DataLoader:
    """
    Create a standard PyTorch DataLoader.
    """
    dataset = get_dataset(config, split=split)
    do_shuffle = (split == "train") if shuffle is None else shuffle
    bs = batch_size or config.batch_size

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=do_shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
