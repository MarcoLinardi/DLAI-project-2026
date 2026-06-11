"""CIFAR-10 data loaders.

Standard CIFAR-10 setup used across the literature (Frankle 2020,
Wortsman 2022, Ainsworth 2023):
- Train augmentation: RandomCrop(32, pad=4) + RandomHorizontalFlip.
- Per-channel normalization with train-set statistics.
- Test transform: deterministic (ToTensor + Normalize).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Train-set statistics (computed once on CIFAR-10 train split; standard values).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = test_tf
    return train_tf, test_tf


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    root: str | Path = "./data",
    augment: bool = True,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders for CIFAR-10.

    Downloads to `root` on first call.
    `pin_memory` defaults to True iff CUDA is available (faster H2D transfers).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_tf, test_tf = _build_transforms(augment)

    train_set = datasets.CIFAR10(root=str(root), train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=str(root), train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, num_workers=0)
    x, y = next(iter(train_loader))
    print(f"train: {len(train_loader.dataset)} samples, batch={tuple(x.shape)}, labels={tuple(y.shape)}")
    print(f"test : {len(test_loader.dataset)} samples")
    print(f"x stats: mean={x.mean().item():+.3f}, std={x.std().item():.3f}  (should be ~0 / ~1 after Normalize)")
