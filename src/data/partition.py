"""Disjoint partitioning of the CIFAR-10 training set.

Used by experiment E1 (settings B/D) to train pair members on disjoint
data slices. The test loader is never partitioned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.data.cifar import _build_transforms


def make_disjoint_indices(n_samples: int, n_parts: int, seed: int) -> list[list[int]]:
    """Deterministic partition of {0, ..., n_samples-1} into `n_parts` disjoint slices.

    The same `seed` always yields the same partition, independent of any
    other RNG state (uses a local numpy Generator).
    """
    if n_parts < 1:
        raise ValueError(f"n_parts must be >= 1, got {n_parts}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    chunks = np.array_split(perm, n_parts)
    return [chunk.tolist() for chunk in chunks]


def get_cifar10_partition_loaders(
    part_id: int,
    n_parts: int,
    partition_seed: int,
    batch_size: int = 128,
    num_workers: int = 2,
    root: str | Path = "./data",
    augment: bool = True,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build (train_partition_loader, full_test_loader) for partition `part_id`.

    Test loader is the full CIFAR-10 test set — never partitioned.
    """
    if not (0 <= part_id < n_parts):
        raise ValueError(f"part_id={part_id} out of range [0, {n_parts})")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_tf, test_tf = _build_transforms(augment)
    train_set = datasets.CIFAR10(root=str(root), train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=str(root), train=False, download=True, transform=test_tf)

    indices = make_disjoint_indices(len(train_set), n_parts, partition_seed)
    train_subset = Subset(train_set, indices[part_id])

    train_loader = DataLoader(
        train_subset,
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
    # Smoke test: partizioni disgiunte, copertura completa, determinismo.
    idx = make_disjoint_indices(50000, 2, seed=7)
    assert len(idx) == 2
    assert len(idx[0]) + len(idx[1]) == 50000
    assert len(set(idx[0]) & set(idx[1])) == 0, "partitions overlap!"
    assert sorted(idx[0] + idx[1]) == list(range(50000)), "union != full set"

    idx2 = make_disjoint_indices(50000, 2, seed=7)
    assert idx == idx2, "non-deterministic"

    idx3 = make_disjoint_indices(50000, 2, seed=8)
    assert idx != idx3, "different seeds gave same partition"

    print(f"OK: 2 disjoint partitions of 50000 -> {len(idx[0])} + {len(idx[1])} samples")
    print(f"OK: deterministic (same seed -> same partition)")
    print(f"OK: different seeds -> different partitions")
