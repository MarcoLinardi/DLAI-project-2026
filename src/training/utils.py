"""Reusable training utilities: seeding, checkpoints, metrics."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA). Optional cuDNN determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save state_dict + optional metadata (epoch, accuracy, config, ...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    map_location: str | torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint into `model` (in place) and return the metadata dict."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=strict)
    return payload.get("metadata", {})


@dataclass
class AverageMeter:
    """Tracks a running average; pair with .update(value, n) inside the train loop."""

    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy in [0, 1]. Detached, suitable for logging."""
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return (pred == target).float().mean().item()


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
