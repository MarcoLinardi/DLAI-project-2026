"""Reusable train loop + CLI for ResNet-20 on CIFAR-10.

Used as a library by E1-E4 (merging experiments) and as a script for
the Phase-1 baseline:

"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.cifar import get_cifar10_loaders
from src.models.resnet20 import resnet20
from src.training.utils import (
    AverageMeter,
    accuracy,
    count_parameters,
    device_auto,
    save_checkpoint,
    set_seed,
)


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True
    scheduler: str = "cosine"  # currently only "cosine" supported
    num_workers: int = 2
    data_root: str = "./data"
    checkpoint_dir: str = "results/checkpoints"
    history_dir: str = "results/tables"
    augment: bool = True
    log_every: int = 100  # batches between progress prints
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    known = {f.name for f in TrainConfig.__dataclass_fields__.values()}
    extra = {k: v for k, v in raw.items() if k not in known}
    kwargs = {k: v for k, v in raw.items() if k in known}
    return TrainConfig(extra=extra, **kwargs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """One pass over `loader`. Returns (avg_loss, avg_accuracy)."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(accuracy(logits, y), n=x.size(0))
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Test-set evaluation. Returns (avg_loss, avg_accuracy)."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_meter.update(loss.item(), n=x.size(0))
        acc_meter.update(accuracy(logits, y), n=x.size(0))
    return loss_meter.avg, acc_meter.avg


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device | None = None,
) -> list[dict[str, float]]:
    """Full training run. Returns per-epoch history (list of dicts)."""
    device = device or device_auto()
    model.to(device)

    optimizer = SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
    )
    if cfg.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}")

    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "test_loss": te_loss,
            "test_acc": te_acc,
            "lr": lr_now,
            "seconds": dt,
        }
        history.append(row)
        print(
            f"epoch {epoch:3d}/{cfg.epochs} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc*100:5.2f}% | "
            f"test_loss={te_loss:.4f} acc={te_acc*100:5.2f}% | "
            f"lr={lr_now:.4f} | {dt:.1f}s"
        )
    return history


def write_history_csv(path: str | Path, history: list[dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fields = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-20 on CIFAR-10.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="baseline", help="Run name prefix for outputs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = device_auto()
    print(f"device={device}, seed={args.seed}, tag={args.tag}")

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        root=cfg.data_root,
        augment=cfg.augment,
    )

    model = resnet20(num_classes=10)
    n_params = count_parameters(model)
    print(f"model: ResNet-20, params={n_params:,}")

    history = train_model(model, train_loader, test_loader, cfg, device=device)

    run_name = f"{args.tag}_seed{args.seed}"
    ckpt_path = Path(cfg.checkpoint_dir) / f"{run_name}.pt"
    csv_path = Path(cfg.history_dir) / f"{run_name}.csv"
    final = history[-1]
    save_checkpoint(
        ckpt_path,
        model,
        metadata={
            "seed": args.seed,
            "tag": args.tag,
            "config": args.config,
            "final_test_acc": final["test_acc"],
            "final_train_acc": final["train_acc"],
            "epochs": cfg.epochs,
        },
    )
    write_history_csv(csv_path, history)
    print(f"\nsaved checkpoint -> {ckpt_path}")
    print(f"saved history    -> {csv_path}")
    print(f"final test_acc   = {final['test_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
