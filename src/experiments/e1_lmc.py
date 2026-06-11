"""Evaluate Linear Mode Connectivity for one E1 setting.

For each pair_seed in the config, loads the two trained checkpoints,
sweeps the interpolation alpha grid, re-estimates BatchNorm stats at
every point, evaluates on a 5000-sample train subset and the full test
set, computes the error barrier, and writes results/tables/e1_{setting}.csv.

CLI:
    python -m src.experiments.e1_lmc --config configs/e1_B_sameinit_splitdata.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.data.cifar import _build_transforms, get_cifar10_loaders
from src.merging.lmc import error_barrier, evaluate_on_alpha_grid
from src.models.resnet20 import resnet20
from src.training.train import load_config
from src.training.utils import device_auto

TRAIN_EVAL_SUBSET_SIZE = 5000
TRAIN_EVAL_SEED = 12345  # fixed across runs/settings → comparable train metrics


def _build_train_eval_loader(
    root: str,
    batch_size: int = 512,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> DataLoader:
    """5000-sample deterministic subset of CIFAR-10 train, no augmentation."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    _, test_tf = _build_transforms(augment=False)  # test_tf = no augmentation
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=test_tf)
    rng = np.random.default_rng(TRAIN_EVAL_SEED)
    idx = rng.choice(len(train_set), size=TRAIN_EVAL_SUBSET_SIZE, replace=False).tolist()
    return DataLoader(
        Subset(train_set, idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load just the state_dict (no side-effect on any model)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["state_dict"]


def _checkpoint_paths(cfg_ckpt_dir: str, setting: str, pair_seed: int, e1: dict) -> tuple[Path, Path]:
    """Return (path_m1, path_m2). Setting A reuses the baseline for both."""
    if e1.get("reuse_baseline", False):
        baseline = Path(e1.get("baseline_checkpoint", "results/checkpoints/baseline_seed0.pt"))
        return baseline, baseline
    base = Path(cfg_ckpt_dir)
    p1 = base / f"e1{setting}_pair{pair_seed}_m1.pt"
    p2 = base / f"e1{setting}_pair{pair_seed}_m2.pt"
    return p1, p2


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E1 LMC for one setting.")
    parser.add_argument("--config", type=str, required=True, help="Path to E1 YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    e1 = cfg.extra.get("e1", {})
    if not e1:
        sys.exit(f"error: config {args.config} has no top-level `e1:` block")

    setting = e1["setting"]
    alphas = list(e1["alpha_grid"])
    pair_seeds = list(e1["pair_seeds"])
    bn_reset_batches = int(e1.get("bn_reset_batches", 40))

    device = device_auto()
    print(f"device={device}, setting={setting}, pair_seeds={pair_seeds}, alphas={alphas}")

    bn_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        root=cfg.data_root,
        augment=True,  # bn_loader needs augmentation (Frankle 2020)
    )
    train_eval_loader = _build_train_eval_loader(
        root=cfg.data_root,
        num_workers=cfg.num_workers,
    )
    print(f"loaders: bn (full train aug), train_eval ({TRAIN_EVAL_SUBSET_SIZE} no-aug), test (full)")

    out_rows: list[dict] = []
    for s in pair_seeds:
        print(f"\n=== pair_seed={s} ===")
        p1, p2 = _checkpoint_paths(cfg.checkpoint_dir, setting, s, e1)
        print(f"m1: {p1}")
        print(f"m2: {p2}")
        sd1 = _load_state_dict(p1)
        sd2 = _load_state_dict(p2)

        rows = evaluate_on_alpha_grid(
            model_ctor=lambda: resnet20(num_classes=10),
            sd1=sd1,
            sd2=sd2,
            alphas=alphas,
            bn_loader=bn_loader,
            train_eval_loader=train_eval_loader,
            test_loader=test_loader,
            device=device,
            bn_reset_batches=bn_reset_batches,
        )
        b_loss = error_barrier(rows, metric="test_loss")
        b_acc = error_barrier(rows, metric="test_acc")
        print(f"  barrier(test_loss)={b_loss:+.4f}, barrier(test_acc)={b_acc*100:+.2f}%")

        for r in rows:
            r["setting"] = setting
            r["pair_seed"] = s
            out_rows.append(r)

    # Write CSV
    out_path = Path(cfg.history_dir) / f"e1_{setting}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["setting", "pair_seed", "alpha", "train_loss", "train_acc", "test_loss", "test_acc"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nsaved {len(out_rows)} rows -> {out_path}")

    # Summary across pair_seeds (if more than one)
    if len(pair_seeds) > 1:
        per_pair_loss = []
        per_pair_acc = []
        for s in pair_seeds:
            sub = [r for r in out_rows if r["pair_seed"] == s]
            per_pair_loss.append(error_barrier(sub, "test_loss"))
            per_pair_acc.append(error_barrier(sub, "test_acc"))
        loss_arr = np.array(per_pair_loss)
        acc_arr = np.array(per_pair_acc)
        print(
            f"barrier mean ± std over {len(pair_seeds)} pairs: "
            f"test_loss={loss_arr.mean():+.4f} ± {loss_arr.std():.4f}, "
            f"test_acc={acc_arr.mean()*100:+.2f}% ± {acc_arr.std()*100:.2f}%"
        )


if __name__ == "__main__":
    main()
