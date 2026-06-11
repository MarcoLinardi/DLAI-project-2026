"""Sweep TIES-Merging on the 5 E2 models over a grid of lambda values.

The main E2 merging script uses TIES with lam=1.0 (paper default). To
defend the "TIES catastrophic failure" claim quantitatively, we sweep
lam over a small grid and report test accuracy for each lam, keeping
the trim ratio fixed at the same value used in `e2_merging.py`.

Outputs:
  - results/tables/e2_ties_sweep.csv  (one row per lam value)

CLI:
    python -m src.experiments.e2_ties_sweep --config configs/e2_soup.yaml \
        [--lams 0.3 0.5 1.0 1.5]
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
from src.merging.lmc import _eval_at_point
from src.merging.ties import ties_merge
from src.models.resnet20 import resnet20
from src.training.train import load_config
from src.training.utils import device_auto

TRAIN_EVAL_SUBSET_SIZE = 5000
TRAIN_EVAL_SEED = 12345


def _build_train_eval_loader(root: str, num_workers: int = 2) -> DataLoader:
    pin_memory = torch.cuda.is_available()
    _, test_tf = _build_transforms(augment=False)
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=test_tf)
    rng = np.random.default_rng(TRAIN_EVAL_SEED)
    idx = rng.choice(len(train_set), size=TRAIN_EVAL_SUBSET_SIZE, replace=False).tolist()
    return DataLoader(
        Subset(train_set, idx),
        batch_size=512,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _load_payload(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep TIES lam on E2 models.")
    parser.add_argument("--config", type=str, required=True, help="Path to E2 YAML config.")
    parser.add_argument(
        "--lams",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 1.0, 1.5],
        help="Lambda values to sweep (default: 0.3 0.5 1.0 1.5).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    e2 = cfg.extra.get("e2", {})
    if not e2:
        sys.exit(f"error: config {args.config} has no top-level `e2:` block")

    n_models = int(e2["n_models"])
    bn_reset_batches = int(e2.get("bn_reset_batches", 200))
    ties_cfg = e2.get("ties", {})
    keep_ratio = float(ties_cfg.get("keep_ratio", 0.20))
    base_path = Path(e2.get("baseline_checkpoint", "results/checkpoints/baseline_seed0.pt"))

    device = device_auto()
    print(f"device={device}, n_models={n_models}, bn_reset_batches={bn_reset_batches}")
    print(f"TIES sweep: keep_ratio={keep_ratio}, lams={args.lams}")
    print(f"baseline: {base_path}")

    base_payload = _load_payload(base_path)
    base_sd = base_payload["state_dict"]

    sds: list[dict[str, torch.Tensor]] = []
    accs_meta: list[float] = []
    ckpt_dir = Path(cfg.checkpoint_dir)
    for i in range(n_models):
        path = ckpt_dir / f"e2_model{i}.pt"
        payload = _load_payload(path)
        sds.append(payload["state_dict"])
        accs_meta.append(float(payload.get("metadata", {}).get("final_test_acc", float("nan"))))
        print(f"  loaded e2_model{i}.pt — meta test_acc={accs_meta[-1] * 100:.2f}%")

    bn_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        root=cfg.data_root,
        augment=True,
    )
    train_eval_loader = _build_train_eval_loader(cfg.data_root, cfg.num_workers)
    model_ctor = lambda: resnet20(num_classes=10)

    rows: list[dict] = []
    print()
    for lam in args.lams:
        print(f"=== TIES sweep: lam={lam} ===")
        ties_sd = ties_merge(sds, base_sd, keep_ratio=keep_ratio, lam=lam)
        m = _eval_at_point(
            model_ctor=model_ctor,
            sd=ties_sd,
            bn_loader=bn_loader,
            train_eval_loader=train_eval_loader,
            test_loader=test_loader,
            device=device,
            bn_reset_batches=bn_reset_batches,
        )
        print(
            f"  test_acc={m['test_acc'] * 100:.2f}%  "
            f"test_loss={m['test_loss']:.4f}  "
            f"train_acc={m['train_acc'] * 100:.2f}%  "
            f"train_loss={m['train_loss']:.4f}"
        )
        rows.append({
            "method": "ties",
            "keep_ratio": keep_ratio,
            "lam": lam,
            "test_loss": m["test_loss"],
            "test_acc": m["test_acc"],
            "train_loss": m["train_loss"],
            "train_acc": m["train_acc"],
        })

    out_csv = Path(cfg.history_dir) / "e2_ties_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "keep_ratio", "lam",
              "test_loss", "test_acc", "train_loss", "train_acc"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved -> {out_csv}")

    print("\n=== TIES sweep summary ===")
    print(f"{'lam':>6s} {'test_acc':>10s} {'test_loss':>10s}")
    for r in rows:
        print(f"{r['lam']:>6.2f} {r['test_acc'] * 100:>9.2f}% {r['test_loss']:>10.4f}")


if __name__ == "__main__":
    main()
