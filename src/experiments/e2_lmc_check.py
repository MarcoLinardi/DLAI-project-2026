"""Sanity check: linear-mode connectivity between pairs of E2 models.

Motivation: after the uniform soup of 5 E2 models collapsed to 12.5% test
accuracy (despite shared init), we suspect the models drifted out of the
linear-mode-connected basin under the low-resource regime (10k examples
per model, 30 epochs).

This script measures the barrier of theta(alpha) = (1-alpha)*theta_1 +
alpha*theta_2 for three diagnostic pairs:
  - e2_model0 vs e2_model2   (two E2 members, same init)
  - e2_model1 vs e2_model4   (another E2 pair, same init)
  - e2_model2 vs baseline_seed0  (best E2 model vs full-data baseline)

The first two test "is the E2 setup linear-mode-connected?". The third
tests "are the TIES task vectors tau_i = theta_E2 - theta_baseline LMC
to zero?", which is the implicit assumption of task arithmetic.

Reuses evaluate_on_alpha_grid and error_barrier from src.merging.lmc and
the train_eval loader logic from src.experiments.e1_lmc and e2_merging.

CLI:
    python -m src.experiments.e2_lmc_check --config configs/e2_soup.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from torchvision import datasets  # noqa: E402

from src.data.cifar import _build_transforms, get_cifar10_loaders  # noqa: E402
from src.merging.lmc import error_barrier, evaluate_on_alpha_grid  # noqa: E402
from src.models.resnet20 import resnet20  # noqa: E402
from src.training.train import load_config  # noqa: E402
from src.training.utils import device_auto  # noqa: E402

TRAIN_EVAL_SUBSET_SIZE = 5000
TRAIN_EVAL_SEED = 12345

ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

PAIRS: list[tuple[str, str, str]] = [
    ("e2_model0_x_e2_model2", "e2_model0.pt", "e2_model2.pt"),
    ("e2_model1_x_e2_model4", "e2_model1.pt", "e2_model4.pt"),
    ("e2_model2_x_baseline", "e2_model2.pt", "baseline_seed0.pt"),
]


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
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["state_dict"]


def main() -> None:
    parser = argparse.ArgumentParser(description="LMC sanity check on E2 model pairs.")
    parser.add_argument("--config", type=str, required=True, help="Path to E2 YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    e2 = cfg.extra.get("e2", {})
    if not e2:
        sys.exit(f"error: config {args.config} has no top-level `e2:` block")
    bn_reset_batches = int(e2.get("bn_reset_batches", 200))

    device = device_auto()
    print(f"device={device}, bn_reset_batches={bn_reset_batches}")

    bn_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        root=cfg.data_root,
        augment=True,
    )
    train_eval_loader = _build_train_eval_loader(root=cfg.data_root, num_workers=cfg.num_workers)
    print(f"loaders: bn (full train aug), train_eval ({TRAIN_EVAL_SUBSET_SIZE} no-aug), test (full)")

    ckpt_dir = Path(cfg.checkpoint_dir)
    all_rows: list[dict] = []
    barriers_rows: list[dict] = []

    for pair_name, ckpt1, ckpt2 in PAIRS:
        print(f"\n=== {pair_name} ===")
        p1 = ckpt_dir / ckpt1
        p2 = ckpt_dir / ckpt2
        print(f"  m1: {p1}")
        print(f"  m2: {p2}")
        sd1 = _load_state_dict(p1)
        sd2 = _load_state_dict(p2)

        rows = evaluate_on_alpha_grid(
            model_ctor=lambda: resnet20(num_classes=10),
            sd1=sd1,
            sd2=sd2,
            alphas=ALPHA_GRID,
            bn_loader=bn_loader,
            train_eval_loader=train_eval_loader,
            test_loader=test_loader,
            device=device,
            bn_reset_batches=bn_reset_batches,
        )
        b_loss = error_barrier(rows, "test_loss")
        b_acc = error_barrier(rows, "test_acc")
        print(f"  barrier(test_loss)={b_loss:+.4f}, barrier(test_acc)={b_acc * 100:+.2f}%")

        for r in rows:
            r["pair"] = pair_name
            all_rows.append(r)
        barriers_rows.append({
            "pair": pair_name,
            "ckpt1": ckpt1,
            "ckpt2": ckpt2,
            "barrier_test_loss": b_loss,
            "barrier_test_acc": b_acc,
        })

    # --- CSV outputs ---
    tables_dir = Path(cfg.history_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    curve_path = tables_dir / "e2_lmc_check.csv"
    fields = ["pair", "alpha", "train_loss", "train_acc", "test_loss", "test_acc"]
    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in fields})
    print(f"\nsaved -> {curve_path}")

    bars_path = tables_dir / "e2_lmc_barriers.csv"
    with open(bars_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pair", "ckpt1", "ckpt2", "barrier_test_loss", "barrier_test_acc"]
        )
        writer.writeheader()
        for r in barriers_rows:
            writer.writerow(r)
    print(f"saved -> {bars_path}")

    # --- Plot ---
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "e2_lmc_barrier.png"

    colors = {"e2_model0_x_e2_model2": "#1565C0",
              "e2_model1_x_e2_model4": "#6A1B9A",
              "e2_model2_x_baseline": "#C62828"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=140)
    for pair_name in (p[0] for p in PAIRS):
        sub = [r for r in all_rows if r["pair"] == pair_name]
        sub.sort(key=lambda r: r["alpha"])
        xs = [r["alpha"] for r in sub]
        ys_loss = [r["test_loss"] for r in sub]
        ys_acc = [r["test_acc"] * 100 for r in sub]
        color = colors.get(pair_name, "#000000")
        ax1.plot(xs, ys_loss, marker="o", markersize=4, color=color, label=pair_name)
        ax2.plot(xs, ys_acc, marker="o", markersize=4, color=color, label=pair_name)

    for ax, ylabel, title in [
        (ax1, "test loss", "Test loss vs alpha"),
        (ax2, "test accuracy (%)", "Test accuracy vs alpha"),
    ]:
        ax.set_xlabel(r"interpolation coefficient $\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True, fontsize=8)

    fig.suptitle("E2 LMC sanity check (BN-reset, 200 batches)")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"saved -> {plot_path}")

    # Summary
    print("\n=== E2 LMC barriers ===")
    for r in barriers_rows:
        print(
            f"  {r['pair']:30s}: barrier_test_loss={r['barrier_test_loss']:+.4f}, "
            f"barrier_test_acc={r['barrier_test_acc'] * 100:+.2f}%"
        )


if __name__ == "__main__":
    main()
