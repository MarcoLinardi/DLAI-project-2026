"""REPAIR — mini fine-tune of the merged (post-alignment) model.

After Activation Matching reduces but does not close the barrier (E3),
the merged model (theta_A + theta_B_aligned)/2 still sits in a sub-optimal
region of the loss landscape: BN running stats are mismatched, and the
classifier head sees rotated/scaled features. REPAIR (Jordan et al. 2023)
applies a brief SGD fine-tune at a low learning rate to recover the gap.

For each pair listed in the E3 config:
  1. Load sd_A and sd_B_aligned (produced by `src.experiments.e3_align`).
  2. Build the uniform merge sd_merged = 0.5 * (sd_A + sd_B_aligned).
  3. BN-reset (200 batches of augmented train data).
  4. Mini fine-tune on full CIFAR-10 train for `repair_epochs` epochs at
     low LR (default 0.001, momentum 0.9, weight decay 1e-4).
  5. Evaluate on test set and persist results.

CLI:
    python -m src.experiments.e3_repair --config configs/e3_align.yaml \\
        [--repair-epochs 2] [--repair-lr 0.001]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD

from src.data.cifar import get_cifar10_loaders
from src.merging.lmc import reset_bn_stats
from src.merging.soup import average_state_dicts
from src.models.resnet20 import resnet20
from src.training.train import evaluate, load_config, train_one_epoch
from src.training.utils import device_auto


def _load_sd(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def main() -> None:
    parser = argparse.ArgumentParser(description="REPAIR — mini fine-tune of merged aligned models.")
    parser.add_argument("--config", type=str, required=True, help="Path to E3 YAML config.")
    parser.add_argument("--repair-epochs", type=int, default=2,
                        help="Mini fine-tune epochs (default 2).")
    parser.add_argument("--repair-lr", type=float, default=0.001,
                        help="Mini fine-tune learning rate (default 0.001).")
    parser.add_argument("--history-csv", type=str, default=None,
                        help="If set, save per-epoch history for all pairs to this CSV path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    e3 = cfg.extra.get("e3", {})
    if not e3:
        sys.exit(f"error: config {args.config} has no top-level `e3:` block")
    bn_reset_batches = int(e3.get("bn_reset_batches", 200))
    pairs = e3["pairs"]

    device = device_auto()
    print(
        f"device={device}, repair_epochs={args.repair_epochs}, repair_lr={args.repair_lr}, "
        f"bn_reset_batches={bn_reset_batches}"
    )

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        root=cfg.data_root,
        augment=True,
    )
    criterion = nn.CrossEntropyLoss()

    rows: list[dict] = []
    epoch_history: list[dict] = []
    ckpt_dir = Path(cfg.checkpoint_dir)

    for p in pairs:
        name = p["name"]
        path_A = Path(p["ckpt1"])
        path_B_aligned = ckpt_dir / f"e3_{name}_m2_aligned.pt"
        if not path_B_aligned.exists():
            print(f"warn: missing {path_B_aligned}; run e3_align first. Skipping {name}.")
            continue
        print(f"\n=== REPAIR pair {name} ===")
        print(f"  A: {path_A}")
        print(f"  B_aligned: {path_B_aligned}")

        sd_A = _load_sd(path_A)
        sd_B_aligned = _load_sd(path_B_aligned)
        sd_merged = average_state_dicts([sd_A, sd_B_aligned])

        model = resnet20(num_classes=10).to(device)
        model.load_state_dict(sd_merged, strict=True)

        # BN reset (matches the protocol used in eval everywhere else).
        print(f"  BN reset on {bn_reset_batches} batches...")
        reset_bn_stats(model, train_loader, device, num_batches=bn_reset_batches)

        # Pre-repair test accuracy (should match the midpoint_post from e3_align).
        loss_pre, acc_pre = evaluate(model, test_loader, criterion, device)
        print(f"  pre-repair test: loss={loss_pre:.4f}, acc={acc_pre * 100:.2f}%")

        # Mini fine-tune. Plain SGD+momentum, no cosine (only 1-3 epochs).
        optimizer = SGD(model.parameters(), lr=args.repair_lr,
                        momentum=0.9, weight_decay=1e-4, nesterov=True)
        history: list[dict] = []
        for ep in range(1, args.repair_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            te_loss, te_acc = evaluate(model, test_loader, criterion, device)
            print(
                f"  repair epoch {ep}/{args.repair_epochs}: "
                f"train_loss={tr_loss:.4f} acc={tr_acc * 100:.2f}% | "
                f"test_loss={te_loss:.4f} acc={te_acc * 100:.2f}%"
            )
            history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                            "test_loss": te_loss, "test_acc": te_acc})
            epoch_history.append({"pair": name, "epoch": ep, "train_loss": tr_loss,
                                   "train_acc": tr_acc, "test_loss": te_loss, "test_acc": te_acc})

        # Persist repaired merged model.
        repaired_path = ckpt_dir / f"e3_{name}_merged_repaired.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": {
                    "experiment": "E3-REPAIR",
                    "pair_name": name,
                    "source_A": str(path_A),
                    "source_B_aligned": str(path_B_aligned),
                    "repair_epochs": args.repair_epochs,
                    "repair_lr": args.repair_lr,
                    "test_acc_pre_repair": acc_pre,
                    "test_acc_post_repair": te_acc,
                },
            },
            repaired_path,
        )
        print(f"  saved -> {repaired_path}")

        rows.append({
            "pair": name,
            "test_acc_pre_repair": acc_pre,
            "test_loss_pre_repair": loss_pre,
            "test_acc_post_repair": te_acc,
            "test_loss_post_repair": te_loss,
            "delta_pp": (te_acc - acc_pre) * 100.0,
            "repair_epochs": args.repair_epochs,
            "repair_lr": args.repair_lr,
        })

    # Persist per-epoch history if requested (E6 epoch ablation)
    if args.history_csv:
        hist_path = Path(args.history_csv)
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_fields = ["pair", "epoch", "train_loss", "train_acc", "test_loss", "test_acc"]
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hist_fields)
            w.writeheader()
            for r in epoch_history:
                w.writerow(r)
        print(f"\nsaved epoch history -> {hist_path}")

    # Persist results table
    out_csv = Path(cfg.history_dir) / "e3_repair.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair", "test_acc_pre_repair", "test_loss_pre_repair",
              "test_acc_post_repair", "test_loss_post_repair", "delta_pp",
              "repair_epochs", "repair_lr"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved -> {out_csv}")

    # Summary
    print("\n=== E3 REPAIR summary ===")
    print(f"{'pair':<20s} {'pre_acc':>10s} {'post_acc':>10s} {'delta_pp':>10s}")
    for r in rows:
        print(
            f"{r['pair']:<20s} {r['test_acc_pre_repair'] * 100:>9.2f}% "
            f"{r['test_acc_post_repair'] * 100:>9.2f}% "
            f"{r['delta_pp']:>+9.2f}"
        )


if __name__ == "__main__":
    main()
