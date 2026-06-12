"""Run Git Re-Basin activation matching on E3 pairs.

For each pair (ckpt1, ckpt2) listed in the config:
  1. Evaluate the linear interpolation curve theta(alpha) = (1-alpha)*A + alpha*B
     on an 11-point grid (BN-reset at every alpha) -> barrier_pre.
  2. Align ckpt2 to ckpt1 with `find_perms_activation_matching` ->  sd_B_aligned.
  3. Evaluate the curve again with (A, B_aligned) -> barrier_post.
  4. Save B_aligned to results/checkpoints/e3_{pair_name}_m2_aligned.pt and
     write the curve + barrier tables.

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
from src.merging.permute import (
    apply_permutations,
    find_perms_activation_matching,
)
from src.models.resnet20 import resnet20
from src.training.train import load_config
from src.training.utils import device_auto

TRAIN_EVAL_SUBSET_SIZE = 5000
TRAIN_EVAL_SEED = 12345


def _build_align_loader(root: str, n_samples: int, num_workers: int = 2) -> DataLoader:
    """No-aug train subset for activation matching (calibration set)."""
    pin_memory = torch.cuda.is_available()
    _, test_tf = _build_transforms(augment=False)
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=test_tf)
    # Use the first n_samples deterministically (no shuffle needed — calibration only).
    return DataLoader(
        Subset(train_set, list(range(n_samples))),
        batch_size=min(n_samples, 256),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _build_train_eval_loader(root: str, num_workers: int = 2) -> DataLoader:
    """5000-sample deterministic CIFAR-10 train subset, no augmentation.

    Same seed as e1_lmc / e2_merging so train metrics are directly comparable.
    """
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


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def _save_aligned(path: Path, sd_aligned: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd_aligned, "metadata": meta}, path)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 permutation alignment + eval.")
    parser.add_argument("--config", type=str, required=True, help="Path to E3 YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    e3 = cfg.extra.get("e3", {})
    if not e3:
        sys.exit(f"error: config {args.config} has no top-level `e3:` block")

    alphas = list(e3["alpha_grid"])
    bn_reset_batches = int(e3.get("bn_reset_batches", 200))
    align_n_samples = int(e3.get("align_n_samples", 512))
    align_max_iters = int(e3.get("align_max_iters", 5))
    pairs = e3["pairs"]

    device = device_auto()
    print(f"device={device}, bn_reset_batches={bn_reset_batches}, align_n_samples={align_n_samples}, "
          f"align_max_iters={align_max_iters}")
    print(f"pairs: {[p['name'] for p in pairs]}")

    bn_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        root=cfg.data_root,
        augment=True,
    )
    train_eval_loader = _build_train_eval_loader(root=cfg.data_root, num_workers=cfg.num_workers)
    align_loader = _build_align_loader(root=cfg.data_root, n_samples=align_n_samples,
                                       num_workers=cfg.num_workers)
    print(
        f"loaders: bn (full train aug), train_eval ({TRAIN_EVAL_SUBSET_SIZE} no-aug), "
        f"test (full), align ({align_n_samples} no-aug)"
    )

    model_ctor = lambda: resnet20(num_classes=10)

    all_curve_rows: list[dict] = []
    barrier_rows: list[dict] = []

    for p in pairs:
        name = p["name"]
        path1 = Path(p["ckpt1"])
        path2 = Path(p["ckpt2"])
        print(f"\n=== {name} ===")
        print(f"  m1: {path1}")
        print(f"  m2: {path2}")
        sd_A = _load_state_dict(path1)
        sd_B = _load_state_dict(path2)

        # 1. Pre-alignment curve
        print("\n  --- pre-alignment curve ---")
        rows_pre = evaluate_on_alpha_grid(
            model_ctor=model_ctor,
            sd1=sd_A, sd2=sd_B,
            alphas=alphas,
            bn_loader=bn_loader,
            train_eval_loader=train_eval_loader,
            test_loader=test_loader,
            device=device,
            bn_reset_batches=bn_reset_batches,
        )
        b_loss_pre = error_barrier(rows_pre, "test_loss")
        b_acc_pre = error_barrier(rows_pre, "test_acc")
        soup_pre_acc = next(r["test_acc"] for r in rows_pre if abs(r["alpha"] - 0.5) < 1e-9)
        print(
            f"  pre : barrier_test_loss={b_loss_pre:+.4f}, barrier_test_acc={b_acc_pre * 100:+.2f}%, "
            f"midpoint_acc={soup_pre_acc * 100:.2f}%"
        )

        # 2. Alignment
        print("\n  --- activation matching ---")
        model_A = model_ctor().to(device).eval()
        model_B = model_ctor().to(device).eval()
        model_A.load_state_dict(sd_A, strict=True)
        model_B.load_state_dict(sd_B, strict=True)
        perms = find_perms_activation_matching(
            model_A, model_B, align_loader, device,
            n_samples=align_n_samples, max_iters=align_max_iters,
        )
        n_moved = sum(
            int(not torch.equal(perms.get(n), torch.arange(perms.get(n).numel(), dtype=torch.long)))
            for n in perms.perms
        )
        print(f"  permutations: {n_moved}/{len(perms.perms)} differ from identity")
        sd_B_aligned = apply_permutations(sd_B, perms)

        # Persist B_aligned (useful for later TIES-post-align or debugging)
        out_ckpt = Path(cfg.checkpoint_dir) / f"e3_{name}_m2_aligned.pt"
        _save_aligned(
            out_ckpt,
            sd_B_aligned,
            meta={
                "experiment": "E3",
                "pair_name": name,
                "source_ckpt": str(path2),
                "aligned_to": str(path1),
                "align_n_samples": align_n_samples,
                "n_perms_moved": n_moved,
            },
        )
        print(f"  saved aligned checkpoint -> {out_ckpt}")

        # 3. Post-alignment curve
        print("\n  --- post-alignment curve ---")
        rows_post = evaluate_on_alpha_grid(
            model_ctor=model_ctor,
            sd1=sd_A, sd2=sd_B_aligned,
            alphas=alphas,
            bn_loader=bn_loader,
            train_eval_loader=train_eval_loader,
            test_loader=test_loader,
            device=device,
            bn_reset_batches=bn_reset_batches,
        )
        b_loss_post = error_barrier(rows_post, "test_loss")
        b_acc_post = error_barrier(rows_post, "test_acc")
        soup_post_acc = next(r["test_acc"] for r in rows_post if abs(r["alpha"] - 0.5) < 1e-9)
        print(
            f"  post: barrier_test_loss={b_loss_post:+.4f}, barrier_test_acc={b_acc_post * 100:+.2f}%, "
            f"midpoint_acc={soup_post_acc * 100:.2f}%"
        )

        # Accumulate
        for r in rows_pre:
            r2 = dict(r)
            r2["pair"] = name
            r2["alignment"] = "pre"
            all_curve_rows.append(r2)
        for r in rows_post:
            r2 = dict(r)
            r2["pair"] = name
            r2["alignment"] = "post"
            all_curve_rows.append(r2)

        barrier_rows.append({
            "pair": name,
            "barrier_test_loss_pre": b_loss_pre,
            "barrier_test_loss_post": b_loss_post,
            "barrier_test_acc_pre": b_acc_pre,
            "barrier_test_acc_post": b_acc_post,
            "midpoint_acc_pre": soup_pre_acc,
            "midpoint_acc_post": soup_post_acc,
            "n_perms_moved": n_moved,
        })

    # --- Persist tables ---
    tables_dir = Path(cfg.history_dir)
    _write_csv(
        tables_dir / "e3_curves.csv",
        all_curve_rows,
        fieldnames=["pair", "alignment", "alpha", "train_loss", "train_acc",
                    "test_loss", "test_acc"],
    )
    _write_csv(
        tables_dir / "e3_barriers.csv",
        barrier_rows,
        fieldnames=["pair", "barrier_test_loss_pre", "barrier_test_loss_post",
                    "barrier_test_acc_pre", "barrier_test_acc_post",
                    "midpoint_acc_pre", "midpoint_acc_post", "n_perms_moved"],
    )
    print(f"\nsaved -> {tables_dir / 'e3_curves.csv'}")
    print(f"saved -> {tables_dir / 'e3_barriers.csv'}")

    # --- Summary to stdout ---
    print("\n=== E3 alignment summary ===")
    print(f"{'pair':<20s} {'barrier_pre':>12s} {'barrier_post':>12s} "
          f"{'midpoint_pre':>13s} {'midpoint_post':>14s}")
    for r in barrier_rows:
        print(
            f"{r['pair']:<20s} {r['barrier_test_loss_pre']:>+12.4f} "
            f"{r['barrier_test_loss_post']:>+12.4f} "
            f"{r['midpoint_acc_pre'] * 100:>12.2f}% "
            f"{r['midpoint_acc_post'] * 100:>13.2f}%"
        )


if __name__ == "__main__":
    main()
