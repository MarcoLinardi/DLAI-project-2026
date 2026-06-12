"""Apply the 4 merging methods of experiment E2 and write result tables.

Loads:
  - results/checkpoints/baseline_seed0.pt  (theta_base for TIES task vectors)
  - results/checkpoints/e2_model{0..N-1}.pt  (trained by train_e2.py)

Computes for each method:
  1. Best single model: argmax_i test_acc(theta_i)
  2. Uniform soup    : theta = (1/N) * sum_i theta_i
  3. Greedy soup     : Wortsman 2022, Algorithm 1
  4. TIES merge      : Yadav 2023 (trim + elect + disjoint merge), lam, keep_ratio
                       read from config.

For every method we re-estimate BatchNorm running stats (40 batches of
augmented train data) before evaluation, exactly as in E1. Skipping the
BN reset on merged models collapses accuracy to ~10%, the rest of the
pipeline assumes this step is always done.

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
from src.merging.soup import greedy_soup, uniform_soup
from src.merging.ties import ties_merge
from src.models.resnet20 import resnet20
from src.training.train import load_config
from src.training.utils import device_auto

TRAIN_EVAL_SUBSET_SIZE = 5000
TRAIN_EVAL_SEED = 12345  # same as E1 → train-set metrics comparable across experiments


def _build_train_eval_loader(
    root: str,
    batch_size: int = 512,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> DataLoader:
    """5000-sample deterministic subset of CIFAR-10 train, no augmentation.

    Duplicated from src.experiments.e1_lmc to keep the two scripts standalone.
    Same TRAIN_EVAL_SEED → identical indices → train metrics directly
    comparable between E1 and E2 tables.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    _, test_tf = _build_transforms(augment=False)
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


def _load_payload(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply 4 merging methods to E2 trained models.")
    parser.add_argument("--config", type=str, required=True, help="Path to E2 YAML config.")
    parser.add_argument("--val-split", action="store_true",
                        help="Use a 5k val split for greedy soup selection (avoids test-set leak).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    e2 = cfg.extra.get("e2", {})
    if not e2:
        sys.exit(f"error: config {args.config} has no top-level `e2:` block")

    n_models = int(e2["n_models"])
    bn_reset_batches = int(e2.get("bn_reset_batches", 40))
    ties_cfg = e2.get("ties", {})
    keep_ratio = float(ties_cfg.get("keep_ratio", 0.20))
    lam = float(ties_cfg.get("lam", 1.0))
    base_path = Path(e2.get("baseline_checkpoint", "results/checkpoints/baseline_seed0.pt"))

    device = device_auto()
    print(f"device={device}, n_models={n_models}, bn_reset_batches={bn_reset_batches}")
    print(f"TIES: keep_ratio={keep_ratio}, lam={lam}")
    print(f"baseline: {base_path}")

    # --- Load checkpoints ---
    base_payload = _load_payload(base_path)
    base_sd = base_payload["state_dict"]

    sds: list[dict[str, torch.Tensor]] = []
    accs: list[float] = []
    ckpt_dir = Path(cfg.checkpoint_dir)
    for i in range(n_models):
        path = ckpt_dir / f"e2_model{i}.pt"
        payload = _load_payload(path)
        sd = payload["state_dict"]
        meta = payload.get("metadata", {})
        acc = float(meta.get("final_test_acc", float("nan")))
        sds.append(sd)
        accs.append(acc)
        print(f"  loaded e2_model{i}.pt — final_test_acc={acc * 100:.2f}%")

    # --- Loaders ---
    bn_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        root=cfg.data_root,
        augment=True,  # BN reset wants augmented train data (matches train-time stats)
    )
    train_eval_loader = _build_train_eval_loader(
        root=cfg.data_root,
        num_workers=cfg.num_workers,
    )
    print(
        f"loaders: bn (full train aug), train_eval ({TRAIN_EVAL_SUBSET_SIZE} no-aug), test (full)"
    )

    # Optional validation split for greedy soup selection (E8: avoids test-set data leak)
    val_loader = None
    VAL_SIZE = 5000
    VAL_SEED = 99999
    if args.val_split:
        _, test_tf = _build_transforms(augment=False)
        from torchvision import datasets as tv_datasets
        train_set = tv_datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=test_tf)
        rng = np.random.default_rng(VAL_SEED)
        val_idx = rng.choice(len(train_set), size=VAL_SIZE, replace=False).tolist()
        from torch.utils.data import DataLoader as _DL, Subset as _Sub
        val_loader = _DL(
            _Sub(train_set, val_idx),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        print(f"val split: {VAL_SIZE} samples (seed={VAL_SEED}) — used for greedy selection")

    model_ctor = lambda: resnet20(num_classes=10)

    # --- 1. Best single model ---
    print("\n=== Best single model ===")
    best_idx = int(np.argmax(accs))
    print(f"argmax over individual accs -> model {best_idx} (acc={accs[best_idx] * 100:.2f}%)")
    best_metrics = _eval_at_point(
        model_ctor=model_ctor,
        sd=sds[best_idx],
        bn_loader=bn_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        device=device,
        bn_reset_batches=bn_reset_batches,
    )
    print(f"  test_acc={best_metrics['test_acc'] * 100:.2f}% test_loss={best_metrics['test_loss']:.4f}")

    # --- 2. Uniform soup ---
    print("\n=== Uniform soup ===")
    _, uniform_metrics = uniform_soup(
        sds=sds,
        model_ctor=model_ctor,
        bn_loader=bn_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        device=device,
        bn_reset_batches=bn_reset_batches,
    )
    print(
        f"  test_acc={uniform_metrics['test_acc'] * 100:.2f}% "
        f"test_loss={uniform_metrics['test_loss']:.4f}"
    )

    # --- 3. Greedy soup ---
    print("\n=== Greedy soup ===")
    _, included_idx, greedy_history = greedy_soup(
        sds_with_acc=list(zip(sds, accs)),
        model_ctor=model_ctor,
        bn_loader=bn_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        device=device,
        bn_reset_batches=bn_reset_batches,
        selection_loader=val_loader,  # None → uses test_loader (canonical); set for E8
    )
    final_greedy_acc = greedy_history[-1]["soup_acc_after"]
    print(f"  included models (orig idx): {included_idx}")
    print(f"  final greedy soup test_acc={final_greedy_acc * 100:.2f}%")
    # Re-evaluate final greedy soup to get full metrics (train + test loss)
    from src.merging.soup import average_state_dicts
    greedy_sd_final = average_state_dicts([sds[i] for i in included_idx])
    greedy_metrics = _eval_at_point(
        model_ctor=model_ctor,
        sd=greedy_sd_final,
        bn_loader=bn_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        device=device,
        bn_reset_batches=bn_reset_batches,
    )
    print(
        f"  test_acc={greedy_metrics['test_acc'] * 100:.2f}% "
        f"test_loss={greedy_metrics['test_loss']:.4f}"
    )

    # --- 4. TIES merge ---
    print(f"\n=== TIES merge (keep_ratio={keep_ratio}, lam={lam}) ===")
    ties_sd = ties_merge(sds, base_sd, keep_ratio=keep_ratio, lam=lam)
    ties_metrics = _eval_at_point(
        model_ctor=model_ctor,
        sd=ties_sd,
        bn_loader=bn_loader,
        train_eval_loader=train_eval_loader,
        test_loader=test_loader,
        device=device,
        bn_reset_batches=bn_reset_batches,
    )
    print(
        f"  test_acc={ties_metrics['test_acc'] * 100:.2f}% "
        f"test_loss={ties_metrics['test_loss']:.4f}"
    )

    # --- Write CSVs ---
    tables_dir = Path(cfg.history_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Re-evaluate each individual under the same BN-reset protocol used for
    # the merged methods, so all rows of the comparison table are apples-to-apples.
    print("\n=== Re-evaluating individuals with BN reset (apples-to-apples) ===")
    individuals_rows: list[dict] = []
    for i in range(n_models):
        m = _eval_at_point(
            model_ctor=model_ctor,
            sd=sds[i],
            bn_loader=bn_loader,
            train_eval_loader=train_eval_loader,
            test_loader=test_loader,
            device=device,
            bn_reset_batches=bn_reset_batches,
        )
        print(
            f"  model {i}: test_acc={m['test_acc'] * 100:.2f}% "
            f"(metadata={accs[i] * 100:.2f}%), test_loss={m['test_loss']:.4f}"
        )
        individuals_rows.append({
            "model_idx": i,
            "test_loss": m["test_loss"],
            "test_acc": m["test_acc"],
            "test_acc_metadata": accs[i],
        })
    _write_csv(
        tables_dir / "e2_individuals.csv",
        individuals_rows,
        fieldnames=["model_idx", "test_loss", "test_acc", "test_acc_metadata"],
    )

    merging_rows = [
        {
            "method": "best_single",
            "test_loss": best_metrics["test_loss"],
            "test_acc": best_metrics["test_acc"],
            "n_models_used": 1,
            "hyperparams": f"best_idx={best_idx}",
        },
        {
            "method": "uniform_soup",
            "test_loss": uniform_metrics["test_loss"],
            "test_acc": uniform_metrics["test_acc"],
            "n_models_used": n_models,
            "hyperparams": "",
        },
        {
            "method": "greedy_soup",
            "test_loss": greedy_metrics["test_loss"],
            "test_acc": greedy_metrics["test_acc"],
            "n_models_used": len(included_idx),
            "hyperparams": f"included={','.join(map(str, included_idx))}",
        },
        {
            "method": "ties",
            "test_loss": ties_metrics["test_loss"],
            "test_acc": ties_metrics["test_acc"],
            "n_models_used": n_models,
            "hyperparams": f"keep_ratio={keep_ratio},lam={lam}",
        },
    ]
    _write_csv(
        tables_dir / "e2_merging.csv",
        merging_rows,
        fieldnames=["method", "test_loss", "test_acc", "n_models_used", "hyperparams"],
    )

    _write_csv(
        tables_dir / "e2_greedy_steps.csv",
        greedy_history,
        fieldnames=["step", "candidate_orig_idx", "candidate_acc_alone",
                    "soup_acc_after", "included", "n_in_soup"],
    )

    print(f"\nsaved -> {tables_dir / 'e2_individuals.csv'}")
    print(f"saved -> {tables_dir / 'e2_merging.csv'}")
    print(f"saved -> {tables_dir / 'e2_greedy_steps.csv'}")

    # Summary table to stdout
    print("\n=== E2 summary ===")
    print(f"  individuals mean test_acc: {np.mean(accs) * 100:.2f}%")
    print(f"  individuals max  test_acc: {np.max(accs) * 100:.2f}%")
    print(f"  best_single   : {best_metrics['test_acc'] * 100:.2f}%")
    print(f"  uniform_soup  : {uniform_metrics['test_acc'] * 100:.2f}%")
    print(f"  greedy_soup   : {greedy_metrics['test_acc'] * 100:.2f}% ({len(included_idx)}/{n_models} models)")
    print(f"  ties          : {ties_metrics['test_acc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
