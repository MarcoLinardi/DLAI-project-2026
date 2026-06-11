"""Train N ResNet-20 models on N disjoint CIFAR-10 splits for experiment E2.

All N models share the same initialization (controlled by `init_seed`) but
see disjoint training subsets. Each member has its own SGD-noise seed so
the batch order / augmentation are independent per model.

The shared init is the canonical Wortsman 2022 / Yadav 2023 setup for
model soup and task arithmetic: keeps all theta_i in the same linear-mode
basin (we verified this empirically in E1 setting B → low barrier), so
that uniform averaging has a geometric meaning.

CLI:
    python -m src.experiments.train_e2 --config configs/e2_soup.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.partition import get_cifar10_partition_loaders
from src.models.resnet20 import resnet20
from src.training.train import load_config, train_model, write_history_csv
from src.training.utils import count_parameters, device_auto, save_checkpoint, set_seed


def _sgd_seed(i: int) -> int:
    """SGD-noise seed for model i. Distinct per model so batch order /
    augmentation are independent across the N members."""
    return 20_000 + i


def main() -> None:
    parser = argparse.ArgumentParser(description="Train N ResNet-20 models on disjoint splits for E2.")
    parser.add_argument("--config", type=str, required=True, help="Path to E2 YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    e2 = cfg.extra.get("e2", {})
    if not e2:
        sys.exit(f"error: config {args.config} has no top-level `e2:` block")

    n_models = int(e2["n_models"])
    n_parts = int(e2["n_parts"])
    partition_seed = int(e2["partition_seed"])
    init_seed = int(e2["init_seed"])

    if n_models > n_parts:
        sys.exit(f"error: n_models={n_models} > n_parts={n_parts}; cannot assign disjoint splits")

    device = device_auto()
    print(
        f"device={device}, n_models={n_models}, n_parts={n_parts}, "
        f"partition_seed={partition_seed}, init_seed={init_seed}, epochs={cfg.epochs}"
    )

    for i in range(n_models):
        print(f"\n=== model {i + 1}/{n_models} (part_id={i}) ===")
        sgd_seed = _sgd_seed(i)
        print(f"--- init_seed={init_seed} (shared), sgd_seed={sgd_seed} ---")

        # Step 1: shared init — same theta_0 for all members of the soup.
        set_seed(init_seed)
        model = resnet20(num_classes=10)
        n_params = count_parameters(model)
        print(f"model: ResNet-20, params={n_params:,}")

        # Step 2: re-seed for SGD/data randomness, then build loader for this split.
        set_seed(sgd_seed)
        train_loader, test_loader = get_cifar10_partition_loaders(
            part_id=i,
            n_parts=n_parts,
            partition_seed=partition_seed,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            root=cfg.data_root,
            augment=cfg.augment,
        )
        print(f"split: {len(train_loader.dataset)} train samples (part_id={i}/{n_parts})")

        # Step 3: train.
        history = train_model(model, train_loader, test_loader, cfg, device=device)
        final = history[-1]

        # Step 4: save checkpoint + history.
        run_name = f"e2_model{i}"
        ckpt_path = Path(cfg.checkpoint_dir) / f"{run_name}.pt"
        csv_path = Path(cfg.history_dir) / f"{run_name}.csv"
        save_checkpoint(
            ckpt_path,
            model,
            metadata={
                "experiment": "E2",
                "model_idx": i,
                "part_id": i,
                "n_parts": n_parts,
                "partition_seed": partition_seed,
                "init_seed": init_seed,
                "sgd_seed": sgd_seed,
                "epochs": cfg.epochs,
                "config": args.config,
                "final_test_acc": final["test_acc"],
                "final_train_acc": final["train_acc"],
            },
        )
        write_history_csv(csv_path, history)
        print(f"saved checkpoint -> {ckpt_path}")
        print(f"saved history    -> {csv_path}")
        print(f"final test_acc   = {final['test_acc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
