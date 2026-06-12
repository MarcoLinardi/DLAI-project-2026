"""Train pairs of ResNet-20 models for experiment E1.

Reads an E1 YAML config (settings B or D) and produces two checkpoints per
pair_seed, with isolated seeds for initialization vs SGD-noise:

    init_seed = s                              # same_init=True (setting B)
    init_seed = 1000 + 10*s + k                # same_init=False (setting D)
    sgd_seed  = 10_000*s + k                   # always distinct per member

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.cifar import get_cifar10_loaders
from src.data.partition import get_cifar10_partition_loaders
from src.models.resnet20 import resnet20
from src.training.train import load_config, train_model, write_history_csv
from src.training.utils import count_parameters, device_auto, save_checkpoint, set_seed


def _resolve_init_seed(s: int, k: int, same_init: bool) -> int:
    """Initialization seed: shared across members in same-init, distinct in diff-init."""
    return s if same_init else 1000 + 10 * s + k


def _resolve_sgd_seed(s: int, k: int) -> int:
    """SGD-noise seed: always distinct between members of a pair.

    Frankle 2020 measures stability to *independent* instances of SGD noise.
    Sharing this seed would correlate the batch ordering / augmentation of
    m1 and m2 → would underestimate the barrier.
    """
    return 10_000 * s + k


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pairs of ResNet-20 for E1.")
    parser.add_argument("--config", type=str, required=True, help="Path to E1 YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    e1 = cfg.extra.get("e1", {})
    if not e1:
        sys.exit(f"error: config {args.config} has no top-level `e1:` block")

    setting = e1["setting"]
    same_init = bool(e1["same_init"])
    split_data = bool(e1["split_data"])
    pair_seeds = list(e1["pair_seeds"])

    if e1.get("reuse_baseline", False):
        baseline = e1.get("baseline_checkpoint", "results/checkpoints/baseline_seed0.pt")
        print(
            f"setting={setting}: reuse_baseline=true → no training. "
            f"e1_lmc.py will load {baseline} for both members."
        )
        return

    device = device_auto()
    print(f"device={device}, setting={setting}, same_init={same_init}, split_data={split_data}")
    print(f"pair_seeds={pair_seeds}, epochs={cfg.epochs}")

    n_pairs = len(pair_seeds)
    for i, s in enumerate(pair_seeds, start=1):
        print(f"\n=== pair {i}/{n_pairs} (pair_seed={s}) ===")
        for k in (1, 2):
            init_seed = _resolve_init_seed(s, k, same_init)
            sgd_seed = _resolve_sgd_seed(s, k)
            print(
                f"\n--- member {k}: init_seed={init_seed}, sgd_seed={sgd_seed} ---"
            )

            # Step 1: model init (isolated from data/SGD randomness).
            set_seed(init_seed)
            model = resnet20(num_classes=10)
            n_params = count_parameters(model)
            print(f"model: ResNet-20, params={n_params:,}")

            # Step 2: re-seed for SGD/data randomness, then build loaders.
            set_seed(sgd_seed)
            if split_data:
                train_loader, test_loader = get_cifar10_partition_loaders(
                    part_id=k - 1,
                    n_parts=int(e1["n_parts"]),
                    partition_seed=int(e1["partition_seed"]),
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    root=cfg.data_root,
                    augment=cfg.augment,
                )
            else:
                train_loader, test_loader = get_cifar10_loaders(
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    root=cfg.data_root,
                    augment=cfg.augment,
                )

            # Step 3: train.
            history = train_model(model, train_loader, test_loader, cfg, device=device)
            final = history[-1]

            # Step 4: save checkpoint + history.
            run_name = f"e1{setting}_pair{s}_m{k}"
            ckpt_path = Path(cfg.checkpoint_dir) / f"{run_name}.pt"
            csv_path = Path(cfg.history_dir) / f"{run_name}.csv"
            save_checkpoint(
                ckpt_path,
                model,
                metadata={
                    "setting": setting,
                    "pair_seed": s,
                    "member": k,
                    "init_seed": init_seed,
                    "sgd_seed": sgd_seed,
                    "same_init": same_init,
                    "split_data": split_data,
                    "n_parts": int(e1.get("n_parts", 1)),
                    "partition_seed": int(e1.get("partition_seed", -1)),
                    "epochs": cfg.epochs,
                    "config": args.config,
                    "final_test_acc": final["test_acc"],
                    "final_train_acc": final["train_acc"],
                },
            )
            write_history_csv(csv_path, history)
            print(f"saved checkpoint -> {ckpt_path}")
            print(f"saved history    -> {csv_path}")
            print(f"final test_acc   = {final['test_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
