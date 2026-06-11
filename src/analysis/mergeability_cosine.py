"""E4 — Mergeability cosine metric.

For each unique pair (i, j) of the 5 E2 models (10 pairs in total),
compute cos(τ_i, τ_j) where τ_k = θ_k − θ_baseline is the task vector,
and pair it with the empirical accuracy_drop = (acc(θ_i) + acc(θ_j))/2
− acc(midpoint), measured after a BN-reset evaluation at α=0.5.

If the cosine similarity predicts mergeability (the README hypothesis),
we should see a strong negative correlation: high cos_sim → small drop.

Reuses:
  - `compute_task_vectors` from src.merging.ties for tau_i.
  - `interpolate_state_dicts` + `_eval_at_point` from src.merging.lmc for midpoint eval.

CLI:
    python -m src.analysis.mergeability_cosine --config configs/e2_soup.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from torchvision import datasets  # noqa: E402

from src.data.cifar import _build_transforms, get_cifar10_loaders  # noqa: E402
from src.merging.lmc import _eval_at_point, interpolate_state_dicts  # noqa: E402
from src.merging.ties import compute_task_vectors  # noqa: E402
from src.models.resnet20 import resnet20  # noqa: E402
from src.training.train import load_config  # noqa: E402
from src.training.utils import device_auto  # noqa: E402

TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")
OUT_CSV = TABLES_DIR / "e4_cosine.csv"
OUT_PLOT = PLOTS_DIR / "e4_cosine_scatter.png"

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


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def _flatten_task_vector(tv: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate every float-typed entry of `tv` into a single 1-D tensor."""
    parts = [v.flatten() for v in tv.values() if v.dtype.is_floating_point]
    return torch.cat(parts, dim=0).float()


def main() -> None:
    parser = argparse.ArgumentParser(description="E4 — Mergeability cosine metric.")
    parser.add_argument("--config", type=str, required=True, help="Path to E2 YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    e2 = cfg.extra.get("e2", {})
    if not e2:
        sys.exit(f"error: config {args.config} has no top-level `e2:` block")
    n_models = int(e2["n_models"])
    bn_reset_batches = int(e2.get("bn_reset_batches", 200))
    base_path = Path(e2.get("baseline_checkpoint", "results/checkpoints/baseline_seed0.pt"))

    device = device_auto()
    print(f"device={device}, n_models={n_models}, bn_reset_batches={bn_reset_batches}")
    print(f"baseline: {base_path}")

    base_sd = _load_state_dict(base_path)
    sds: list[dict[str, torch.Tensor]] = []
    for i in range(n_models):
        p = Path(cfg.checkpoint_dir) / f"e2_model{i}.pt"
        sds.append(_load_state_dict(p))
        print(f"  loaded e2_model{i}.pt")

    # --- Task vectors and flat representation ---
    tvs = compute_task_vectors(sds, base_sd)
    tv_flat = [_flatten_task_vector(tv) for tv in tvs]
    norms = [t.norm().item() for t in tv_flat]
    print("\ntask vector norms:", [f"{n:.2f}" for n in norms])

    # --- Loaders for midpoint eval ---
    bn_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        root=cfg.data_root,
        augment=True,
    )
    train_eval_loader = _build_train_eval_loader(cfg.data_root, cfg.num_workers)
    model_ctor = lambda: resnet20(num_classes=10)

    # Individual accuracies under BN-reset protocol (cached: we redo them here).
    print("\n=== Re-evaluating individuals (BN-reset apples-to-apples) ===")
    indiv_acc: list[float] = []
    indiv_loss: list[float] = []
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
        indiv_acc.append(m["test_acc"])
        indiv_loss.append(m["test_loss"])
        print(f"  model {i}: test_acc={m['test_acc'] * 100:.2f}%, test_loss={m['test_loss']:.4f}")

    # --- Iterate over all pairs ---
    rows: list[dict] = []
    print(f"\n=== Midpoint eval for {n_models * (n_models - 1) // 2} pairs ===")
    for i, j in combinations(range(n_models), 2):
        cos_sim = torch.nn.functional.cosine_similarity(
            tv_flat[i].unsqueeze(0), tv_flat[j].unsqueeze(0)
        ).item()

        sd_mid = interpolate_state_dicts(sds[i], sds[j], 0.5)
        m = _eval_at_point(
            model_ctor=model_ctor,
            sd=sd_mid,
            bn_loader=bn_loader,
            train_eval_loader=train_eval_loader,
            test_loader=test_loader,
            device=device,
            bn_reset_batches=bn_reset_batches,
        )
        midpoint_acc = m["test_acc"]
        midpoint_loss = m["test_loss"]
        avg_endpoint_acc = 0.5 * (indiv_acc[i] + indiv_acc[j])
        avg_endpoint_loss = 0.5 * (indiv_loss[i] + indiv_loss[j])
        accuracy_drop = avg_endpoint_acc - midpoint_acc        # >= 0 if barrier exists
        loss_barrier = midpoint_loss - avg_endpoint_loss        # >= 0 if barrier exists

        rows.append({
            "i": i,
            "j": j,
            "cos_sim": cos_sim,
            "midpoint_acc": midpoint_acc,
            "midpoint_loss": midpoint_loss,
            "avg_endpoint_acc": avg_endpoint_acc,
            "avg_endpoint_loss": avg_endpoint_loss,
            "accuracy_drop": accuracy_drop,
            "loss_barrier": loss_barrier,
        })
        print(
            f"  pair ({i},{j}): cos_sim={cos_sim:+.4f}, midpoint_acc={midpoint_acc * 100:.2f}%, "
            f"accuracy_drop={accuracy_drop * 100:+.2f}pp, loss_barrier={loss_barrier:+.4f}"
        )

    # --- Save CSV ---
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["i", "j", "cos_sim", "midpoint_acc", "midpoint_loss",
              "avg_endpoint_acc", "avg_endpoint_loss", "accuracy_drop", "loss_barrier"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved -> {OUT_CSV}")

    # --- Pearson correlation ---
    cos_arr = np.array([r["cos_sim"] for r in rows])
    drop_arr = np.array([r["accuracy_drop"] for r in rows]) * 100.0  # pp for display
    if len(cos_arr) >= 2:
        r_pearson = np.corrcoef(cos_arr, drop_arr)[0, 1]
    else:
        r_pearson = float("nan")
    print(f"\nPearson r (cos_sim vs accuracy_drop) = {r_pearson:+.4f}")

    # --- Scatter plot + regression line ---
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=140)
    ax.scatter(cos_arr, drop_arr, s=60, color="#1565C0", edgecolor="#0D47A1", alpha=0.85)

    # Annotate each point with (i, j)
    for r in rows:
        ax.annotate(
            f"({r['i']},{r['j']})",
            (r["cos_sim"], r["accuracy_drop"] * 100.0),
            xytext=(4, 4), textcoords="offset points", fontsize=8, color="#333333",
        )

    # Linear fit
    if len(cos_arr) >= 2:
        slope, intercept = np.polyfit(cos_arr, drop_arr, 1)
        xs = np.linspace(cos_arr.min(), cos_arr.max(), 50)
        ax.plot(xs, slope * xs + intercept, "--", color="#C62828",
                linewidth=1.5, label=f"fit (r={r_pearson:+.3f})")
        ax.legend(loc="best", frameon=True)

    ax.set_xlabel(r"cos$(\tau_i, \tau_j)$")
    ax.set_ylabel("accuracy drop at midpoint (pp)")
    ax.set_title("E4 — Task-vector cosine vs merge-induced accuracy drop")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PLOT)
    plt.close(fig)
    print(f"saved plot -> {OUT_PLOT}")


if __name__ == "__main__":
    main()
