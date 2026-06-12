"""Task vector statistics: ||tau_i||_2 norms (E2) and E4 correlation stats.

For each E2 model i:
  tau_i = sd_i - sd_baseline
  ||tau_i||_2 over the concatenated float-typed entries (the same support
  TIES uses for its trim/elect/disjoint-merge math).

For the E4 correlation table (cos_sim vs accuracy_drop on the 10 unique
E2 pairs), compute:
  - Pearson r and two-sided p-value (Student's t test).
  - 95% CI via Fisher z-transform.

Outputs:
  - results/tables/e2_task_vector_norms.csv
  - results/tables/e4_cosine_stats.csv

"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from src.merging.ties import compute_task_vectors

CHECKPOINT_DIR = Path("results/checkpoints")
TABLES_DIR = Path("results/tables")

BASELINE_PATH = CHECKPOINT_DIR / "baseline_seed0.pt"
N_MODELS = 5

E4_INPUT = TABLES_DIR / "e4_cosine.csv"
NORMS_OUT = TABLES_DIR / "e2_task_vector_norms.csv"
STATS_OUT = TABLES_DIR / "e4_cosine_stats.csv"


def _load_sd(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def _flatten_float(sd: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [v.flatten().float() for v in sd.values() if v.dtype.is_floating_point]
    return torch.cat(parts, dim=0)


def _pearson_p_value(r: float, n: int) -> float:
    """Two-sided p-value for Pearson r via Student's t with n-2 dof."""
    if n < 3 or abs(r) >= 1.0:
        return float("nan")
    t = r * math.sqrt((n - 2) / max(1 - r * r, 1e-12))
    # survival function of |t| under t-distribution with n-2 dof, two-sided.
    from math import erf, sqrt
    # Use scipy if available for accuracy; fall back to a normal-approx otherwise.
    try:
        from scipy.stats import t as t_dist
        return float(2.0 * (1.0 - t_dist.cdf(abs(t), df=n - 2)))
    except ImportError:
        # Normal approximation (rough; for n=10 it overestimates significance).
        z = abs(t)
        p_one = 0.5 * (1.0 - erf(z / sqrt(2.0)))
        return float(2.0 * p_one)


def _fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n < 4 or abs(r) >= 1.0:
        return float("nan"), float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    try:
        from scipy.stats import norm
        zcrit = float(norm.ppf(1 - alpha / 2))
    except ImportError:
        zcrit = 1.959963984540054  # 97.5th percentile of standard normal
    lo_z = z - zcrit * se
    hi_z = z + zcrit * se
    lo = (math.exp(2 * lo_z) - 1) / (math.exp(2 * lo_z) + 1)
    hi = (math.exp(2 * hi_z) - 1) / (math.exp(2 * hi_z) + 1)
    return lo, hi


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx2 = sum((xs[i] - mx) ** 2 for i in range(n))
    dy2 = sum((ys[i] - my) ** 2 for i in range(n))
    denom = math.sqrt(dx2 * dy2)
    if denom == 0.0:
        return float("nan")
    return num / denom


def main() -> None:
    print(f"loading baseline {BASELINE_PATH}")
    sd_base = _load_sd(BASELINE_PATH)
    sds = []
    for i in range(N_MODELS):
        p = CHECKPOINT_DIR / f"e2_model{i}.pt"
        sds.append(_load_sd(p))
        print(f"  loaded e2_model{i}.pt")

    print("\ncomputing task vectors...")
    tvs = compute_task_vectors(sds, sd_base)

    flat_base = _flatten_float(sd_base)
    norm_base = flat_base.norm().item()
    print(f"||theta_base||_2 = {norm_base:.4f}  (over {flat_base.numel()} float params)")

    norm_rows = []
    for i, tv in enumerate(tvs):
        flat = _flatten_float(tv)
        norm_tau = flat.norm().item()
        rel = norm_tau / norm_base if norm_base > 0 else float("nan")
        norm_rows.append({
            "model_idx": i,
            "tau_norm_l2": norm_tau,
            "theta_base_norm_l2": norm_base,
            "rel_norm": rel,
        })
        print(f"  ||tau_{i}||_2 = {norm_tau:.4f}  (rel to base: {rel:.4f})")

    NORMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(NORMS_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["model_idx", "tau_norm_l2", "theta_base_norm_l2", "rel_norm"]
        )
        w.writeheader()
        w.writerows(norm_rows)
    print(f"saved -> {NORMS_OUT}")

    # --- E4 correlation stats ---
    print(f"\nloading {E4_INPUT}")
    cos: list[float] = []
    drop_pp: list[float] = []
    with open(E4_INPUT, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cos.append(float(row["cos_sim"]))
            drop_pp.append(float(row["accuracy_drop"]) * 100.0)
    n = len(cos)

    r = _pearson(cos, drop_pp)
    p = _pearson_p_value(r, n)
    lo, hi = _fisher_z_ci(r, n)
    print(
        f"n={n}, Pearson r={r:+.4f}, p-value(two-sided)={p:.4f}, "
        f"95% CI=[{lo:+.4f}, {hi:+.4f}]"
    )

    stats_rows = [{
        "n_pairs": n,
        "pearson_r": r,
        "p_value_two_sided": p,
        "ci_lo_95": lo,
        "ci_hi_95": hi,
        "cos_range_lo": float(min(cos)),
        "cos_range_hi": float(max(cos)),
        "drop_pp_range_lo": float(min(drop_pp)),
        "drop_pp_range_hi": float(max(drop_pp)),
    }]
    with open(STATS_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    print(f"saved -> {STATS_OUT}")


if __name__ == "__main__":
    main()
