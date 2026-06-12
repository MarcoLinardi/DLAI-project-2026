"""Aggregate E2 results into a summary table + final bar plot.

Reads:
  - results/tables/e2_individuals.csv  (5 rows, one per model)
  - results/tables/e2_merging.csv      (4 rows: best, uniform, greedy, ties)
  - results/tables/baseline_seed0.csv  (50 epoch history -> final test_acc as
                                        upper-bound oracle on the bar plot)

Writes:
  - results/tables/e2_summary.csv: for each method: test_acc, delta vs
                                     best_single (pp), delta vs uniform (pp)
  - results/plots/e2_methods_bar.png: bar chart of the 4 methods + 5
                                       individuals (grey background), with
                                       reference lines for individual-mean
                                       and full-data baseline.

"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")

INDIVIDUALS_CSV = TABLES_DIR / "e2_individuals.csv"
MERGING_CSV = TABLES_DIR / "e2_merging.csv"
BASELINE_HISTORY_CSV = TABLES_DIR / "baseline_seed0.csv"
SUMMARY_CSV = TABLES_DIR / "e2_summary.csv"
PLOT_PATH = PLOTS_DIR / "e2_methods_bar.png"

METHOD_ORDER = ["best_single", "uniform_soup", "greedy_soup", "ties"]
METHOD_LABELS = {
    "best_single": "Best single",
    "uniform_soup": "Uniform soup",
    "greedy_soup": "Greedy soup",
    "ties": "TIES",
}
METHOD_COLORS = {
    "best_single": "#2E7D32",
    "uniform_soup": "#F9A825",
    "greedy_soup": "#EF6C00",
    "ties": "#C62828",
}


def _build_summary(individuals: pd.DataFrame, merging: pd.DataFrame) -> pd.DataFrame:
    """Long-form summary table for the report."""
    by_method = merging.set_index("method")
    if "best_single" not in by_method.index or "uniform_soup" not in by_method.index:
        raise ValueError("merging CSV must contain rows for best_single and uniform_soup")
    best_acc = float(by_method.loc["best_single", "test_acc"])
    uniform_acc = float(by_method.loc["uniform_soup", "test_acc"])
    individuals_mean = float(individuals["test_acc"].mean())
    individuals_max = float(individuals["test_acc"].max())

    rows = []
    for method in METHOD_ORDER:
        if method not in by_method.index:
            continue
        acc = float(by_method.loc[method, "test_acc"])
        rows.append({
            "method": method,
            "test_acc": acc,
            "delta_vs_best_pp": (acc - best_acc) * 100.0,
            "delta_vs_uniform_pp": (acc - uniform_acc) * 100.0,
            "delta_vs_individuals_mean_pp": (acc - individuals_mean) * 100.0,
            "delta_vs_individuals_max_pp": (acc - individuals_max) * 100.0,
        })
    return pd.DataFrame(rows)


def _baseline_final_acc() -> float | None:
    """Final test_acc of the full-data baseline (Phase 1 50-epoch run)."""
    if not BASELINE_HISTORY_CSV.exists():
        return None
    df = pd.read_csv(BASELINE_HISTORY_CSV)
    if "test_acc" not in df.columns or df.empty:
        return None
    return float(df["test_acc"].iloc[-1])


def _plot_bar(individuals: pd.DataFrame, merging: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_indiv = len(individuals)
    n_methods = sum(1 for m in METHOD_ORDER if m in set(merging["method"]))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=140)

    # Individuals on the left (grey)
    indiv_xs = np.arange(n_indiv)
    indiv_accs = individuals.sort_values("model_idx")["test_acc"].values * 100.0
    ax.bar(indiv_xs, indiv_accs, width=0.7, color="#BDBDBD", edgecolor="#616161",
           label="Individual models")

    # Gap between individuals and methods
    gap = 0.8
    method_xs = []
    method_accs = []
    method_colors = []
    method_labels = []
    for i, method in enumerate([m for m in METHOD_ORDER if m in set(merging["method"])]):
        row = merging[merging["method"] == method].iloc[0]
        x = n_indiv + gap + i
        method_xs.append(x)
        method_accs.append(float(row["test_acc"]) * 100.0)
        method_colors.append(METHOD_COLORS[method])
        method_labels.append(METHOD_LABELS[method])

    bars = ax.bar(method_xs, method_accs, width=0.7, color=method_colors,
                  edgecolor="#212121")

    # Annotate method bars with their value
    for x, acc in zip(method_xs, method_accs):
        ax.text(x, acc + 0.4, f"{acc:.1f}", ha="center", va="bottom", fontsize=9)

    # X labels
    all_xs = list(indiv_xs) + method_xs
    all_labels = [f"m{i}" for i in range(n_indiv)] + method_labels
    ax.set_xticks(all_xs)
    ax.set_xticklabels(all_labels, rotation=20, ha="right")

    # Reference lines
    mean_acc = float(individuals["test_acc"].mean()) * 100.0
    ax.axhline(mean_acc, color="#757575", linestyle="--", linewidth=1.2,
               label=f"Mean of individuals = {mean_acc:.1f}%")

    baseline_acc = _baseline_final_acc()
    if baseline_acc is not None:
        ax.axhline(baseline_acc * 100.0, color="#1565C0", linestyle="-", linewidth=1.2,
                   label=f"Baseline (full data) = {baseline_acc * 100:.1f}%")

    # Y range: zoom to where the action happens
    ymin = min(min(indiv_accs), min(method_accs)) - 3
    ymax = max(max(indiv_accs), max(method_accs)) + 3
    if baseline_acc is not None:
        ymax = max(ymax, baseline_acc * 100.0 + 2)
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("E2 — Model merging on 5 disjoint CIFAR-10 splits")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved plot -> {out_path}")


def main() -> None:
    if not INDIVIDUALS_CSV.exists() or not MERGING_CSV.exists():
        raise SystemExit(
            f"missing input CSV(s); run src.experiments.e2_merging first.\n"
            f"  expected: {INDIVIDUALS_CSV} and {MERGING_CSV}"
        )

    individuals = pd.read_csv(INDIVIDUALS_CSV)
    merging = pd.read_csv(MERGING_CSV)

    summary = _build_summary(individuals, merging)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"saved summary -> {SUMMARY_CSV}")
    print(summary.to_string(index=False))

    _plot_bar(individuals, merging, PLOT_PATH)


if __name__ == "__main__":
    main()
