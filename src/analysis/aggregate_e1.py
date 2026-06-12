"""Aggregate per-setting CSVs into a summary table + final plot.

Reads results/tables/e1_{A,B,D}.csv produced by src.experiments.e1_lmc,
computes barriers per (setting, pair_seed), aggregates mean/std over
pair_seed, writes results/tables/e1_barriers_summary.csv, and saves
results/plots/e1_test_loss_vs_alpha.png, the figure that goes in the
2-page report.

"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.merging.lmc import error_barrier

SETTINGS = ("A", "B", "D")
TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")
SUMMARY_CSV = TABLES_DIR / "e1_barriers_summary.csv"
PLOT_PATH = PLOTS_DIR / "e1_test_loss_vs_alpha.png"

SETTING_LABELS = {
    "A": "A: same init, same data",
    "B": "B: same init, split data",
    "D": "D: diff init, split data",
}
SETTING_COLORS = {"A": "#2E7D32", "B": "#1565C0", "D": "#C62828"}


def _load_one(setting: str) -> pd.DataFrame | None:
    path = TABLES_DIR / f"e1_{setting}.csv"
    if not path.exists():
        print(f"warn: {path} not found, skipping setting {setting}")
        return None
    df = pd.read_csv(path)
    needed = {"setting", "pair_seed", "alpha", "train_loss", "train_acc", "test_loss", "test_acc"}
    missing = needed - set(df.columns)
    if missing:
        print(f"warn: {path} missing columns {missing}, skipping")
        return None
    return df


def _barriers_per_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair barriers (one row per pair_seed)."""
    out: list[dict] = []
    for s, group in df.groupby("pair_seed"):
        rows = group.to_dict("records")
        out.append({
            "setting": df["setting"].iloc[0],
            "pair_seed": int(s),
            "barrier_test_loss": error_barrier(rows, "test_loss"),
            "barrier_test_acc": error_barrier(rows, "test_acc"),
        })
    return pd.DataFrame(out)


def _summary(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Mean/std over pair_seed within each setting."""
    grouped = per_pair.groupby("setting")
    summary = grouped.agg(
        n_pairs=("pair_seed", "count"),
        barrier_test_loss_mean=("barrier_test_loss", "mean"),
        barrier_test_loss_std=("barrier_test_loss", "std"),
        barrier_test_acc_mean=("barrier_test_acc", "mean"),
        barrier_test_acc_std=("barrier_test_acc", "std"),
    ).reset_index()
    # std with n=1 -> NaN; replace with 0 for display
    summary = summary.fillna(0.0)
    return summary


def _plot_curves(all_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=140)
    for setting in SETTINGS:
        sub = all_df[all_df["setting"] == setting]
        if sub.empty:
            continue
        agg = sub.groupby("alpha")["test_loss"].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0.0)
        color = SETTING_COLORS[setting]
        ax.plot(agg["alpha"], agg["mean"], marker="o", markersize=4,
                color=color, label=SETTING_LABELS[setting])
        ax.fill_between(agg["alpha"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel(r"interpolation coefficient $\alpha$")
    ax.set_ylabel("test loss")
    ax.set_title(r"E1 — Linear Mode Connectivity: test loss vs $\alpha$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved plot -> {out_path}")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for setting in SETTINGS:
        df = _load_one(setting)
        if df is not None:
            frames.append(df)
    if not frames:
        raise SystemExit("no per-setting CSV found; run src.experiments.e1_lmc first")

    all_df = pd.concat(frames, ignore_index=True)
    per_pair = pd.concat([_barriers_per_pair(f) for f in frames], ignore_index=True)
    summary = _summary(per_pair)

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"saved summary -> {SUMMARY_CSV}")
    print(summary.to_string(index=False))

    _plot_curves(all_df, PLOT_PATH)


if __name__ == "__main__":
    main()
