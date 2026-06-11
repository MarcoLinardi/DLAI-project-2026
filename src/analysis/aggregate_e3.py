"""Aggregate E3 results: pre- vs post-alignment curves into a single figure.

Reads `results/tables/e3_curves.csv` (written by src.experiments.e3_align)
and produces a side-by-side plot, one column per pair, showing test loss
vs alpha before (dashed grey) and after (solid color) Git Re-Basin
alignment. Reads `e3_barriers.csv` for the numeric summary printed at
the end.

CLI:
    python -m src.analysis.aggregate_e3
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")
CURVES_CSV = TABLES_DIR / "e3_curves.csv"
BARRIERS_CSV = TABLES_DIR / "e3_barriers.csv"
PLOT_PATH = PLOTS_DIR / "e3_alignment.png"

PAIR_COLORS = {
    "E1D_pair0": "#C62828",     # red — diff init, classical Re-Basin case
    "E2_model0x2": "#1565C0",   # blue — low-resource regime from E2
}


def main() -> None:
    if not CURVES_CSV.exists():
        raise SystemExit(f"missing {CURVES_CSV}; run src.experiments.e3_align first")
    curves = pd.read_csv(CURVES_CSV)
    barriers = pd.read_csv(BARRIERS_CSV) if BARRIERS_CSV.exists() else None

    pairs = list(curves["pair"].unique())
    n_pairs = len(pairs)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5.5 * n_pairs, 4.0), dpi=140,
                             squeeze=False)
    axes = axes[0]  # 1xN -> N

    for ax, pair in zip(axes, pairs):
        sub = curves[curves["pair"] == pair]
        color = PAIR_COLORS.get(pair, "#444444")

        pre = sub[sub["alignment"] == "pre"].sort_values("alpha")
        post = sub[sub["alignment"] == "post"].sort_values("alpha")

        ax.plot(pre["alpha"], pre["test_loss"], "o--", color="#888888",
                markersize=4, label="pre-alignment")
        ax.plot(post["alpha"], post["test_loss"], "o-", color=color,
                markersize=4, label="post-alignment")

        ax.set_xlabel(r"interpolation coefficient $\alpha$")
        ax.set_ylabel("test loss")
        ax.set_title(pair)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True, fontsize=9)

    fig.suptitle("E3 — Git Re-Basin: linear interpolation curve before vs after alignment")
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")

    # Summary table
    if barriers is not None:
        print("\n=== E3 alignment summary ===")
        print(barriers.to_string(index=False))


if __name__ == "__main__":
    main()
