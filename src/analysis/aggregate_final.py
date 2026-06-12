"""Final pipeline figure for the 2-pp DLAI report.

For each E3 pair (E1D_pair0 + E2_model0x2) produces a 3-bar group showing
the midpoint test accuracy at three stages of the merging pipeline:
  1. pre-merge      -> naive midpoint, no alignment, BN reset only
  2. post-AM        -> after iterative Activation Matching of model_B onto A
  3. post-REPAIR    -> after 2 epochs of low-LR fine-tune of the merged model

A dashed reference line per group shows the average endpoint accuracy
(individual models' test acc, averaged) — the upper bound the merge tries
to approach.

Reads:
  - results/tables/e3_curves.csv     (pre + post AM midpoints, alpha=0.5)
  - results/tables/e3_repair.csv     (post REPAIR test acc per pair)

Writes:
  - results/plots/e3_pipeline.png    (figure 2 of the report)
  - results/tables/e3_pipeline_summary.csv  (raw numbers behind the figure)

"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TABLES_DIR = Path("results/tables")
PLOTS_DIR = Path("results/plots")
CURVES_CSV = TABLES_DIR / "e3_curves.csv"
REPAIR_CSV = TABLES_DIR / "e3_repair.csv"
PLOT_PATH = PLOTS_DIR / "e3_pipeline.png"
SUMMARY_CSV = TABLES_DIR / "e3_pipeline_summary.csv"

STAGE_COLORS = {
    "pre-merge": "#C62828",     # red
    "post-AM": "#F9A825",       # amber
    "post-REPAIR": "#2E7D32",   # green
}
STAGE_ORDER = ["pre-merge", "post-AM", "post-REPAIR"]

PAIR_DISPLAY = {
    "E1D_pair0": "E1-D (diff init, paper-classic)",
    "E2_model0x2": "E2 (same init, low-resource)",
}


def main() -> None:
    if not CURVES_CSV.exists():
        raise SystemExit(f"missing {CURVES_CSV}; run src.experiments.e3_align first")
    if not REPAIR_CSV.exists():
        raise SystemExit(f"missing {REPAIR_CSV}; run src.experiments.e3_repair first")

    curves = pd.read_csv(CURVES_CSV)
    repair = pd.read_csv(REPAIR_CSV)

    # Build per-pair summary
    rows: list[dict] = []
    for pair in curves["pair"].unique():
        sub = curves[curves["pair"] == pair]

        # Endpoint accuracy: average of alpha=0.0 and alpha=1.0 from the "pre"
        # curve (post-AM endpoints are equivalent — only the interior moves).
        pre_curve = sub[sub["alignment"] == "pre"]
        endpoint_acc = 0.5 * (
            pre_curve[pre_curve["alpha"] == 0.0]["test_acc"].iloc[0]
            + pre_curve[pre_curve["alpha"] == 1.0]["test_acc"].iloc[0]
        )

        # pre-merge midpoint
        pre_mid = pre_curve[pre_curve["alpha"] == 0.5]["test_acc"].iloc[0]

        # post-AM midpoint
        post_curve = sub[sub["alignment"] == "post"]
        post_am_mid = post_curve[post_curve["alpha"] == 0.5]["test_acc"].iloc[0]

        # post-REPAIR test acc
        rep_row = repair[repair["pair"] == pair]
        if rep_row.empty:
            print(f"warn: no REPAIR row for {pair}")
            post_rep = float("nan")
        else:
            post_rep = float(rep_row["test_acc_post_repair"].iloc[0])

        rows.append({
            "pair": pair,
            "endpoint_acc_avg": endpoint_acc,
            "pre_merge_midpoint": pre_mid,
            "post_AM_midpoint": post_am_mid,
            "post_REPAIR_test_acc": post_rep,
        })

    summary = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"saved summary -> {SUMMARY_CSV}")
    print(summary.to_string(index=False))

    # ---- Plot ----
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    n_pairs = len(summary)
    n_stages = 3
    bar_width = 0.22
    group_gap = 0.7

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)

    centers = np.arange(n_pairs) * group_gap * (n_stages + 1.5)

    for i, stage in enumerate(STAGE_ORDER):
        col = {"pre-merge": "pre_merge_midpoint",
               "post-AM": "post_AM_midpoint",
               "post-REPAIR": "post_REPAIR_test_acc"}[stage]
        xs = centers + (i - 1) * (bar_width * 1.1)
        ys = summary[col].values * 100.0
        ax.bar(xs, ys, width=bar_width, color=STAGE_COLORS[stage],
               edgecolor="#212121", label=stage)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.8, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

    # Endpoint reference lines per group
    span = bar_width * 1.1 * (n_stages - 1) + bar_width
    for i, ep in enumerate(summary["endpoint_acc_avg"].values):
        x0 = centers[i] - span / 2
        x1 = centers[i] + span / 2
        ax.hlines(ep * 100.0, x0, x1, colors="#555555", linestyles="--", linewidth=1.5)
        ax.text(x1 + 0.02, ep * 100.0, f"avg endpoint = {ep * 100:.1f}%",
                ha="left", va="center", fontsize=8, color="#555555")

    ax.set_xticks(centers)
    ax.set_xticklabels([PAIR_DISPLAY.get(p, p) for p in summary["pair"]],
                       fontsize=9)
    ax.set_ylabel("test accuracy at midpoint (%)")
    ax.set_title("Merging pipeline: pre-merge $\\to$ AM $\\to$ AM + REPAIR")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.set_ylim(0, 95)

    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    print(f"saved plot -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
