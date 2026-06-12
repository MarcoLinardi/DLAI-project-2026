"""Aggregate E5 data-fraction results into a summary table and figure.

Reads:
  results/tables/e5_{frac}k/e2_merging.csv     uniform soup + greedy + TIES per fraction
  results/tables/e5_{frac}k/e3_repair.csv      AM+REPAIR midpoint accuracy per fraction
  results/tables/e2_merging.csv                baselines for the existing 10k fraction
  results/tables/e3_repair.csv                 AM+REPAIR for 10k (existing)

Outputs:
  results/tables/e5_summary.csv   one row per (fraction, method)
  results/plots/e5_fraction.png   accuracy vs samples-per-model, 3 curves
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FRACTIONS = [1000, 2000, 5000, 10000]
FRAC_LABELS = {1000: "1k", 2000: "2k", 5000: "5k", 10000: "10k"}

TABLES = Path("results/tables")
PLOTS = Path("results/plots")
PLOTS.mkdir(parents=True, exist_ok=True)


def _read_merging(csv_path: Path) -> dict[str, float]:
    """Return {method: test_acc} from an e2_merging.csv."""
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["method"]] = float(row["test_acc"])
    return out


def _read_repair(csv_path: Path) -> float | None:
    """Return mean post-REPAIR test_acc across pairs from an e3_repair.csv."""
    if not csv_path.exists():
        return None
    accs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            accs.append(float(row["test_acc_post_repair"]))
    return float(np.mean(accs)) if accs else None


def main() -> None:
    rows = []
    uniform_accs, greedy_accs, repair_accs = [], [], []

    for frac in FRACTIONS:
        tag = FRAC_LABELS[frac]
        if frac == 10000:
            # reuse existing E2/E3 results
            merging_path = TABLES / "e2_merging.csv"
            repair_path = TABLES / "e3_repair.csv"
        else:
            merging_path = TABLES / f"e5_{tag}" / "e2_merging.csv"
            repair_path = TABLES / f"e5_{tag}" / "e3_repair.csv"

        merging = _read_merging(merging_path)
        repair_acc = _read_repair(repair_path)

        # filter only e5 pairs (exclude E1D which has different init)
        if frac == 10000 and repair_path == TABLES / "e3_repair.csv":
            # compute mean only over E2 pairs (excluding E1D_pair0)
            accs_e2 = []
            with open(repair_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if "E1D" not in row["pair"]:
                        accs_e2.append(float(row["test_acc_post_repair"]))
            repair_acc = float(np.mean(accs_e2)) if accs_e2 else repair_acc

        row = {
            "samples_per_model": frac,
            "uniform_soup": merging.get("uniform_soup", float("nan")),
            "greedy_soup": merging.get("greedy_soup", float("nan")),
            "am_repair": repair_acc if repair_acc is not None else float("nan"),
        }
        rows.append(row)
        uniform_accs.append(row["uniform_soup"])
        greedy_accs.append(row["greedy_soup"])
        repair_accs.append(row["am_repair"])
        print(f"frac={frac:>5d}: uniform={row['uniform_soup']*100:.2f}%  "
              f"greedy={row['greedy_soup']*100:.2f}%  AM+REPAIR={row['am_repair']*100:.2f}%")

    # Write summary CSV
    out_csv = TABLES / "e5_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["samples_per_model", "uniform_soup", "greedy_soup", "am_repair"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nsaved -> {out_csv}")

    # Plot
    x = [r["samples_per_model"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(x, [v * 100 for v in uniform_accs], "o--", color="#e15759", label="Uniform soup")
    ax.plot(x, [v * 100 for v in greedy_accs],  "s--", color="#4e79a7", label="Greedy soup")
    ax.plot(x, [v * 100 for v in repair_accs],   "^-",  color="#59a14f", linewidth=2,
            label="AM + REPAIR (midpoint, mean)")
    ax.set_xscale("log")
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels([FRAC_LABELS[f] for f in FRACTIONS])
    ax.set_xlabel("Samples per model")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("E5 — Merging accuracy vs data fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_fig = PLOTS / "e5_fraction.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_fig}")


if __name__ == "__main__":
    main()
