"""Sudoku 9x9: PMF-style histogram of nearest-valid-solution distance --
one bar per integer distance 0..--near-max, plus one aggregated bar for
everything above that, grouped by self_cond_mode.

A finer-grained alternative to `eval/sudoku9_nearest_puzzle_buckets.py`'s
3-bucket (valid/near/far) stacked bar: rather than collapsing distances 1..10
into one "near" bucket, this shows each integer distance as its own bar, so
the internal shape of the near-miss population (e.g. concentrated at 1-2 bits
vs. spread evenly across 1-10) is visible directly instead of being summarized
away.

Reads the CSV `eval/sudoku9_nearest_puzzle_cdf.py` already wrote
(`nearest_puzzle_<metric>distance.csv`) -- this is a different presentation
of the same measurements, not a new experiment.

Usage:
    python eval/sudoku9_nearest_puzzle_pmf.py
    python eval/sudoku9_nearest_puzzle_pmf.py --near-max 10 --metric bit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # standalone subprocess run: no GUI backend available
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODE_ORDER = ["two_pass", "carry"]
COLORS = {"two_pass": "tab:orange", "carry": "tab:purple"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="bit", choices=["cell", "bit"])
    p.add_argument("--near-max", type=int, default=10,
                   help="largest distance shown as its OWN bar; everything "
                        "above is collapsed into one '>near-max' bar")
    p.add_argument("--in-csv", default=None,
                   help="defaults to eval/results/sudoku9/"
                        "nearest_puzzle_<metric>distance.csv")
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku9"))
    args = p.parse_args()

    in_csv = Path(args.in_csv) if args.in_csv else (
        Path(args.out_dir) / f"nearest_puzzle_{args.metric}distance.csv")
    df = pd.read_csv(in_csv)
    print(f"read {len(df)} rows from {in_csv}")

    labels = [str(k) for k in range(0, args.near_max + 1)] + [f">{args.near_max}"]
    taus = sorted(df["tau"].unique())
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    fig, axes = plt.subplots(1, len(taus), figsize=(7 * len(taus), 5), sharey=True)
    if len(taus) == 1:
        axes = [axes]

    x = np.arange(len(labels))
    width = 0.8 / len(modes)

    for ax, tau in zip(axes, taus):
        sub = df[df["tau"] == tau]
        for mi, m in enumerate(modes):
            s = sub[sub["mode"] == m]
            n = len(s)
            capped = s["distance"].clip(upper=args.near_max + 1)  # collapse the tail
            counts = capped.value_counts().reindex(
                list(range(args.near_max + 1)) + [args.near_max + 1], fill_value=0)
            fracs = counts.values / n if n else counts.values * 0.0
            offset = (mi - (len(modes) - 1) / 2) * width
            bars = ax.bar(x + offset, fracs, width, label=m, color=COLORS.get(m))
            for xi, f in zip(x, fracs):
                if f > 0.01:
                    ax.text(xi + offset, f + 0.005, f"{f:.0%}", ha="center",
                            va="bottom", fontsize=6, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(f"{args.metric} distance to nearest valid solution")
        ax.set_title(f"tau={tau}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("fraction of samples")
    fig.suptitle(f"Sudoku 9x9: PMF of distance to nearest valid solution  "
                 f"(bars 0-{args.near_max} individually, '>{args.near_max}' collapsed)")
    fig.tight_layout()
    out_path = Path(args.out_dir) / f"nearest_puzzle_{args.metric}distance_pmf.png"
    fig.savefig(out_path, dpi=150)

    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
