"""Sudoku 9x9: bucketed view of the nearest-valid-solution distance -- an
alternative to the full CDF in `eval/sudoku9_nearest_puzzle_cdf.py`.

Reads the CSV that script already wrote (`nearest_puzzle_<metric>distance.csv`)
rather than re-sampling -- this is a different PRESENTATION of the same
measurements, not a new experiment.

Three buckets, chosen to separate "solved," "near miss," and "genuinely far"
into one readable stacked bar per (tau, mode):

  0            exactly valid
  1..--near-max   a near miss (default cutoff 10 bits, roughly <=2-3 cells)
  >--near-max     far -- shown as a box plot of the ACTUAL distances in this
                  bucket (not just its count), since the far tail's spread is
                  exactly the thing a single stacked-bar percentage would hide
                  (see the "far tail" finding in the CDF version: tau=3's far
                  samples ranged up to bit-distance 70, not a tight cluster
                  just above the cutoff).

Usage:
    python eval/sudoku9_nearest_puzzle_buckets.py
    python eval/sudoku9_nearest_puzzle_buckets.py --near-max 10 --metric bit
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
COLORS = {"valid": "tab:green", "near": "tab:orange", "far": "tab:red"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="bit", choices=["cell", "bit"])
    p.add_argument("--near-max", type=int, default=10,
                   help="upper (inclusive) edge of the 'near miss' bucket; "
                        "anything above this is 'far'")
    p.add_argument("--in-csv", default=None,
                   help="defaults to eval/results/sudoku9/"
                        "nearest_puzzle_<metric>distance.csv")
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku9"))
    args = p.parse_args()

    in_csv = Path(args.in_csv) if args.in_csv else (
        Path(args.out_dir) / f"nearest_puzzle_{args.metric}distance.csv")
    df = pd.read_csv(in_csv)
    print(f"read {len(df)} rows from {in_csv}")

    def bucket(d):
        if d == 0:
            return "valid"
        if d <= args.near_max:
            return "near"
        return "far"

    df["bucket"] = df["distance"].apply(bucket)
    taus = sorted(df["tau"].unique())
    modes = [m for m in MODE_ORDER if m in df["mode"].unique()]

    fig, axes = plt.subplots(2, len(taus), figsize=(5.5 * len(taus), 8),
                              sharex="col", gridspec_kw={"height_ratios": [1.3, 1]})
    if len(taus) == 1:
        axes = axes.reshape(2, 1)

    for col, tau in enumerate(taus):
        sub = df[df["tau"] == tau]

        # --- top: stacked bar of bucket percentages, one bar per mode ------
        ax = axes[0, col]
        x = np.arange(len(modes))
        bottoms = np.zeros(len(modes))
        counts_by_mode = {}
        for bname in ("valid", "near", "far"):
            fracs = []
            for m in modes:
                s = sub[sub["mode"] == m]
                n = len(s)
                cnt = int((s["bucket"] == bname).sum())
                counts_by_mode.setdefault(m, {})[bname] = cnt
                fracs.append(cnt / n if n else 0.0)
            fracs = np.array(fracs)
            ax.bar(x, fracs, bottom=bottoms, color=COLORS[bname],
                   label=f"{bname} (0)" if bname == "valid"
                   else (f"near (1-{args.near_max})" if bname == "near"
                         else f"far (>{args.near_max})"))
            for xi, (f, b) in enumerate(zip(fracs, bottoms)):
                if f > 0.02:
                    ax.text(xi, b + f / 2, f"{f:.0%}", ha="center", va="center", fontsize=8)
            bottoms += fracs
        ax.set_xticks(x)
        ax.set_xticklabels(modes)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"tau={tau}")
        if col == 0:
            ax.set_ylabel(f"fraction of samples ({args.metric} distance)")
        ax.legend(fontsize=7, loc="lower right")

        # --- bottom: box plot of the FAR bucket's actual distances --------
        ax2 = axes[1, col]
        far_data = [sub[(sub["mode"] == m) & (sub["bucket"] == "far")]["distance"].values
                    for m in modes]
        n_far = [len(a) for a in far_data]
        bp = ax2.boxplot([a if len(a) else [np.nan] for a in far_data], showmeans=True)
        ax2.set_xticks(range(1, len(modes) + 1))
        ax2.set_xticklabels([f"{m}\n(n={nf})" for m, nf in zip(modes, n_far)])
        ax2.set_ylabel(f"{args.metric} distance\n(far bucket only)")
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Sudoku 9x9: valid / near-miss / far breakdown  "
                 f"(near-max={args.near_max} {args.metric}s)")
    fig.tight_layout()
    out_path = Path(args.out_dir) / f"nearest_puzzle_{args.metric}distance_buckets.png"
    fig.savefig(out_path, dpi=150)

    # console summary
    print(df.groupby(["tau", "mode", "bucket"]).size().unstack(fill_value=0))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
