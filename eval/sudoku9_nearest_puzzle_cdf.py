"""Sudoku 9x9: is an "invalid" grid actually close to a real solution, or way off?

The validity reward is binary (`is_valid_sudoku9_batch`), so the validity rate
alone can't distinguish "these grids are one digit away from solved" from
"these grids are nowhere close" -- two very different failure modes that would
score identically. `tilt/sudoku9_distance.py` (a CP-SAT
exact-distance solver) supplies the graded companion measure: the
EXACT minimum number of cells/bits that must change for a grid to become a
legal solution, via CP-SAT (not a heuristic).

This script draws samples at a fixed (tau, K), scores every one of them with
`score_grid` (0 distance for already-valid grids, no solver call needed), and
plots the empirical CDF of the distance -- separately per self_cond_mode, so
the two_pass/carry comparison from `eval/sudoku9_k_sweep.py` /
`eval/sudoku9_selfcond_mode_check.py` gets a graded answer too, not just a
pass/fail rate.

Cost note: CP-SAT is only invoked for INVALID grids (score_grid skips it for
already-valid ones), so the two_pass points (lower validity) cost more solver
time than the carry points, not less.

Usage:
    python eval/sudoku9_nearest_puzzle_cdf.py --n 50 --taus 3 --K 1000  # quick timing check
    python eval/sudoku9_nearest_puzzle_cdf.py --n 300 --taus 3 5 --K 1000 --modes two_pass carry
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # standalone subprocess run: no GUI backend available
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.checkpoints import load_premier  # noqa: E402
from common.sample_config import SCHEDULE, T  # noqa: E402

from tilt.sudoku9 import D, decode_grid_batch, success_rate_exact  # noqa: E402
from tilt.sudoku9_distance import score_grid  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku9_v1_l17_d512x8_xf_sc_s8264_180k.pt"
ROW_NOISE = 0.02
P_INJECT = 0.05


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--row-noise", type=float, default=ROW_NOISE)
    p.add_argument("--p-inject", type=float, default=P_INJECT)
    p.add_argument("--taus", type=float, nargs="+", default=[3.0])
    p.add_argument("--K", type=int, default=1000)
    p.add_argument("--modes", nargs="+", default=["two_pass", "carry"],
                   choices=["two_pass", "carry"])
    p.add_argument("--n", type=int, default=50, help="draws per (mode, tau) point")
    p.add_argument("--metric", default="bit", choices=["cell", "bit"])
    p.add_argument("--solver-time-limit", type=float, default=5.0,
                   help="per-grid CP-SAT cap in seconds")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku9"))
    p.add_argument("--replot-only", action="store_true",
                   help="skip sampling/solving entirely and just re-plot from "
                        "the existing nearest_puzzle_<metric>distance.csv in "
                        "--out-dir (or --in-csv) -- use this after changing "
                        "only the PLOT, not the underlying data/taus/modes")
    p.add_argument("--in-csv", default=None,
                   help="CSV to replot from when --replot-only is set; "
                        "defaults to nearest_puzzle_<metric>distance.csv in --out-dir")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    if args.replot_only:
        in_csv = Path(args.in_csv) if args.in_csv else (
            out_dir / f"nearest_puzzle_{args.metric}distance.csv")
        df = pd.read_csv(in_csv)
        print(f"--replot-only: read {len(df)} rows from {in_csv}, skipping sampling")
        args.taus = sorted(df["tau"].unique())
        args.modes = [m for m in ("two_pass", "carry") if m in df["mode"].unique()]
        _make_plot(df, args, out_dir)
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  device={device}")
    d = shim.net.cfg.d
    print(f"d={d}  taus={args.taus}  K={args.K}  n={args.n}/point  modes={args.modes}  "
          f"metric={args.metric}")

    rows = []
    for tau in args.taus:
        exact = success_rate_exact(args.row_noise, args.p_inject, tau=tau)
        for mode in args.modes:
            t0 = time.time()
            h0 = torch.zeros(d)
            h0[-1] = tau
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                raw = sl(drift_arg, d=d, n=args.n, h=h0, T=T, K=args.K, schedule=SCHEDULE,
                         seed=args.seed, self_cond_mode=mode)
            hard = decode(raw, drift_arg, h0)
            core = hard[:, :D]
            grids = decode_grid_batch(core).numpy()
            t_sample = time.time() - t0

            t1 = time.time()
            n_solved = 0
            for i in range(args.n):
                out = score_grid(grids[i], metrics=(args.metric,),
                                  time_limit=args.solver_time_limit)
                if out["status"] != "VALID":
                    n_solved += 1
                rows.append({
                    "mode": mode, "tau": tau, "K": args.K, "exact": exact,
                    "sample_idx": i, "distance": out[f"{args.metric}_distance"],
                    "n_bad_symbol": out["n_bad_symbol"], "status": out["status"],
                    "row_viol": out["row_viol"], "col_viol": out["col_viol"],
                    "box_viol": out["box_viol"],
                })
            t_solve = time.time() - t1
            print(f"  mode={mode:9s} tau={tau:4.1f}  sampled {args.n} in {t_sample:.1f}s, "
                  f"solved {n_solved} invalid grids in {t_solve:.1f}s "
                  f"({t_solve / max(n_solved, 1):.3f}s/grid)")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"nearest_puzzle_{args.metric}distance.csv", index=False)
    print(f"saved: {out_dir / f'nearest_puzzle_{args.metric}distance.csv'}")

    _make_plot(df, args, out_dir)


def _make_plot(df: pd.DataFrame, args, out_dir: Path) -> None:
    """Row 1: linear y, with a horizontal line at the EXACT theoretical success

    rate for that tau -- this is "the percent correct we'd recover if we
    accepted anything within X bits of a valid solution" against the ceiling
    a perfect sampler could reach, on the same axes. Row 2: identical data,
    log-scale y, to make the (usually tiny) residual failure mass visible
    instead of getting crushed against y=1.
    """
    # Row 2 uses matplotlib's built-in 'logit' y-scale rather than plain log:
    # a log-scale y only stretches out values near 0, but almost everything
    # here (the CDF's whole interesting range) sits close to 1 -- a log-odds
    # transform stretches BOTH ends, which is what actually separates the two
    # modes' curves near the top instead of leaving them crushed together.
    # Points at cdf==1.0 (logit undefined) are automatically masked/dropped by
    # the scale rather than erroring (nonpositive="mask", the default).
    exact_by_tau = {tau: float(df.loc[df["tau"] == tau, "exact"].iloc[0]) for tau in args.taus}
    fig, axes = plt.subplots(2, len(args.taus), figsize=(6 * len(args.taus), 9), sharex="col")
    if len(args.taus) == 1:
        axes = axes.reshape(2, 1)
    colors = {"two_pass": "tab:orange", "carry": "tab:purple"}
    for col, tau in enumerate(args.taus):
        for row_i, yscale in enumerate(("linear", "logit")):
            ax = axes[row_i, col]
            for mode in args.modes:
                sub = df[(df["tau"] == tau) & (df["mode"] == mode)]
                dist = np.sort(sub["distance"].values)
                cdf = np.arange(1, len(dist) + 1) / len(dist)
                ax.step(dist, cdf, where="post", label=mode, color=colors.get(mode))
            ax.axhline(exact_by_tau[tau], ls="--", color="tab:blue",
                       label="exact theoretical valid rate" if row_i == 0 else None)
            ax.set_yscale(yscale)
            ax.grid(alpha=0.3, which="both")
            if row_i == 0:
                ax.set_title(f"tau={tau}")
                ax.set_ylim(0, 1.02)
                ax.legend(fontsize=8)
            else:
                ax.set_xlabel(f"{args.metric} distance to nearest valid solution")
    axes[0, 0].set_ylabel("cumulative fraction of samples")
    axes[1, 0].set_ylabel("cumulative fraction of samples (logit scale)")
    # Derived from the data itself (not args.K/args.n), so a --replot-only run
    # (which never set those from the CLI) still gets an accurate title.
    k_vals = sorted(df["K"].unique())
    n_per_point = df.groupby(["tau", "mode"]).size().min()
    fig.suptitle(f"Sudoku 9x9: CDF of distance to nearest valid solution  "
                 f"(K={k_vals[0] if len(k_vals) == 1 else k_vals}, "
                 f"n={n_per_point}/point, metric={args.metric})")
    fig.tight_layout()
    out_path = out_dir / f"nearest_puzzle_{args.metric}distance_cdf.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
