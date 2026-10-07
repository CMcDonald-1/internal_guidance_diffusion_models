"""Sudoku 4x4 check 3: are valid-grid samples UNIFORM over the 288 solutions?

Checkpoint : ckpt/sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt
Dataset    : data/sudoku_reward_v2  (d=33)

The base measure is symmetric under digit relabelling and the reward set is
the full 288-solution set, so the exact prior over WHICH of the 288 a valid
grid lands on is uniform -- any departure is sampler or network error, not a
property of the target. At 4x4 this is checkable exactly (unlike 6x6/9x9,
where the valid set is far too large to enumerate); this is the "Panel B"
check from `scripts/run_sudoku4_chisq.py`, redone from scratch here at this
project's current canonical sampling config (T=20, K=K_DEFAULTS['sudoku4x4']
=1000, schedule='sstar' -- see eval/common/sample_config.py) rather than
reusing that script's stale K=200/'late' sweep outputs. `scripts/
run_sudoku4_chisq.py` scored pre-generated CSVs from an old sweep directory;
this script draws N fresh samples itself for both arms.

WHY BOTH SAMPLERS. Reading the net alone tells you nothing -- Euler-Maruyama
at finite K is itself a biased sampler. Running the identical chi-square
test on the SDE driven by the EXACT denoiser (drift_fn = target.natural_mean,
same construction as eval/sudoku4x4_check2_reward_logodds.py's "exact-operator
sampler") separates the two error sources:

    exact-denoiser SDE off uniform  ->  discretisation (same for both arms)
    net-denoiser  SDE additionally off ->  network-attributable

READING THE OUTPUT
------------------
chi2 with 287 df: expect ~287 +- 24 if uniform. z = (chi2-df)/sqrt(2*df);
|z| < 2 is consistent with uniform. `n_missing` counts how many of the 288
were never produced -- the original v3 report found the net sampler "missed"
solutions (n_missing > 0, chi2 driven by the low tail) rather than
over-visiting a favourite few.

Untilted (tau=0, h=None) by default -- this checks the base measure's
solution-index distribution, not a reward-tilted one; pass --tau to probe a
tilted natural parameter instead (still exactly uniform in closed form, since
the tilt only reweights BETWEEN reward/non-reward branches, not among the 288
solutions themselves -- see tilt/sudoku.py's S/C mixture decomposition).

Usage:
    python eval/sudoku4x4_check3_chisq_uniform.py --n 3000
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
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.checkpoints import load_premier  # noqa: E402
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.data import load_dataset  # noqa: E402
from tilt.sampler import decode_drift, sl_sample  # noqa: E402
from tilt.sudoku import D, is_valid_sudoku_batch, valid_solutions  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt"
DATA = ROOT / "data" / "sudoku_reward_v2"


def solution_index(codes: np.ndarray, sols: np.ndarray) -> np.ndarray:
    """Map each valid {-1,1}^32 row to its index in the 288-solution list.

    Exact lookup on the packed bit pattern -- no nearest-neighbour, since
    every input here is already known valid (is_valid_sudoku_batch) and
    therefore IS one of the 288.
    """
    pw = 1 << np.arange(D, dtype=np.int64)

    def key(a):
        return ((a > 0).astype(np.int64) * pw).sum(1)

    table = {int(k): i for i, k in enumerate(key(sols))}
    out = np.array([table.get(int(k), -1) for k in key(codes)], dtype=np.int64)
    if (out < 0).any():
        raise RuntimeError(
            f"{int((out < 0).sum())} valid grids not found among the 288 -- "
            "validity check and solution list disagree"
        )
    return out


def chisq_uniform(counts: np.ndarray) -> dict:
    """Chi-square against uniform over len(counts) cells."""
    n = int(counts.sum())
    k = len(counts)
    exp = n / k
    stat = float(((counts - exp) ** 2 / exp).sum()) if n else float("nan")
    df = k - 1
    return {
        "n_valid": n, "expected_per_cell": exp,
        "chi2": stat, "df": df,
        "z": (stat - df) / np.sqrt(2 * df) if n else float("nan"),
        "n_missing": int((counts == 0).sum()),
        "max_count": int(counts.max()) if n else 0,
        "min_count": int(counts.min()) if n else 0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--n", type=int, default=3000, help="samples to draw, per sampler")
    p.add_argument("--tau", type=float, default=0.0, help="reward-tilt strength (default 0 = untilted)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku4x4"))
    p.add_argument("--counts-out", default=None,
                   help="optional .npz of the raw 288-length count vectors, for plotting")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = K_DEFAULTS["sudoku4x4"]

    _x_train, _frozen, target, _meta = load_dataset(args.data)
    if target is None:
        raise SystemExit(f"{args.data} has no target.json -- exact ground truth unavailable")

    _shim, info, sl, decode, _needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  device={device}")

    sols = valid_solutions().cpu().numpy()
    print(f"  {len(sols)} valid solutions")

    h0 = target.tilt_direction(args.tau) if args.tau != 0.0 else None
    print(f"n={args.n}  tau={args.tau}  T={T}  K={K}  schedule={SCHEDULE!r}  seed={args.seed}")

    rows, all_counts = [], {}
    for arm, drift, drift_arg_for_decode in (
        ("net", drift_arg, drift_arg),
        ("exact", target.natural_mean, target.natural_mean),
    ):
        raw = sl(drift, d=target.d, n=args.n, h=h0, T=T, K=K, schedule=SCHEDULE, seed=args.seed) \
            if arm == "net" else \
            sl_sample(drift, d=target.d, n=args.n, h=h0, T=T, K=K, schedule=SCHEDULE, seed=args.seed)
        hard = decode(raw, drift_arg_for_decode, h0) if arm == "net" else \
            decode_drift(raw, drift_arg_for_decode, h0)

        core = hard[:, :D]
        ok = is_valid_sudoku_batch(core).cpu().numpy().astype(bool)
        counts = np.zeros(len(sols), dtype=np.int64)
        if ok.any():
            idx = solution_index(core[ok].cpu().numpy(), sols)
            counts = np.bincount(idx, minlength=len(sols))

        res = {"arm": arm, "n_samples": len(raw), "valid_rate": float(ok.mean()), **chisq_uniform(counts)}
        rows.append(res)
        all_counts[arm] = counts
        print(f"  {arm:<5}  valid_rate={res['valid_rate']:.4f}  n_valid={res['n_valid']:>5}  "
              f"chi2={res['chi2']:8.1f} (df={res['df']})  z={res['z']:+6.2f}  "
              f"n_missing={res['n_missing']:>3}  max={res['max_count']}  min={res['min_count']}")

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.tau == 0.0 else f"_tau{args.tau:g}"
    out_path = out_dir / f"check3_chisq_uniform{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    if args.counts_out:
        np.savez_compressed(args.counts_out, **all_counts)
        print(f"wrote {args.counts_out}")

    # Two figures, matching the style of an earlier ad hoc console version the
    # user preferred over a plain overlaid histogram: per-solution-index bar
    # charts (with a uniform-expectation reference line) plus a sorted-
    # descending rank view, which together show not just the summary chi2/z
    # but the actual shape -- concentrated favoritism vs. diffuse overdispersion.
    net_counts, exact_counts = all_counts["net"], all_counts["exact"]
    net_row = next(r for r in rows if r["arm"] == "net")
    exact_row = next(r for r in rows if r["arm"] == "exact")
    idx = np.arange(len(sols))

    # Figure 1: combined net-vs-exact summary. Panel order (by-index left,
    # sorted right) matches the net-only figure below.
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(13, 5))
    ax1a.bar(idx, exact_counts, width=1.0, color="tab:blue")
    ax1a.axhline(exact_row["expected_per_cell"], color="red", linestyle="--",
                 label=f"uniform expectation ({exact_row['expected_per_cell']:.1f})")
    ax1a.set_xlabel("solution index (0-287)")
    ax1a.set_ylabel("count")
    ax1a.set_title("Exact sampler: counts per solution, by index")
    ax1a.legend()

    ax1b.bar(idx, np.sort(net_counts)[::-1], width=1.0, color="tab:red",
             alpha=0.6, label="net")
    ax1b.bar(idx, np.sort(exact_counts)[::-1], width=1.0, color="tab:blue",
             alpha=0.4, label="exact")
    ax1b.axhline(net_row["expected_per_cell"], color="tab:red", linestyle="--", lw=1)
    ax1b.axhline(exact_row["expected_per_cell"], color="tab:blue", linestyle="--", lw=1)
    ax1b.set_xlabel("rank")
    ax1b.set_ylabel("count")
    ax1b.set_title("Sorted counts per solution: net vs exact sampler")
    ax1b.legend()

    fig1.suptitle(f"Net vs exact-sampler distribution over 288 solutions "
                  f"(n={args.n}, tau={args.tau:g})")
    fig1.tight_layout()
    combined_path = out_dir / f"check3_chisq_uniform_combined{suffix}.png"
    fig1.savefig(combined_path, dpi=150)
    print(f"wrote {combined_path}")

    # Figure 2: net-only detail (by index, and sorted descending).
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5))
    ax2a.bar(idx, net_counts, width=1.0, color="tab:blue")
    ax2a.axhline(net_row["expected_per_cell"], color="red", linestyle="--",
                 label=f"uniform expectation ({net_row['expected_per_cell']:.1f})")
    ax2a.set_xlabel("solution index (0-287)")
    ax2a.set_ylabel("count")
    ax2a.set_title("Counts per canonical solution, by index")
    ax2a.legend()

    ax2b.bar(idx, np.sort(net_counts)[::-1], width=1.0, color="tab:blue")
    ax2b.axhline(net_row["expected_per_cell"], color="red", linestyle="--",
                 label=f"uniform expectation ({net_row['expected_per_cell']:.1f})")
    ax2b.set_xlabel("rank")
    ax2b.set_ylabel("count")
    ax2b.set_title("Counts per solution, sorted descending")
    ax2b.legend()

    fig2.suptitle(f"Distribution of net-sampled solutions over the 288 valid sudoku grids "
                  f"(n={args.n}, {net_row['n_valid']} valid, tau={args.tau:g})")
    fig2.tight_layout()
    net_path = out_dir / f"check3_chisq_uniform_net{suffix}.png"
    fig2.savefig(net_path, dpi=150)
    print(f"wrote {net_path}")

    print("\n|z| < 2 is consistent with uniform over the 288.")
    print("exact arm off uniform -> discretisation; net additionally off -> network.")


if __name__ == "__main__":
    main()
