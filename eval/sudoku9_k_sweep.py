"""Sudoku 9x9: sweep the Euler-Maruyama step count K to find where reward-tilt
sampling stops improving with more steps, before picking a canonical K for the
9x9 reward-tilt/log-odds and memorization checks.

Checkpoint : ckpt/sudoku9_v1_l17_d512x8_xf_sc_s8264_180k.pt
             (TransformerTilt, self-conditioned, d=325 = 324 core + 1 reward
             coordinate)

Self-conditioning correctness
-------------------------------------------
Past 9x9 sampling in this repo (`scripts/run_sudoku9_reward_tilt_check.py`,
`run_sudoku9_memorization.py`) went through `sl_sample_selfcond` without an
explicit `self_cond_mode`, so whatever numbers came out depended on which side
of the self-conditioning default change (Appendix B.4) the run happened
on: the old default ("carry") fed each Euler step the PREVIOUS step's
self-cond prediction, making the sampled chain non-Markov and not the SL law
of the trained operator at all -- not a discretization error, a different
(wrong) stochastic process. This script goes through
`eval/common/checkpoints.py::load_premier`, which inherits the corrected
`two_pass` default (fresh self-cond estimate at the CURRENT state every step)
with no override -- the same thing the 6x6 scripts already rely on.

Why this problem needs its own K sweep, not the 6x6 one
---------------------------------------------------------
An earlier 9x9 study already found
that the 6x6 K~=1000 plateau does NOT transfer: on a 10.86M-param 9x9
transformer, K=4000->8000 at tau=8 moved full-grid validity 0.8870->0.9025,
individually only ~1.6 SE at n=2000 but consistent across every tau tested and
matched by a Richardson extrapolation putting the converged value at ~0.918
(i.e. K=8000 itself had not converged). That prior check never included
K=1000, and the deltas were too close to their own noise floor to call a
clean "elbow" rather than a suggestive trend -- this sweep exists to nail that
down with a wider K ladder, larger n, and a metric designed to isolate where
the K-sensitivity actually lives.

Two things sharpen the signal here relative to just re-plotting full-grid
validity vs. K:

1. **Score `P(valid | all_rows valid)`, not just raw full-grid validity.**
   The same note found the K-sensitive part of the failure lives entirely in
   column/box coordination: `all_rows` was flat (0.961->0.960) from
   K=4000->8000 while `P(valid|rows)` moved 0.923->0.941. Raw full-grid
   validity mixes in the row channel, which is already near ceiling and
   essentially insensitive to K, diluting the signal with noise unrelated to
   discretization.
2. **tau in {3, 5}, not just 8.** tau=8's exact target is ~100% -- a literal
   ceiling that flattens any curve near it for a reason that has nothing to
   do with convergence. tau=3 (exact ~95.5%) and tau=5 (exact ~99.9%) both
   leave real headroom above whatever the network currently achieves, and
   running both checks whether the K-elbow's location/shape depends on tilt
   strength or is a fixed property of the sampler/checkpoint.

Progress reporting
-------------------
Each (K, tau) point's cost is dominated by n * K network forward passes (2x
that for self-cond two_pass), so elapsed time scales roughly linearly in K for
fixed n. `_ProgressTracker` below tracks a running "seconds per K-unit" rate
from completed points and uses it to print an ETA after every point, rather
than just a raw per-point timer -- useful here specifically because the K
values span a 16x range (1000 to 16000) within one run, so a naive "time
remaining = time so far / fraction done" estimate (by point COUNT) would be
badly wrong early on if the small-K points happen to run first.

--self-cond-mode
----------------------------------------
A same-checkpoint, same-(K,tau) comparison (`eval/sudoku9_selfcond_mode_check.py`)
found the deprecated `carry` mode gives substantially higher actual grid
validity than `two_pass` at K=1000 (e.g. tau=3: 0.869 vs. 0.557), with no
within-batch duplication. Whether that gap grows, shrinks, or holds steady as
K increases is an open question this flag exists to answer -- carry is also
~half the per-step cost of two_pass (1 forward pass vs. 2), so re-running the
whole K ladder under carry costs less than the two_pass sweep already did.
Output filenames get a `_carry` suffix so a carry run never overwrites a
two_pass run in the same `--out-dir`.

Usage:
    python eval/sudoku9_k_sweep.py
    python eval/sudoku9_k_sweep.py --K-list 1000 2000 4000 8000 16000 --taus 3 5 --n 1000
    python eval/sudoku9_k_sweep.py --self-cond-mode carry --taus 3 5 --n 1000
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from datetime import datetime, timedelta
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
from common.reward_check import clopper_pearson_ci, logit  # noqa: E402
from common.sample_config import SCHEDULE, T  # noqa: E402

from tilt.sudoku9 import (  # noqa: E402
    D,
    box_valid_mask,
    col_valid_mask,
    decode_grid_batch,
    is_valid_sudoku9_batch,
    row_valid_mask,
    success_rate_exact,
)

CKPT = ROOT / "ckpt" / "sudoku9_v1_l17_d512x8_xf_sc_s8264_180k.pt"
ROW_NOISE = 0.02
P_INJECT = 0.05


class _ProgressTracker:
    """Prints a per-point timer plus a running ETA for the whole sweep.

    Cost is assumed roughly linear in K (fixed n per point, dominated by
    K network forward passes), so the ETA is computed from a running
    "seconds per K-unit" rate rather than from the fraction of POINTS
    completed -- points at different K cost very different amounts here.
    """

    def __init__(self, points: list[tuple[int, float]]):
        self.points = points
        self.total_weight = sum(K for K, _ in points)
        self.done_weight = 0
        self.t_start = time.time()
        print(f"planned: {len(points)} (K, tau) points, "
              f"K-weighted work = {self.total_weight} K-units\n")

    def report(self, K: int, tau: float, i: int, elapsed_this_point: float) -> None:
        self.done_weight += K
        elapsed_total = time.time() - self.t_start
        rate = elapsed_total / self.done_weight  # seconds per K-unit, running average
        remaining_weight = self.total_weight - self.done_weight
        eta_s = remaining_weight * rate
        pct = 100.0 * self.done_weight / self.total_weight
        finish = datetime.now() + timedelta(seconds=eta_s)
        print(f"    [{i}/{len(self.points)} points, {pct:5.1f}% of K-weighted work]  "
              f"this point={elapsed_this_point:.1f}s  elapsed={elapsed_total / 60:.1f}min  "
              f"ETA={eta_s / 60:.1f}min  (est. finish {finish:%H:%M:%S})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--row-noise", type=float, default=ROW_NOISE)
    p.add_argument("--p-inject", type=float, default=P_INJECT)
    p.add_argument("--taus", type=float, nargs="+", default=[3.0, 5.0])
    p.add_argument("--K-list", type=int, nargs="+", default=[1000, 2000, 4000, 8000, 16000])
    p.add_argument("--n", type=int, default=1000, help="draws per (K, tau) point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku9"))
    p.add_argument("--self-cond-mode", default="two_pass", choices=["two_pass", "carry"],
                   help="'carry' is deprecated (non-Markov drift, see "
                        "Appendix B.4) but empirically found to give "
                        "higher actual grid validity at K=1000 -- see this "
                        "script's module docstring. Output files get a "
                        "'_carry' suffix so they never collide with a "
                        "two_pass run's outputs.")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    suffix = "" if args.self_cond_mode == "two_pass" else f"_{args.self_cond_mode}"

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  device={device}")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- expected the "
                          "premier sudoku9 transformer/self-cond checkpoint")

    d = shim.net.cfg.d
    print(f"d={d}  taus={args.taus}  K_list={args.K_list}  n={args.n}/point  "
          f"T={T}, schedule={SCHEDULE!r}  self_cond_mode={args.self_cond_mode!r}")

    exact_by_tau = {tau: success_rate_exact(args.row_noise, args.p_inject, tau=tau) for tau in args.taus}
    for tau, ex in exact_by_tau.items():
        print(f"  exact success_rate(tau={tau}) = {ex:.4%}")

    # K outer, tau inner: cheapest points run first, which is also what makes
    # the running K-unit rate (see _ProgressTracker) meaningful early on.
    points = [(K, tau) for K in args.K_list for tau in args.taus]
    tracker = _ProgressTracker(points)

    rows = []
    for i, (K, tau) in enumerate(points, start=1):
        t0 = time.time()
        h0 = torch.zeros(d)
        h0[-1] = tau
        with warnings.catch_warnings():
            # "carry is deprecated" is the whole point of running it here --
            # this flag exists BECAUSE of that finding, not despite it.
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            raw = sl(drift_arg, d=d, n=args.n, h=h0, T=T, K=K, schedule=SCHEDULE,
                     seed=args.seed, self_cond_mode=args.self_cond_mode)
        hard = decode(raw, drift_arg, h0)

        core = hard[:, :D]
        n_hits = int((hard[:, -1] > 0).sum())
        model = n_hits / args.n
        p_lo, p_hi = clopper_pearson_ci(n_hits, args.n)

        grid = decode_grid_batch(core)
        rows_ok = row_valid_mask(grid)
        cols_ok = col_valid_mask(grid)
        boxes_ok = box_valid_mask(grid)
        all_rows = rows_ok.all(dim=-1)
        full_valid = is_valid_sudoku9_batch(core)
        assert torch.equal(full_valid, all_rows & cols_ok.all(dim=-1) & boxes_ok.all(dim=-1))

        n_valid = int(full_valid.sum())
        actual_valid = n_valid / args.n
        v_lo, v_hi = clopper_pearson_ci(n_valid, args.n)

        n_all_rows = int(all_rows.sum())
        all_rows_rate = n_all_rows / args.n
        ar_lo, ar_hi = clopper_pearson_ci(n_all_rows, args.n)

        n_valid_given_rows = int((full_valid & all_rows).sum())
        if n_all_rows > 0:
            p_given_rows = n_valid_given_rows / n_all_rows
            pg_lo, pg_hi = clopper_pearson_ci(n_valid_given_rows, n_all_rows)
        else:
            p_given_rows, pg_lo, pg_hi = float("nan"), float("nan"), float("nan")

        exact = exact_by_tau[tau]
        elapsed = time.time() - t0
        row = {
            "K": K, "tau": tau, "self_cond_mode": args.self_cond_mode, "exact": exact,
            "model": model, "model_ci_lo": p_lo, "model_ci_hi": p_hi,
            "logit_model": logit(model, args.n), "logit_exact": logit(exact),
            "actual_valid": actual_valid, "actual_valid_ci_lo": v_lo, "actual_valid_ci_hi": v_hi,
            "all_rows_rate": all_rows_rate, "all_rows_ci_lo": ar_lo, "all_rows_ci_hi": ar_hi,
            "p_valid_given_rows": p_given_rows, "p_given_rows_ci_lo": pg_lo, "p_given_rows_ci_hi": pg_hi,
            "n_all_rows": n_all_rows,
            "n": args.n, "elapsed_s": elapsed,
        }
        rows.append(row)
        print(f"  K={K:6d} tau={tau:4.1f}  model={model:.4f} [{p_lo:.4f},{p_hi:.4f}]  "
              f"actual_valid={actual_valid:.4f} [{v_lo:.4f},{v_hi:.4f}]  "
              f"all_rows={all_rows_rate:.4f}  p(valid|rows)={p_given_rows:.4f} "
              f"(n_rows={n_all_rows})  exact={exact:.4f}")
        tracker.report(K, tau, i, elapsed)

        # Written after EVERY point, not just at the end: this sweep can run
        # for hours (see module docstring's progress-reporting note), so a
        # partial CSV lets progress be checked (e.g. `tail`/reopen in another
        # run) without waiting on the whole run or losing everything to
        # an interruption.
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / f"k_sweep{suffix}.csv", index=False)

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, len(args.taus), figsize=(5.5 * len(args.taus), 9), sharex=True)
    if len(args.taus) == 1:
        axes = axes.reshape(2, 1)
    for col, tau in enumerate(args.taus):
        sub = df[df["tau"] == tau].sort_values("K")

        ax = axes[0, col]
        ax.errorbar(sub["K"], sub["model"],
                    yerr=np.vstack([sub["model"] - sub["model_ci_lo"], sub["model_ci_hi"] - sub["model"]]),
                    fmt="o-", capsize=3, color="tab:orange", label="model P(x*=+1)")
        ax.errorbar(sub["K"], sub["actual_valid"],
                    yerr=np.vstack([sub["actual_valid"] - sub["actual_valid_ci_lo"],
                                     sub["actual_valid_ci_hi"] - sub["actual_valid"]]),
                    fmt="s-", capsize=3, color="tab:green", label="actual grid validity")
        ax.axhline(exact_by_tau[tau], ls="--", color="tab:blue", label="exact (closed form)")
        ax.set_xscale("log")
        ax.set_title(f"tau={tau}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)

        ax2 = axes[1, col]
        ax2.errorbar(sub["K"], sub["all_rows_rate"],
                     yerr=np.vstack([sub["all_rows_rate"] - sub["all_rows_ci_lo"],
                                      sub["all_rows_ci_hi"] - sub["all_rows_rate"]]),
                     fmt="^-", capsize=3, color="tab:gray", label="P(all rows valid)")
        ax2.errorbar(sub["K"], sub["p_valid_given_rows"],
                     yerr=np.vstack([sub["p_valid_given_rows"] - sub["p_given_rows_ci_lo"],
                                      sub["p_given_rows_ci_hi"] - sub["p_valid_given_rows"]]),
                     fmt="d-", capsize=3, color="tab:red", label="P(valid | all rows valid)")
        ax2.set_xscale("log")
        ax2.set_xlabel("K (Euler steps)")
        ax2.set_ylim(-0.02, 1.02)
        ax2.legend(fontsize=8)

    axes[0, 0].set_ylabel("probability")
    axes[1, 0].set_ylabel("probability (decomposition)")
    fig.suptitle(f"Sudoku 9x9 K-sweep  (n={args.n}/point, row_noise={args.row_noise}, "
                 f"p_inject={args.p_inject}, self_cond_mode={args.self_cond_mode!r})")
    fig.tight_layout()
    fig.savefig(out_dir / f"k_sweep{suffix}.png", dpi=150)

    print(f"\nsaved: {out_dir / f'k_sweep{suffix}.csv'}")
    print(f"saved: {out_dir / f'k_sweep{suffix}.png'}")


if __name__ == "__main__":
    main()
