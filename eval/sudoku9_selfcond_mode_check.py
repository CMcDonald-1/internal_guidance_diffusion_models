"""Sudoku 9x9: does the deprecated self-cond 'carry' mode actually sample
BETTER than the corrected 'two_pass' default, in practice?

Motivation
-------------------------
Earlier 9x9 sample-quality numbers from this project were generated
under `self_cond_mode="carry"` (Appendix B.4) -- and switching to the corrected `two_pass`
mode on the SAME checkpoint collapsed apparent tau=3 grid validity (e.g.
d256x4: 62-69% carry -> 14% two_pass). `carry` is mathematically wrong: it
feeds each Euler step the PREVIOUS step's self-cond prediction, so the
sampled chain is not Markov in X and its path law is not the SL law of any
fixed operator G(h) -- an argument from first principles, not an empirical
one.

But "is it the SL law of the trained operator" and "does it produce more
useful samples in practice" are different questions, and it's a fair ask to
check the second empirically rather than dismiss carry on theoretical grounds
alone: the carry-mode note itself observes carry "tends to sharpen samples
(entropy down)", which is exactly the kind of thing that COULD make
reward-tilted sampling concentrate harder on valid/high-reward grids, at the
cost of being a different (and diversity-costing) stochastic process. This
script runs the SAME (checkpoint, tau, K, n, seed) point under both modes and
reports, side by side:

  - model P(x*=+1) vs. exact success_rate_exact -- does carry track log-odds
    at least as well as two_pass?
  - actual grid validity (is_valid_sudoku9_batch) -- does carry produce more
    VALID grids, not just a higher predicted-reward-bit rate?
  - row/column decomposition (all_rows, P(valid|rows))
  - within-batch DISTINCT-grid rate among the valid samples -- a cheap,
    immediate proxy for "carry sharpens entropy down" manifesting as mode
    collapse (few repeated solutions rather than many distinct ones). This is
    NOT a full memorization-against-training-data check (that's a separate,
    larger check -- scripts/run_sudoku9_memorization.py / a future
    eval/sudoku9_check*_memorization.py -- left for later if this comparison
    looks promising enough to pursue further); it only asks whether the
    sampler is repeating itself WITHIN one batch.

Both modes are available directly through `eval/common/checkpoints.py::
load_premier`'s returned `sl` (== `tilt.sampler_selfcond.sl_sample_selfcond`
itself, not a partial), so `self_cond_mode` is simply passed as an extra
kwarg at the call site -- no need to bypass the shared loader.

Usage:
    python eval/sudoku9_selfcond_mode_check.py
    python eval/sudoku9_selfcond_mode_check.py --taus 3 5 --K 1000 --n 1000
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
MODES = ("two_pass", "carry")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--row-noise", type=float, default=ROW_NOISE)
    p.add_argument("--p-inject", type=float, default=P_INJECT)
    p.add_argument("--taus", type=float, nargs="+", default=[3.0, 5.0])
    p.add_argument("--K", type=int, default=1000)
    p.add_argument("--n", type=int, default=1000, help="draws per (mode, tau) point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku9"))
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  device={device}")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- expected the "
                          "premier sudoku9 transformer/self-cond checkpoint")

    d = shim.net.cfg.d
    print(f"d={d}  taus={args.taus}  K={args.K}  n={args.n}/point  "
          f"T={T}, schedule={SCHEDULE!r}  modes={MODES}")

    exact_by_tau = {tau: success_rate_exact(args.row_noise, args.p_inject, tau=tau) for tau in args.taus}
    for tau, ex in exact_by_tau.items():
        print(f"  exact success_rate(tau={tau}) = {ex:.4%}")

    points = [(mode, tau) for tau in args.taus for mode in MODES]
    rows = []
    with warnings.catch_warnings():
        # The "carry is deprecated" warning is the whole point of this script
        # (we KNOW, that's what's being checked) -- silence it here rather
        # than let it print twice per point.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        for i, (mode, tau) in enumerate(points, start=1):
            t0 = time.time()
            h0 = torch.zeros(d)
            h0[-1] = tau
            raw = sl(drift_arg, d=d, n=args.n, h=h0, T=T, K=args.K, schedule=SCHEDULE,
                     seed=args.seed, self_cond_mode=mode)
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

            n_valid_given_rows = int((full_valid & all_rows).sum())
            p_given_rows = n_valid_given_rows / n_all_rows if n_all_rows else float("nan")

            # Within-batch distinct-grid rate among the VALID samples: a
            # collapsed sampler repeats the same few solutions, which this
            # catches even though "distinct" alone can't rule out subtler
            # near-duplication (this is the weakest of the duplication tests --
            # treat this number
            # as a first pass, not a full mode-collapse check).
            if n_valid > 0:
                n_distinct_valid = int(torch.unique(core[full_valid], dim=0).shape[0])
                distinct_rate = n_distinct_valid / n_valid
            else:
                n_distinct_valid, distinct_rate = 0, float("nan")

            exact = exact_by_tau[tau]
            elapsed = time.time() - t0
            row = {
                "mode": mode, "tau": tau, "K": args.K, "exact": exact,
                "model": model, "model_ci_lo": p_lo, "model_ci_hi": p_hi,
                "logit_model": logit(model, args.n), "logit_exact": logit(exact),
                "actual_valid": actual_valid, "actual_valid_ci_lo": v_lo, "actual_valid_ci_hi": v_hi,
                "all_rows_rate": all_rows_rate, "p_valid_given_rows": p_given_rows,
                "n_all_rows": n_all_rows, "n_valid": n_valid,
                "n_distinct_valid": n_distinct_valid, "distinct_rate": distinct_rate,
                "n": args.n, "elapsed_s": elapsed,
            }
            rows.append(row)
            print(f"  [{i}/{len(points)}] mode={mode:9s} tau={tau:4.1f}  "
                  f"model={model:.4f} [{p_lo:.4f},{p_hi:.4f}]  "
                  f"actual_valid={actual_valid:.4f} [{v_lo:.4f},{v_hi:.4f}]  "
                  f"all_rows={all_rows_rate:.4f}  p(valid|rows)={p_given_rows:.4f}  "
                  f"distinct_valid={distinct_rate:.4f} ({n_distinct_valid}/{n_valid})  "
                  f"exact={exact:.4f}  ({elapsed:.1f}s)")

            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "selfcond_mode_check.csv", index=False)

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)

    fig, axes = plt.subplots(1, len(args.taus), figsize=(6 * len(args.taus), 5))
    if len(args.taus) == 1:
        axes = [axes]
    colors = {"two_pass": "tab:orange", "carry": "tab:purple"}
    for ax, tau in zip(axes, args.taus):
        sub = df[df["tau"] == tau]
        x = np.arange(len(MODES))
        actual = [sub[sub["mode"] == m]["actual_valid"].iloc[0] for m in MODES]
        distinct = [sub[sub["mode"] == m]["distinct_rate"].iloc[0] for m in MODES]
        model_p = [sub[sub["mode"] == m]["model"].iloc[0] for m in MODES]
        width = 0.25
        ax.bar(x - width, actual, width, label="actual grid validity", color="tab:green")
        ax.bar(x, model_p, width, label="model P(x*=+1)", color="tab:orange")
        ax.bar(x + width, distinct, width, label="distinct rate | valid", color="tab:blue")
        ax.axhline(exact_by_tau[tau], ls="--", color="tab:red", label="exact (closed form)")
        ax.set_xticks(x)
        ax.set_xticklabels(MODES)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"tau={tau}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("probability / rate")
    fig.suptitle(f"Sudoku 9x9: self_cond_mode comparison  (K={args.K}, n={args.n}/point)")
    fig.tight_layout()
    fig.savefig(out_dir / "selfcond_mode_check.png", dpi=150)

    print(f"\nsaved: {out_dir / 'selfcond_mode_check.csv'}")
    print(f"saved: {out_dir / 'selfcond_mode_check.png'}")


if __name__ == "__main__":
    main()
