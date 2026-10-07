"""Sudoku 6x6: sweep the Euler-Maruyama step count K to find where reward-tilt
sampling stops improving with more steps, before picking a canonical K for
eval/sudoku6_check2_reward_logodds.py.

Checkpoint : ckpt/sudoku6_v2_l10_xf_sc_s8264_180k.pt (transformer, self-cond)

Why a fresh sweep, and why two metrics
-----------------------------------------
The pre-existing 6x6 "K~=1000-1600 asymptote" doesn't transfer here:
different checkpoint (an earlier flat/sc MLP), different schedule ("late" vs.
this project's "sstar"), a Monte-Carlo bit-distance metric rather than reward
rate, n=200/point with no Richardson extrapolation, and a later partial
retraction once schedule was found to explain part of the earlier "K-independent
residual gap". A 9x9 note separately found transformer/higher-capacity
checkpoints can need K well past 8000. None of this reliably predicts what THIS
checkpoint needs on THIS metric, so K is instead swept directly here.

Two things are tracked at every (K, tau) point, both from the SAME drawn batch
(no extra sampling cost):

  - `model` = P(x* = +1), the network's own predicted reward bit -- exactly
    check2's own metric, scored against `success_rate_exact` via a 95%
    Clopper-Pearson CI (`common/reward_check.py`).
  - `actual_valid` = P(decoded grid is genuinely a valid 6x6 sudoku solution)
    (`tilt.sudoku6.is_valid_sudoku6_batch` on the core 108 bits) -- the network's
    own reward PREDICTION can decouple from what it actually generated as tau
    grows (a "reward-hacking-like" self-consistency failure observed on a
    different 6x6 checkpoint), so `model`
    alone plateauing with K is not sufficient evidence that K is no longer the
    bottleneck: if `actual_valid` keeps climbing with K while `model` has already
    flattened, that means check2's own reported metric is not yet trustworthy at
    that K, not that sampling has converged.

tau grid: {3, 5, 8}, matching the earlier 6x6 K-sweep's own tau choices (for
whatever loose comparability that's worth, given none of its other settings
match).

Usage:
    python eval/sudoku6_k_sweep.py
    python eval/sudoku6_k_sweep.py --K-list 200 500 1000 2000 4000 8000 --taus 3 5 8 --n 500
"""
from __future__ import annotations

import argparse
import sys
import time
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

from tilt.sudoku6 import is_valid_sudoku6_batch, success_rate_exact  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku6_v2_l10_xf_sc_s8264_180k.pt"
ROW_NOISE = 0.02
P_INJECT = 0.05


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--row-noise", type=float, default=ROW_NOISE)
    p.add_argument("--p-inject", type=float, default=P_INJECT)
    p.add_argument("--taus", type=float, nargs="+", default=[3.0, 5.0, 8.0])
    p.add_argument("--K-list", type=int, nargs="+", default=[200, 500, 1000, 2000, 4000, 8000])
    p.add_argument("--n", type=int, default=500, help="draws per (K, tau) point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku6"))
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  device={device}")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- expected the "
                          "premier sudoku6 transformer/self-cond checkpoint")

    d = shim.net.cfg.d
    print(f"d={d}  taus={args.taus}  K_list={args.K_list}  n={args.n}/point  "
          f"T={T}, schedule={SCHEDULE!r}")

    exact_by_tau = {tau: success_rate_exact(args.row_noise, args.p_inject, tau=tau) for tau in args.taus}
    for tau, ex in exact_by_tau.items():
        print(f"  exact success_rate(tau={tau}) = {ex:.4%}")

    rows = []
    for K in args.K_list:
        for tau in args.taus:
            t0 = time.time()
            h0 = torch.zeros(d)
            h0[-1] = tau
            raw = sl(drift_arg, d=d, n=args.n, h=h0, T=T, K=K, schedule=SCHEDULE, seed=args.seed)
            hard = decode(raw, drift_arg, h0)

            n_hits = int((hard[:, -1] > 0).sum())
            model = n_hits / args.n
            p_lo, p_hi = clopper_pearson_ci(n_hits, args.n)

            n_valid = int(is_valid_sudoku6_batch(hard[:, :108]).sum())
            actual_valid = n_valid / args.n
            v_lo, v_hi = clopper_pearson_ci(n_valid, args.n)

            exact = exact_by_tau[tau]
            elapsed = time.time() - t0
            row = {
                "K": K, "tau": tau, "exact": exact,
                "model": model, "model_ci_lo": p_lo, "model_ci_hi": p_hi,
                "logit_model": logit(model, args.n), "logit_exact": logit(exact),
                "actual_valid": actual_valid, "actual_valid_ci_lo": v_lo, "actual_valid_ci_hi": v_hi,
                "n": args.n, "elapsed_s": elapsed,
            }
            rows.append(row)
            print(f"  K={K:5d} tau={tau:4.1f}  model={model:.4f} [{p_lo:.4f},{p_hi:.4f}]  "
                  f"actual_valid={actual_valid:.4f} [{v_lo:.4f},{v_hi:.4f}]  "
                  f"exact={exact:.4f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "k_sweep.csv", index=False)

    fig, axes = plt.subplots(1, len(args.taus), figsize=(5 * len(args.taus), 5), sharex=True)
    if len(args.taus) == 1:
        axes = [axes]
    for ax, tau in zip(axes, args.taus):
        sub = df[df["tau"] == tau].sort_values("K")
        ax.errorbar(sub["K"], sub["model"],
                    yerr=np.vstack([sub["model"] - sub["model_ci_lo"], sub["model_ci_hi"] - sub["model"]]),
                    fmt="o-", capsize=3, color="tab:orange", label="model P(x*=+1)")
        ax.errorbar(sub["K"], sub["actual_valid"],
                    yerr=np.vstack([sub["actual_valid"] - sub["actual_valid_ci_lo"],
                                     sub["actual_valid_ci_hi"] - sub["actual_valid"]]),
                    fmt="s-", capsize=3, color="tab:green", label="actual grid validity")
        ax.axhline(exact_by_tau[tau], ls="--", color="tab:blue", label="exact (closed form)")
        ax.set_xscale("log")
        ax.set_xlabel("K (Euler steps)")
        ax.set_title(f"tau={tau}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("probability")
    fig.suptitle(f"Sudoku 6x6 K-sweep  (n={args.n}/point, row_noise={args.row_noise}, "
                 f"p_inject={args.p_inject})")
    fig.tight_layout()
    fig.savefig(out_dir / "k_sweep.png", dpi=150)

    print(f"\nsaved: {out_dir / 'k_sweep.csv'}")
    print(f"saved: {out_dir / 'k_sweep.png'}")


if __name__ == "__main__":
    main()
