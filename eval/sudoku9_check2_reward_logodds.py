"""Sudoku 9x9 check 2: reward-tilt log-odds/probability vs. tau.

Checkpoint : ckpt/sudoku9_v1_l17_d512x8_xf_sc_s8264_180k.pt
             (TransformerTilt, self-conditioned, d=325 = 324 core + 1 reward
             coordinate)
Ground truth: tilt.sudoku9.success_rate_exact(row_noise, p_inject, tau) --
CLOSED FORM (no enumeration, no Monte Carlo), exactly mirroring the 6x6
version of this check (tilt.sudoku6.success_rate_exact). row_noise=0.02,
p_inject=0.05 match data/sudoku9_v1's recipe (data/sudoku9_v1/meta.json)
-- not read from a target.json, since none exists for this problem.

self_cond_mode is now a first-class argument, not an implicit default
-------------------------------------------------------------------------
In this repo's own
`eval/sudoku9_selfcond_mode_check.py` / `eval/sudoku9_k_sweep.py`), the
deprecated non-Markov "carry" self-conditioning mode was found to give
SUBSTANTIALLY higher actual grid validity than the corrected "two_pass"
default on this exact checkpoint (e.g. tau=3, K=1000: actual_valid 0.860 vs.
0.557), with no within-batch sample duplication, and the gap does not close
with more Euler steps. So unlike every other problem's check2 in this folder,
this script is meant to be run under BOTH modes and compared, not just under
the "correct" one -- carry is empirically the better sampler for this
checkpoint even though its path law is not the SL law of any fixed operator
(a fact worth stating plainly: this script does not resolve that theoretical
tension, it just measures which mode a report should actually use).

Two curves are tracked (from the SAME drawn batch, no extra sampling cost),
matching `eval/sudoku9_k_sweep.py`'s convention rather than the 6x6 check2's
single-curve one:
  - `model`        = P(x* = +1), the network's own predicted reward bit.
  - `actual_valid` = P(decoded grid is genuinely a valid 9x9 sudoku solution).
Earlier experiments observed `model` and
`actual_valid` decoupling under reward tilt (a "reward-hacking-like"
self-consistency failure) -- for 9x9 specifically this decoupling turned out
to be much larger for two_pass than for carry, so both curves are worth
seeing side by side here, not just the model's own self-report.

Choosing the tau grid (matches the 6x6 script's convention exactly)
-------------------------------------------------------------------------------
A plain `linspace(0, sqrt(d), n_tau)` over-samples the tail once p0 is small
and d is large. Instead: `tau_98` = the tau at which the EXACT rate first
reaches 98% (closed-form log-odds inversion, since `success_rate_exact` obeys
`logit(p_tau) = logit(p0) + 2*tau` exactly), then 5 points on `[0, tau_98]`
(the informative rising region) plus 5 more on `(tau_98, tau_max]` (the tail,
`tau_max = floor(sqrt(d))`) via `common/reward_check.py::build_tau_grid` --
the exact same helper the 6x6 script uses, just with `tilt.sudoku9`'s own p0.

K = K_DEFAULTS["sudoku9"] = 1000 (see `common/sample_config.py`'s
note for why -- the K-sweep (eval/sudoku9_k_sweep.py) found neither self_cond_mode
benefits meaningfully from more than K=1000 on this checkpoint, superseding
the older, carry-contaminated "9x9 may need K>>8000" claim).

Usage:
    python eval/sudoku9_check2_reward_logodds.py --self-cond-mode carry
    python eval/sudoku9_check2_reward_logodds.py --self-cond-mode two_pass
"""
from __future__ import annotations

import argparse
import math
import sys
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
from common.reward_check import (  # noqa: E402
    build_tau_grid,
    clip_logit_yerr,
    clopper_pearson_ci,
    finite_axis_range,
    logit,
    logit_ci,
)
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.sudoku9 import D, decode_grid_batch, is_valid_sudoku9_batch, success_rate_exact  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku9_v1_l17_d512x8_xf_sc_s8264_180k.pt"
ROW_NOISE = 0.02
P_INJECT = 0.05


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--self-cond-mode", required=True, choices=["two_pass", "carry"],
                   help="which self-conditioning mode to sample with -- see this "
                        "script's module docstring for why both are worth running")
    p.add_argument("--row-noise", type=float, default=ROW_NOISE)
    p.add_argument("--p-inject", type=float, default=P_INJECT)
    p.add_argument("--target-p", type=float, default=0.98,
                   help="exact success rate defining tau_98, the boundary between the "
                        "'before'/'after' grid segments")
    p.add_argument("--n-before", type=int, default=5, help="points on [0, tau_98]")
    p.add_argument("--n-after", type=int, default=5, help="points on (tau_98, tau_max]")
    p.add_argument("--tau-max", type=float, default=None,
                   help="override floor(sqrt(d)) as the grid's upper end")
    p.add_argument("--tau-list", type=float, nargs="+", default=None,
                   help="explicit tau values instead of the tau_98-anchored grid")
    p.add_argument("--n", type=int, default=2000, help="draws per tau")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku9"))
    p.add_argument("--K", type=int, default=None,
                   help="override K_DEFAULTS['sudoku9']; output filenames get a "
                        "_K<K> suffix whenever this differs from the default")
    p.add_argument("--tag", default=None, help="extra output-filename suffix")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K_default = K_DEFAULTS["sudoku9"]
    K = args.K if args.K is not None else K_default
    suffix = (f"_{args.self_cond_mode}"
              + ("" if K == K_default else f"_K{K}")
              + (f"_{args.tag}" if args.tag else ""))

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  "
          f"self_cond_mode={args.self_cond_mode!r}  device={device}")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- expected the "
                          "premier sudoku9 transformer/self-cond checkpoint")

    d = shim.net.cfg.d
    p0 = success_rate_exact(args.row_noise, args.p_inject, tau=0.0)
    print(f"d={d}  p0 (exact, tau=0) = {p0:.4%}  (row_noise={args.row_noise}, p_inject={args.p_inject})")

    if args.tau_list is not None:
        taus = np.array(args.tau_list, dtype=float)
        tau_98 = None
    else:
        tau_max = args.tau_max if args.tau_max is not None else math.floor(math.sqrt(d))
        taus, tau_98 = build_tau_grid(p0, tau_max, target_p=args.target_p,
                                       n_before=args.n_before, n_after=args.n_after)
        print(f"tau_98 (exact rate={args.target_p:.0%}) = {tau_98:.4f}  tau_max = {tau_max}")
    print(f"tau grid ({len(taus)} pts): {taus}")
    print(f"sampling config: T={T}, K={K}, schedule={SCHEDULE!r}, n={args.n}/tau, "
          f"self_cond_mode={args.self_cond_mode!r}")

    rows = []
    for i, tau in enumerate(taus, start=1):
        tau = float(tau)
        h0 = torch.zeros(d)
        h0[-1] = tau
        exact = success_rate_exact(args.row_noise, args.p_inject, tau=tau)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            raw = sl(drift_arg, d=d, n=args.n, h=h0, T=T, K=K, schedule=SCHEDULE,
                     seed=args.seed, self_cond_mode=args.self_cond_mode)
        hard = decode(raw, drift_arg, h0)

        core = hard[:, :D]
        n_hits = int((hard[:, -1] > 0).sum())
        model = n_hits / args.n
        p_lo, p_hi = clopper_pearson_ci(n_hits, args.n)

        n_valid = int(is_valid_sudoku9_batch(core).sum())
        actual_valid = n_valid / args.n
        v_lo, v_hi = clopper_pearson_ci(n_valid, args.n)

        logit_exact = logit(exact)
        logit_model = logit(model, args.n)
        logit_lo, logit_hi = logit_ci(p_lo, p_hi)
        logit_actual = logit(actual_valid, args.n)
        logit_actual_lo, logit_actual_hi = logit_ci(v_lo, v_hi)
        row = {
            "tau": tau, "p0": p0, "K": K, "self_cond_mode": args.self_cond_mode,
            "exact": exact, "model": model, "actual_valid": actual_valid,
            "n_hits": n_hits, "ci_lo": p_lo, "ci_hi": p_hi,
            "actual_ci_lo": v_lo, "actual_ci_hi": v_hi,
            "logit_exact": logit_exact, "logit_model": logit_model,
            "logit_ci_lo": logit_lo, "logit_ci_hi": logit_hi,
            "logit_actual": logit_actual,
            "logit_actual_ci_lo": logit_actual_lo, "logit_actual_ci_hi": logit_actual_hi,
            "n": args.n,
        }
        rows.append(row)
        print(f"  [{i}/{len(taus)}] tau={tau:6.3f}  exact={exact:.4f} (logit {logit_exact:+.3f})  "
              f"model={model:.4f} [{p_lo:.4f},{p_hi:.4f}] (logit {logit_model:+.3f})  "
              f"actual_valid={actual_valid:.4f} [{v_lo:.4f},{v_hi:.4f}]")

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / f"check2_reward_logodds{suffix}.csv", index=False)

    df = pd.DataFrame(rows)

    tau_dense = np.linspace(0.0, float(df["tau"].max()), 400)
    logit_exact_dense = np.array(
        [logit(success_rate_exact(args.row_noise, args.p_inject, tau=float(t))) for t in tau_dense])
    exact_dense = 1.0 / (1.0 + np.exp(-logit_exact_dense))

    yerr_prob_model = np.vstack([df["model"] - df["ci_lo"], df["ci_hi"] - df["model"]])
    yerr_prob_actual = np.vstack([df["actual_valid"] - df["actual_ci_lo"],
                                   df["actual_ci_hi"] - df["actual_valid"]])

    fig, (ax_logit, ax_prob) = plt.subplots(1, 2, figsize=(13, 5))

    ax_logit.plot(tau_dense, logit_exact_dense, "-", color="tab:blue", label="exact (closed form)")
    y_lo_lim, y_hi_lim = finite_axis_range(logit_exact_dense, df["logit_model"], df["logit_actual"])
    yerr_logit_model = clip_logit_yerr(df["logit_model"], df["logit_ci_lo"], df["logit_ci_hi"],
                                        y_max=y_hi_lim, y_min=y_lo_lim)
    yerr_logit_actual = clip_logit_yerr(df["logit_actual"], df["logit_actual_ci_lo"],
                                         df["logit_actual_ci_hi"], y_max=y_hi_lim, y_min=y_lo_lim)
    ax_logit.errorbar(df["tau"], df["logit_model"], yerr=yerr_logit_model, fmt="o", capsize=3,
                       color="tab:orange", label="model P(x*=+1) (95% CP CI)")
    ax_logit.errorbar(df["tau"], df["logit_actual"], yerr=yerr_logit_actual, fmt="s", capsize=3,
                       color="tab:green", label="actual grid validity (95% CP CI)")
    if tau_98 is not None:
        ax_logit.axvline(tau_98, ls="--", color="gray", label=f"tau_98={tau_98:.3f}")
    ax_logit.set_ylim(y_lo_lim, y_hi_lim)
    ax_logit.set_xlabel("tau (reward-tilt strength)")
    ax_logit.set_ylabel("log-odds  logit(P)")
    ax_logit.set_title("log-odds scale")
    ax_logit.legend(fontsize=8)

    ax_prob.plot(tau_dense, exact_dense, "-", color="tab:blue", label="exact (closed form)")
    ax_prob.errorbar(df["tau"], df["model"], yerr=yerr_prob_model, fmt="o", capsize=3,
                      color="tab:orange", label="model P(x*=+1) (95% CP CI)")
    ax_prob.errorbar(df["tau"], df["actual_valid"], yerr=yerr_prob_actual, fmt="s", capsize=3,
                      color="tab:green", label="actual grid validity (95% CP CI)")
    if tau_98 is not None:
        ax_prob.axvline(tau_98, ls="--", color="gray", label=f"tau_98={tau_98:.3f}")
    ax_prob.set_xlabel("tau (reward-tilt strength)")
    ax_prob.set_ylabel("P")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.set_title("probability scale (sigmoid)")
    ax_prob.legend(fontsize=8)

    fig.suptitle(f"Sudoku 9x9 d={d} reward-tilt vs. tau  (p0={p0:.2%}, n={args.n}/tau, K={K}, "
                 f"self_cond_mode={args.self_cond_mode!r})")
    fig.tight_layout()
    fig.savefig(out_dir / f"check2_reward_logodds{suffix}.png", dpi=150)

    print(f"\nsaved: {out_dir / f'check2_reward_logodds{suffix}.csv'}")
    print(f"saved: {out_dir / f'check2_reward_logodds{suffix}.png'}")


if __name__ == "__main__":
    main()
