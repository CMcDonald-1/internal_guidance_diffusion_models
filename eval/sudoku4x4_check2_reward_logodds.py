"""Sudoku 4x4 check 2: reward-tilt log-odds vs. tau, net sampler AND exact-operator sampler.

Checkpoint : ckpt/sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt
Dataset    : data/sudoku_reward_v2  (d=33)

For a linear reward tilt pi_tau(x) propto pi(x) * exp(tau * x_star), the EXACT
marginal success rate is a closed form regardless of any correlation between
x_star and the rest of x:

    p_tau = p0 * e^tau / (p0 * e^tau + (1 - p0) * e^-tau)
    logit(p_tau) = logit(p0) + 2*tau                          <- exactly linear

p0 = exact_reward_rate_from_natural_mean(target, tau=0.0) is the EXACT
untilted rate -- exact here even though target.states is None (see
tilt/sudoku.py: SudokuRewardAugmented's
natural_mean is exact via the S-vs-C mixture decomposition regardless of
whether the base density materializes an explicit state list).

Unlike eval/ising_check3_reward_logodds.py, this script draws from TWO
different samplers at each tau, both run via the identical SL SDE
(tilt/sampler.py::sl_sample/decode_drift) with h = tau * e_star, differing
only in which drift function is used:

  - network sampler:        drift_fn = net.natural   (what check1 also probes,
                             but end-to-end through the sampler rather than a
                             single denoising step)
  - exact-operator sampler:  drift_fn = target.natural_mean   -- the SAME SDE,
                             discretization, and Brownian path shape, but
                             driven by the EXACT ground-truth operator instead
                             of the network. This isolates Euler-Maruyama
                             discretization error (present in both) from
                             network-attributable error: if the exact sampler
                             already misses the exact closed-form rate by some
                             margin, that margin is not the network's fault.
                             Same construction as scripts/run_reward_sample_quality.py.

Sampling config: T=20.0, K=K_DEFAULTS["sudoku4x4"]=1000, schedule="sstar"
(see eval/common/sample_config.py).

Usage:
    python eval/sudoku4x4_check2_reward_logodds.py
"""
from __future__ import annotations

import argparse
import math
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
from common.reward_check import (  # noqa: E402
    build_tau_grid,
    clip_logit_yerr,
    clopper_pearson_ci,
    exact_reward_rate_from_natural_mean,
    finite_axis_range,
    logit,
    logit_ci,
)
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.data import load_dataset  # noqa: E402
from tilt.sampler import decode_drift, sl_sample  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt"
DATA = ROOT / "data" / "sudoku_reward_v2"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--target-p", type=float, default=0.98,
                   help="exact success rate defining tau_98, the boundary between the "
                        "'before'/'after' grid segments (see build_tau_grid)")
    p.add_argument("--n-before", type=int, default=5, help="points on [0, tau_98]")
    p.add_argument("--n-after", type=int, default=5, help="points on (tau_98, tau_max]")
    p.add_argument("--tau-max", type=float, default=None,
                   help="override floor(sqrt(d)) as the grid's upper end")
    p.add_argument("--n", type=int, default=2000, help="draws per tau, per sampler")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku4x4"))
    p.add_argument("--K", type=int, default=None,
                   help="override K_DEFAULTS['sudoku4x4'] for a one-off comparison run "
                        "(e.g. a K-sweep check); output filenames get a _K<K> suffix "
                        "whenever this differs from the default so the canonical "
                        "check2_reward_logodds.{csv,png} from the standard K is preserved")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K_default = K_DEFAULTS["sudoku4x4"]
    K = args.K if args.K is not None else K_default
    suffix = "" if K == K_default else f"_K{K}"

    _x_train, _frozen, target, _meta = load_dataset(args.data)
    if target is None:
        raise SystemExit(f"{args.data} has no target.json -- exact ground truth unavailable")

    _shim, info, sl, decode, _needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  device={device}")

    p0 = exact_reward_rate_from_natural_mean(target, 0.0)
    print(f"p0 (exact, tau=0) = {p0:.4%}")

    tau_max = args.tau_max if args.tau_max is not None else math.floor(math.sqrt(target.d))
    taus, tau_98 = build_tau_grid(p0, tau_max, args.target_p, args.n_before, args.n_after)
    if tau_98 is not None:
        print(f"tau_98 (exact rate={args.target_p:.0%}) = {tau_98:.4f}  tau_max = {tau_max}")
    else:
        print(f"tau_98 >= tau_max={tau_max} -- falling back to a plain linspace(0, tau_max, "
              f"{args.n_before + args.n_after})")
    print(f"tau grid ({len(taus)} pts): {taus}")
    print(f"sampling config: T={T}, K={K}, schedule={SCHEDULE!r}, n={args.n}/tau/sampler")

    rows = []
    for tau in taus:
        tau = float(tau)
        h0 = target.tilt_direction(tau)
        exact_closed_form = exact_reward_rate_from_natural_mean(target, tau)

        # network sampler
        raw_net = sl(drift_arg, d=target.d, n=args.n, h=h0, T=T, K=K,
                     schedule=SCHEDULE, seed=args.seed)
        hard_net = decode(raw_net, drift_arg, h0)
        n_hits_model = int((hard_net[:, -1] > 0).sum())
        model = n_hits_model / args.n

        # exact-operator sampler: identical SDE/decoder, drift = target.natural_mean
        raw_exact = sl_sample(target.natural_mean, d=target.d, n=args.n, h=h0, T=T, K=K,
                              schedule=SCHEDULE, seed=args.seed)
        hard_exact = decode_drift(raw_exact, target.natural_mean, h0)
        n_hits_exact_sampler = int((hard_exact[:, -1] > 0).sum())
        exact_sampler = n_hits_exact_sampler / args.n

        model_lo, model_hi = clopper_pearson_ci(n_hits_model, args.n)
        exact_sampler_lo, exact_sampler_hi = clopper_pearson_ci(n_hits_exact_sampler, args.n)

        logit_exact = logit(exact_closed_form)
        logit_model = logit(model, args.n)
        logit_exact_sampler = logit(exact_sampler, args.n)
        # TRUE (possibly +-inf) log-odds of each CI's own bounds -- see
        # logit_ci's docstring for why these must NOT use the continuity-
        # corrected logit() reserved for point estimates.
        logit_model_lo, logit_model_hi = logit_ci(model_lo, model_hi)
        logit_exact_sampler_lo, logit_exact_sampler_hi = logit_ci(exact_sampler_lo, exact_sampler_hi)
        logit_theory = logit(p0) + 2.0 * tau  # closed-form check
        row = {
            "tau": tau, "p0": p0,
            "exact_closed_form": exact_closed_form,
            "model": model,
            "exact_sampler": exact_sampler,
            "model_ci_lo": model_lo, "model_ci_hi": model_hi,
            "exact_sampler_ci_lo": exact_sampler_lo, "exact_sampler_ci_hi": exact_sampler_hi,
            "logit_exact": logit_exact, "logit_model": logit_model,
            "logit_exact_sampler": logit_exact_sampler,
            "logit_model_ci_lo": logit_model_lo, "logit_model_ci_hi": logit_model_hi,
            "logit_exact_sampler_ci_lo": logit_exact_sampler_lo,
            "logit_exact_sampler_ci_hi": logit_exact_sampler_hi,
            "logit_theory": logit_theory,
            "n": args.n,
        }
        rows.append(row)
        print(f"  tau={tau:6.3f}  exact_closed_form={exact_closed_form:.4f} (logit {logit_exact:+.3f}, "
              f"theory {logit_theory:+.3f})  model={model:.4f} [{model_lo:.4f}, {model_hi:.4f}] "
              f"(logit {logit_model:+.3f})  "
              f"exact_sampler={exact_sampler:.4f} [{exact_sampler_lo:.4f}, {exact_sampler_hi:.4f}] "
              f"(logit {logit_exact_sampler:+.3f})")

    df = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"check2_reward_logodds{suffix}.csv", index=False)

    # Exact closed-form curve costs nothing extra at extra points, no sampling needed.
    tau_dense = np.linspace(0.0, float(df["tau"].max()), 400)
    logit_exact_dense = logit(p0) + 2.0 * tau_dense
    exact_dense = 1.0 / (1.0 + np.exp(-logit_exact_dense))

    yerr_prob_model = np.vstack([df["model"] - df["model_ci_lo"], df["model_ci_hi"] - df["model"]])
    yerr_prob_exact_sampler = np.vstack(
        [df["exact_sampler"] - df["exact_sampler_ci_lo"], df["exact_sampler_ci_hi"] - df["exact_sampler"]])

    fig, (ax_logit, ax_prob) = plt.subplots(1, 2, figsize=(13, 5))

    ax_logit.plot(tau_dense, logit_exact_dense, "-", color="tab:blue", label="exact (ground truth)")
    # logit_*_ci_hi/lo can be literal +-inf (e.g. once a sampler hits 0/n or
    # n/n out of n draws -- see logit_ci's docstring) -- clip the DRAWN bar to
    # a range fixed from every FINITE value that will actually appear on this
    # axis (theory curve + both point-estimate series), not just the theory
    # curve's own autoscale -- a point estimate itself can land outside that
    # curve's own range (see finite_axis_range's docstring; this is exactly
    # what broke here originally, at tau=5 where the exact-operator sampler's
    # own continuity-corrected point estimate exceeded the theory curve's max).
    y_lo_lim, y_hi_lim = finite_axis_range(logit_exact_dense, df["logit_model"], df["logit_exact_sampler"])
    yerr_logit_model = clip_logit_yerr(df["logit_model"], df["logit_model_ci_lo"], df["logit_model_ci_hi"],
                                        y_max=y_hi_lim, y_min=y_lo_lim)
    yerr_logit_exact_sampler = clip_logit_yerr(
        df["logit_exact_sampler"], df["logit_exact_sampler_ci_lo"], df["logit_exact_sampler_ci_hi"],
        y_max=y_hi_lim, y_min=y_lo_lim)
    ax_logit.errorbar(df["tau"], df["logit_model"], yerr=yerr_logit_model, fmt="o", capsize=3,
                       color="tab:orange", label="network sampler (95% CP CI)")
    ax_logit.errorbar(df["tau"], df["logit_exact_sampler"], yerr=yerr_logit_exact_sampler, fmt="s",
                       capsize=3, color="tab:green", label="exact-operator sampler (95% CP CI)")
    if tau_98 is not None:
        ax_logit.axvline(tau_98, ls="--", color="gray", label=f"tau_98={tau_98:.3f}")
    ax_logit.set_ylim(y_lo_lim, y_hi_lim)
    ax_logit.set_xlabel("tau (reward-tilt strength)")
    ax_logit.set_ylabel("log-odds  logit(P(x* = +1))")
    ax_logit.set_title("log-odds scale")
    ax_logit.legend()

    ax_prob.plot(tau_dense, exact_dense, "-", color="tab:blue", label="exact (ground truth)")
    ax_prob.errorbar(df["tau"], df["model"], yerr=yerr_prob_model, fmt="o", capsize=3,
                      color="tab:orange", label="network sampler (95% CP CI)")
    ax_prob.errorbar(df["tau"], df["exact_sampler"], yerr=yerr_prob_exact_sampler, fmt="s", capsize=3,
                      color="tab:green", label="exact-operator sampler (95% CP CI)")
    if tau_98 is not None:
        ax_prob.axvline(tau_98, ls="--", color="gray", label=f"tau_98={tau_98:.3f}")
    ax_prob.set_xlabel("tau (reward-tilt strength)")
    ax_prob.set_ylabel("P(x* = +1)")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.set_title("probability scale (sigmoid)")
    ax_prob.legend()

    fig.suptitle(f"Sudoku 4x4 d={target.d} reward-tilt vs. tau  "
                 f"(p0={p0:.2%}, n={args.n}/tau, K={K})")
    fig.tight_layout()
    fig.savefig(out_dir / f"check2_reward_logodds{suffix}.png", dpi=150)

    print(f"\nsaved: {out_dir / f'check2_reward_logodds{suffix}.csv'}")
    print(f"saved: {out_dir / f'check2_reward_logodds{suffix}.png'}")


if __name__ == "__main__":
    main()
