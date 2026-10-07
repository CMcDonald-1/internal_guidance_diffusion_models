"""Ising check 3: reward-tilt log-odds vs. tau.

Checkpoint : ckpt/ising12_exact10_lambda3p5_30k_jac.pt
Dataset    : data/run_exact10  (d=13, IsingTarget(d=12) + reward exact:10)

For a linear reward tilt pi_tau(x) propto pi(x) * exp(tau * x_star), the
EXACT marginal success rate is a closed form regardless of any correlation
between x_star and the rest of x (this is exact whenever the reweighting
depends only on x_star's own value, not an approximation specific to Ising):

    p_tau = p0 * e^tau / (p0 * e^tau + (1 - p0) * e^-tau)
    logit(p_tau) = logit(p0) + 2*tau                          <- exactly linear

p0 = exact_reward_rate_from_natural_mean(target, tau=0.0) is the EXACT
untilted rate (matches data/run_exact10/meta.json's success_rate=0.04412,
i.e. p0~=4.41%, consistent with the training set's own empirical rate of
4.468%). This check compares that exact log-odds line
against the network's ACTUAL sampled log-odds at each tau (drawing samples
via the trained sampler at h = tau * e_star, decoding, and measuring the
empirical hit rate) -- the network has no reason to reproduce the line
exactly at large tau since it was never trained on tau itself as a label.

Sampling config: T=20.0, K=K_DEFAULTS["ising"]=1000, schedule="sstar" (see
eval/common/sample_config.py).

Usage:
    python eval/ising_check3_reward_logodds.py
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

CKPT = ROOT / "ckpt" / "ising12_exact10_lambda3p5_30k_jac.pt"
DATA = ROOT / "data" / "run_exact10"


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
    p.add_argument("--n", type=int, default=2000, help="draws per tau")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "ising"))
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = K_DEFAULTS["ising"]

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
    print(f"sampling config: T={T}, K={K}, schedule={SCHEDULE!r}, n={args.n}/tau")

    rows = []
    for tau in taus:
        tau = float(tau)
        h0 = target.tilt_direction(tau)
        exact = exact_reward_rate_from_natural_mean(target, tau)
        raw = sl(drift_arg, d=target.d, n=args.n, h=h0, T=T, K=K,
                  schedule=SCHEDULE, seed=args.seed)
        hard = decode(raw, drift_arg, h0)
        n_hits = int((hard[:, -1] > 0).sum())
        model = n_hits / args.n

        p_lo, p_hi = clopper_pearson_ci(n_hits, args.n)
        logit_exact = logit(exact)
        logit_model = logit(model, args.n)
        # TRUE (possibly +-inf) log-odds of the CI bounds themselves -- see
        # logit_ci's docstring for why these must NOT be computed via the
        # continuity-corrected logit() used for point estimates above.
        logit_lo, logit_hi = logit_ci(p_lo, p_hi)
        logit_theory = logit(p0) + 2.0 * tau  # closed-form check
        row = {
            "tau": tau, "p0": p0,
            "exact": exact, "model": model,
            "n_hits": n_hits, "ci_lo": p_lo, "ci_hi": p_hi,
            "logit_exact": logit_exact, "logit_model": logit_model,
            "logit_ci_lo": logit_lo, "logit_ci_hi": logit_hi,
            "logit_theory": logit_theory,
            "n": args.n,
        }
        rows.append(row)
        print(f"  tau={tau:6.3f}  exact={exact:.4f} (logit {logit_exact:+.3f}, "
              f"theory {logit_theory:+.3f})  model={model:.4f} [{p_lo:.4f}, {p_hi:.4f}] "
              f"(logit {logit_model:+.3f})")

    df = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "check3_reward_logodds.csv", index=False)

    # Exact is closed-form (logit(p0) + 2*tau), so a smooth curve costs nothing
    # extra -- no sampling needed at these extra points.
    tau_dense = np.linspace(0.0, float(df["tau"].max()), 400)
    logit_exact_dense = logit(p0) + 2.0 * tau_dense
    exact_dense = 1.0 / (1.0 + np.exp(-logit_exact_dense))

    yerr_prob = np.vstack([df["model"] - df["ci_lo"], df["ci_hi"] - df["model"]])

    fig, (ax_logit, ax_prob) = plt.subplots(1, 2, figsize=(13, 5))

    ax_logit.plot(tau_dense, logit_exact_dense, "-", color="tab:blue", label="exact (ground truth)")
    # Some logit_ci_hi (or, less commonly, logit_ci_lo) entries can be literal
    # +-inf (see logit_ci's docstring) -- clip the DRAWN bar to a range fixed
    # from every FINITE value that will actually appear on this axis (theory
    # curve + point estimates), not just from the theory curve's own
    # autoscale -- a point estimate can itself land outside that curve's own
    # range (see finite_axis_range's docstring).
    y_lo_lim, y_hi_lim = finite_axis_range(logit_exact_dense, df["logit_model"])
    yerr_logit = clip_logit_yerr(df["logit_model"], df["logit_ci_lo"], df["logit_ci_hi"],
                                  y_max=y_hi_lim, y_min=y_lo_lim)
    ax_logit.errorbar(df["tau"], df["logit_model"], yerr=yerr_logit, fmt="o", capsize=3,
                       color="tab:orange", label="network sampler (95% Clopper-Pearson CI)")
    if tau_98 is not None:
        ax_logit.axvline(tau_98, ls="--", color="gray", label=f"tau_98={tau_98:.3f}")
    ax_logit.set_ylim(y_lo_lim, y_hi_lim)
    ax_logit.set_xlabel("tau (reward-tilt strength)")
    ax_logit.set_ylabel("log-odds  logit(P(x* = +1))")
    ax_logit.set_title("log-odds scale")
    ax_logit.legend()

    ax_prob.plot(tau_dense, exact_dense, "-", color="tab:blue", label="exact (ground truth)")
    ax_prob.errorbar(df["tau"], df["model"], yerr=yerr_prob, fmt="o", capsize=3,
                      color="tab:orange", label="network sampler (95% Clopper-Pearson CI)")
    if tau_98 is not None:
        ax_prob.axvline(tau_98, ls="--", color="gray", label=f"tau_98={tau_98:.3f}")
    ax_prob.set_xlabel("tau (reward-tilt strength)")
    ax_prob.set_ylabel("P(x* = +1)")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.set_title("probability scale (sigmoid)")
    ax_prob.legend()

    fig.suptitle(f"Ising d={target.d} reward-tilt vs. tau  (p0={p0:.2%}, n={args.n}/tau)")
    fig.tight_layout()
    fig.savefig(out_dir / "check3_reward_logodds.png", dpi=150)

    print(f"\nsaved: {out_dir / 'check3_reward_logodds.csv'}")
    print(f"saved: {out_dir / 'check3_reward_logodds.png'}")


if __name__ == "__main__":
    main()
