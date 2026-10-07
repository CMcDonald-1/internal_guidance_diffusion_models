"""Sudoku 6x6 check 2: reward-tilt log-odds/probability vs. tau.

Checkpoint : ckpt/sudoku6_v2_l10_xf_sc_s8264_180k.pt
             (TransformerTilt, self-conditioned, d=109 = 108 core + 1 reward
             coordinate)
Ground truth: tilt.sudoku6.success_rate_exact(row_noise, p_inject, tau) --
CLOSED FORM (no enumeration, no Monte Carlo; see that function's docstring),
covering the FULL reward coordinate (unlike check1, which is exact only for
the 108 core coordinates at tau=0). row_noise=0.02, p_inject=0.05 match
data/sudoku6_v2's recipe (data/sudoku6_v2/meta.json) -- not read
from a target.json, since none exists for this problem.

Unlike Ising/Sudoku 4x4's check2, there is only ONE sampled curve here (the
network sampler), not also an "exact-operator sampler": those two problems
have a full exact JOINT operator (target.natural_mean) that can drive a
second, network-free SDE to isolate discretization error from network error.
6x6 has no such object -- only the row marginal (used by check1, and only at
tau=0/force_star_zero) -- so there is no exact drift function to sample from
here at tau != 0. This check therefore only tells you net-sampler-vs-exact-
closed-form-rate, not how much of any gap is discretization vs. network --
which is exactly why K itself needs to be checked directly against THIS
metric for THIS checkpoint (see "Choosing K" below) rather than assumed from
an unrelated checkpoint/metric.

Choosing K
-----------------------------------
The pre-existing 6x6 "K~=1000-1600 asymptote" was established on
a DIFFERENT checkpoint (an earlier flat/sc MLP, not this premier transformer),
under `schedule="late"` (this project's now-standard is `sstar`), using a
Monte-Carlo bit-distance-to-nearest-valid-solution metric at n=200/point with
no Richardson extrapolation -- and was later partially walked back once
`sstar`/`early` were found to reach at K=1000 what `late` didn't. A 9x9 note
separately found transformer/higher-capacity checkpoints can need K well past
8000 at comparable dimensionality, but that sweep was run under `sstar`
already, so it reflects a genuine dimension-driven need, not a schedule
artifact -- unlike the old 6x6 finding.

`eval/sudoku6_k_sweep.py` (K in {250,500,750,1000,1500,2000,4000,8000},
tau in {3,5,8}, n=500, run under this project's `sstar`) was used to check
THIS checkpoint directly rather than trust either precedent: it found NO
detectable K-dependence anywhere in that range, for either the model's own
predicted reward rate or actual decoded grid validity (see
eval/results/sudoku6/k_sweep.csv/.png) -- consistent with the old 6x6 K need
having been substantially a `late`-schedule artifact that `sstar` already
resolves, at least for this checkpoint. **K=1000 was chosen for
`K_DEFAULTS["sudoku6"]`** to match the Ising/Sudoku4x4 convention, since
nothing in the tested range showed any benefit from going higher.

That sweep also surfaced a persistent, K-independent gap between the
network's predicted reward bit and actual grid validity (model P(x*=+1)
tracks `success_rate_exact` closely; actual validity sits ~5-10pp lower at
every K) -- a real self-consistency/reward-hacking-like issue, not something
this check (which only scores the predicted reward bit, same as Ising/
Sudoku 4x4's check2) will surface on its own; keep that gap in mind when
reading this script's own P(x*=+1)-vs-exact comparison as "the whole story."

Choosing the tau grid
-----------------------------------------------
A plain `linspace(0, sqrt(d), n_tau)` (the Ising/Sudoku4x4 convention) puts
most points in the tail: for this checkpoint's p0=5%, `success_rate_exact`
already exceeds 99.8% by tau=4.5 out of a sqrt(109)=10.44 max, so 4 of 8
default-grid points landed past the region this check can actually resolve
(the n=2000 log-odds ceiling -- once the network
hits 0 misses out of n draws, its measured log-odds clamps at a fixed value
set by the continuity correction, not a real measurement). Instead:

  tau_98  = the tau at which the EXACT rate first reaches 98% (via the
            closed-form log-odds inversion tau = (logit(0.98) - logit(p0))/2,
            since success_rate_exact obeys logit(p_tau) = logit(p0) + 2*tau
            exactly)
  tau_max = floor(sqrt(d))   -- same ceiling as before, just now the END of
            the grid rather than its natural scale

Grid = 5 evenly-spaced points on [0, tau_98] (the informative rising region)
plus 5 more evenly-spaced points on (tau_98, tau_max] (the tail, where this
check's resolution is limited by n but is still worth sampling to see how
badly log-odds is bounded there) -- 10 points total. Overridable via
`--target-p`/`--n-before`/`--n-after`/`--tau-max`, or bypassed entirely with
an explicit `--tau-list`.

Self-conditioning: proper use in a sampling run
------------------------------------------------------------------------
Self-conditioning is a NETWORK ARCHITECTURE choice, not a mechanism for a
sampler to reuse a *previous Euler step's* output: doing the latter feeds the
drift at time t a function of X_{t-dt}, not just X_t, so the sampled chain is
not Markov in X and its path law is not the SL law of any fixed operator --
exactly the "carry" issue discussed in Appendix B.4 of the paper. `tilt/sampler_selfcond.py::sl_sample_selfcond`'s DEFAULT
(`self_cond_mode="two_pass"`) already does the correct thing: at every step it
recomputes a fresh first-pass estimate `x_pred = net.natural(h+X_t, None)` at
the CURRENT state, then feeds that back into the SAME network call to get the
step's drift (`net.natural(h+X_t, x_pred)`) -- i.e. self-conditioning refines
the network's own current-step estimate, never a past step's. This script
passes no `self_cond_mode` override, so it inherits that Markov-correct
default automatically (both here and via `eval/common/checkpoints.py::
load_premier`'s `sl`/`decode` dispatch).

Usage:
    python eval/sudoku6_check2_reward_logodds.py
    python eval/sudoku6_check2_reward_logodds.py --K 2000 --n-before 2 --n-after 2 --n 500  # K-sweep point
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
    clip_logit_yerr,
    clopper_pearson_ci,
    finite_axis_range,
    logit,
    logit_ci,
)
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.sudoku6 import success_rate_exact  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku6_v2_l10_xf_sc_s8264_180k.pt"
ROW_NOISE = 0.02
P_INJECT = 0.05  # matches data/sudoku6_v2's recipe; not read from a dataset -- no target.json.


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--row-noise", type=float, default=ROW_NOISE)
    p.add_argument("--p-inject", type=float, default=P_INJECT)
    p.add_argument("--target-p", type=float, default=0.98,
                   help="exact success rate defining tau_98, the boundary between the "
                        "'before'/'after' grid segments (see 'Choosing the tau grid' above)")
    p.add_argument("--n-before", type=int, default=5, help="points on [0, tau_98]")
    p.add_argument("--n-after", type=int, default=5, help="points on (tau_98, tau_max]")
    p.add_argument("--tau-max", type=float, default=None,
                   help="override floor(sqrt(d)) as the grid's upper end")
    p.add_argument("--tau-list", type=float, nargs="+", default=None,
                   help="explicit tau values instead of the tau_98-anchored grid (for "
                        "K-sweep points)")
    p.add_argument("--n", type=int, default=2000, help="draws per tau")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku6"))
    p.add_argument("--K", type=int, default=None,
                   help="override K_DEFAULTS['sudoku6'] (see 'Choosing K' in this script's "
                        "module docstring for how that default was chosen); output "
                        "filenames get a _K<K> suffix whenever this differs from the default")
    p.add_argument("--tag", default=None,
                   help="extra output-filename suffix (e.g. for a K-sweep point at a "
                        "non-default n/tau grid) so it doesn't collide with a canonical run "
                        "at the same K")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K_default = K_DEFAULTS["sudoku6"]
    K = args.K if args.K is not None else K_default
    suffix = ("" if K == K_default else f"_K{K}") + (f"_{args.tag}" if args.tag else "")

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  device={device}")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- expected the "
                          "premier sudoku6 transformer/self-cond checkpoint")

    d = shim.net.cfg.d
    p0 = success_rate_exact(args.row_noise, args.p_inject, tau=0.0)
    print(f"d={d}  p0 (exact, tau=0) = {p0:.4%}  (row_noise={args.row_noise}, p_inject={args.p_inject})")

    if args.tau_list is not None:
        taus = np.array(args.tau_list, dtype=float)
        tau_98 = None
    else:
        tau_max = args.tau_max if args.tau_max is not None else math.floor(math.sqrt(d))
        tau_98 = (logit(args.target_p) - logit(p0)) / 2.0
        before = np.linspace(0.0, tau_98, args.n_before)
        after = np.linspace(tau_98, tau_max, args.n_after + 1)[1:]
        taus = np.concatenate([before, after])
        print(f"tau_98 (exact rate={args.target_p:.0%}) = {tau_98:.4f}  tau_max = {tau_max}")
    print(f"tau grid ({len(taus)} pts): {taus}")
    print(f"sampling config: T={T}, K={K}, schedule={SCHEDULE!r}, n={args.n}/tau")

    rows = []
    for tau in taus:
        tau = float(tau)
        h0 = torch.zeros(d)
        h0[-1] = tau
        exact = success_rate_exact(args.row_noise, args.p_inject, tau=tau)

        raw = sl(drift_arg, d=d, n=args.n, h=h0, T=T, K=K, schedule=SCHEDULE, seed=args.seed)
        hard = decode(raw, drift_arg, h0)
        n_hits = int((hard[:, -1] > 0).sum())
        model = n_hits / args.n

        p_lo, p_hi = clopper_pearson_ci(n_hits, args.n)
        logit_exact = logit(exact)
        logit_model = logit(model, args.n)
        # TRUE (possibly +-inf) log-odds of the CI bounds -- see logit_ci's
        # docstring for why these must NOT use the continuity-corrected
        # logit() reserved for point estimates. This matters a lot here:
        # every tau past the n=2000 saturation point (n_hits==n) gets a
        # genuinely infinite logit_hi, not a fake finite ceiling.
        logit_lo, logit_hi = logit_ci(p_lo, p_hi)
        row = {
            "tau": tau, "p0": p0, "K": K,
            "exact": exact, "model": model,
            "n_hits": n_hits, "ci_lo": p_lo, "ci_hi": p_hi,
            "logit_exact": logit_exact, "logit_model": logit_model,
            "logit_ci_lo": logit_lo, "logit_ci_hi": logit_hi,
            "n": args.n,
        }
        rows.append(row)
        print(f"  tau={tau:6.3f}  exact={exact:.4f} (logit {logit_exact:+.3f})  "
              f"model={model:.4f} [{p_lo:.4f}, {p_hi:.4f}] (logit {logit_model:+.3f})")

    df = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"check2_reward_logodds{suffix}.csv", index=False)

    tau_dense = np.linspace(0.0, float(df["tau"].max()), 400)
    logit_exact_dense = np.array(
        [logit(success_rate_exact(args.row_noise, args.p_inject, tau=float(t))) for t in tau_dense])
    exact_dense = 1.0 / (1.0 + np.exp(-logit_exact_dense))

    yerr_prob = np.vstack([df["model"] - df["ci_lo"], df["ci_hi"] - df["model"]])

    fig, (ax_logit, ax_prob) = plt.subplots(1, 2, figsize=(13, 5))

    ax_logit.plot(tau_dense, logit_exact_dense, "-", color="tab:blue", label="exact (closed form)")
    # logit_ci_hi can be literal +inf once n_hits==n (see logit_ci's docstring
    # and the module docstring's n=2000-saturation discussion) -- clip the
    # DRAWN bar to a range fixed from every FINITE value that will actually
    # appear on this axis (theory curve + point estimates), not just the
    # theory curve's own autoscale, which a point estimate can itself exceed
    # (see finite_axis_range's docstring -- this broke the Sudoku4x4 version
    # of this script when read via ax.get_ylim() instead).
    y_lo_lim, y_hi_lim = finite_axis_range(logit_exact_dense, df["logit_model"])
    yerr_logit = clip_logit_yerr(df["logit_model"], df["logit_ci_lo"], df["logit_ci_hi"],
                                  y_max=y_hi_lim, y_min=y_lo_lim)
    ax_logit.errorbar(df["tau"], df["logit_model"], yerr=yerr_logit, fmt="o", capsize=3,
                       color="tab:orange", label="network sampler (95% CP CI)")
    if tau_98 is not None:
        ax_logit.axvline(tau_98, ls="--", color="gray", label=f"tau_98={tau_98:.3f}")
    ax_logit.set_ylim(y_lo_lim, y_hi_lim)
    ax_logit.set_xlabel("tau (reward-tilt strength)")
    ax_logit.set_ylabel("log-odds  logit(P(x* = +1))")
    ax_logit.set_title("log-odds scale")
    ax_logit.legend()

    ax_prob.plot(tau_dense, exact_dense, "-", color="tab:blue", label="exact (closed form)")
    ax_prob.errorbar(df["tau"], df["model"], yerr=yerr_prob, fmt="o", capsize=3,
                      color="tab:orange", label="network sampler (95% CP CI)")
    if tau_98 is not None:
        ax_prob.axvline(tau_98, ls="--", color="gray", label=f"tau_98={tau_98:.3f}")
    ax_prob.set_xlabel("tau (reward-tilt strength)")
    ax_prob.set_ylabel("P(x* = +1)")
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.set_title("probability scale (sigmoid)")
    ax_prob.legend()

    fig.suptitle(f"Sudoku 6x6 d={d} reward-tilt vs. tau  (p0={p0:.2%}, n={args.n}/tau, K={K})")
    fig.tight_layout()
    fig.savefig(out_dir / f"check2_reward_logodds{suffix}.png", dpi=150)

    print(f"\nsaved: {out_dir / f'check2_reward_logodds{suffix}.csv'}")
    print(f"saved: {out_dir / f'check2_reward_logodds{suffix}.png'}")


if __name__ == "__main__":
    main()
