"""MNIST: denoiser norm ||G||/sqrt(d) along the sampling path, two_pass vs carry.

A confidence trace. The operator's output is tanh-bounded, so every coordinate
of G lies in (-1, 1) and

    ||G|| / sqrt(d)  in  [0, 1]

is a natural scalar "how resolved is the prediction right now": ~0 when the
network is still at its mean-field/undecided state, 1 when every coordinate has
committed to +-1. Plotting it along a sampled trajectory shows WHERE in the
path the sample actually gets decided, and comparing the two self-conditioning
modes shows whether they commit at different points -- the dynamic counterpart
to check 1's static class-rate comparison.

X axis is s*, not t
-----------------------------------------------------------------------------
`s* = a_of_beta(||h + X_t|| / sqrt(d))` (tilt/noising.py::normalize_natural) is
the scalar the network is actually handed as its signal level, and the `sstar`
schedule this project samples with equalises steps in exactly it. Plotting
against wall-clock t instead would compress the interesting region: s* runs
~0.05 -> ~0.98 over the path while t runs 0 -> 20, and the drift varies along
s*, not t. Both are recorded in the CSV.

The curve is the sampler's OWN drift, not a stand-in
-----------------------------------------------------------------------------
The recorded states are replayed through each mode's own rule, reproducing the
exact drift sequence `sl_sample_selfcond` used:

  two_pass   x_pred recomputed fresh at the current state each step, so the
             drift is net.natural(h+X, x_pred=net.natural(h+X, None)) -- which
             is just `natural_self_cond` at that state.
  carry      x_pred is the PREVIOUS step's drift output, so the replay must
             run sequentially from X_0 = 0 and thread x_pred through; the
             value depends on the whole history, not on X_t alone. That
             non-Markov dependence is precisely why "carry" is
             deprecated (Appendix B.4), and it is why carry cannot be evaluated
             pointwise at an arbitrary state.

For carry a SECOND curve is also recorded, `deployed`: the two-pass estimate
`natural_self_cond(h + X_t)` at the same visited states. The gap between
carry's own drift and the deployed estimate at the states it visits separates
"this mode goes to different places" from "this mode believes different things
at the same place". two_pass's two curves are identical by construction, so
only one is drawn for it.

Memory note: `return_path=True` keeps every intermediate state, so the path
costs n*K*d*4 bytes (~400 MB at n=16, K=1000, d=6282). Raise --n with that in
mind; the batch only needs to be large enough to average a concentrated
scalar, not to estimate a rate.

Usage:
    python eval/mnist_denoiser_norm.py --cls 7
    python eval/mnist_denoiser_norm.py --cls 7 --tau 0 --n 24
"""
from __future__ import annotations

import argparse
import json
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
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.mnist import D, reward_tilt  # noqa: E402
from tilt.noising import a_of_beta  # noqa: E402

from mnist_sample_figure import tau_for_rate  # noqa: E402

CKPT = ROOT / "ckpt" / "mnist_v1_l24_d256x4_xf_sc_s8264_30k.pt"
DATA = ROOT / "data" / "mnist_v1"


@torch.no_grad()
def trace_mode(shim, sl, hb, h0, mode: str, n: int, K: int, seed: int) -> pd.DataFrame:
    """Sample under `mode`, then replay the path to recover its own drift norms."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        _, path = sl(shim, d=D, n=n, h=h0, T=T, K=K, schedule=SCHEDULE,
                     seed=seed, self_cond_mode=mode, return_path=True)

    # states visited BEFORE each step: X_0 = 0, then every recorded state but the last
    states = [torch.zeros(n, D)] + [xk for _, xk in path[:-1]]
    times = [0.0] + [t for t, _ in path[:-1]]
    rt_d = math.sqrt(D)

    x_pred = torch.zeros(n, D)
    rows = []
    for k, (t, xk) in enumerate(zip(times, states)):
        q = hb + xk
        if mode == "two_pass":
            drift = shim.natural(q, x_pred=shim.natural(q, x_pred=None))
            deployed = drift                      # identical by construction
        else:
            drift = shim.natural(q, x_pred=x_pred)
            x_pred = drift                        # non-Markov carry
            deployed = shim.natural_self_cond(q)
        s_star = a_of_beta(torch.linalg.norm(q, dim=1) / rt_d)
        gn = torch.linalg.norm(drift, dim=1) / rt_d
        dn = torch.linalg.norm(deployed, dim=1) / rt_d
        rows.append({
            "mode": mode, "step": k, "t": t,
            "s_star": float(s_star.mean()),
            "g_norm": float(gn.mean()), "g_lo": float(gn.quantile(0.1)),
            "g_hi": float(gn.quantile(0.9)),
            "deployed_norm": float(dn.mean()),
            "x_norm": float((torch.linalg.norm(q, dim=1) / rt_d).mean()),
        })
    del path, states
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--cls", type=int, default=7)
    p.add_argument("--tau", type=float, default=None,
                   help="default is this class's own tau_95")
    p.add_argument("--n", type=int, default=16, help="paths per mode (see memory note)")
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "mnist"))
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = args.K if args.K is not None else K_DEFAULTS["mnist"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    priors = np.asarray(json.loads((Path(args.data) / "meta.json").read_text())
                        ["class_counts"], dtype=np.float64)
    priors = priors / priors.sum()
    tau = args.tau if args.tau is not None else tau_for_rate(float(priors[args.cls]))

    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned")
    h0 = reward_tilt(tau, args.cls)
    hb = h0.unsqueeze(0)

    print(f"class {args.cls}  tau={tau:.4f}  n={args.n} paths/mode  K={K}  "
          f"schedule={SCHEDULE!r}  device={device}")
    frames = []
    for mode in ("two_pass", "carry"):
        df_m = trace_mode(shim, sl, hb, h0, mode, args.n, K, args.seed)
        frames.append(df_m)
        # where does the trace first cross 0.9 of full saturation?
        cross = df_m[df_m.g_norm >= 0.9]
        where = (f"s*={cross.s_star.iloc[0]:.3f} (step {int(cross.step.iloc[0])}, "
                 f"t={cross.t.iloc[0]:.2f})") if len(cross) else "never"
        print(f"  {mode:9s} final ||G||/sqrt(d)={df_m.g_norm.iloc[-1]:.4f}   "
              f"first reaches 0.9: {where}")

    df = pd.concat(frames, ignore_index=True)
    suffix = f"_cls{args.cls}_tau{tau:.3f}" + (f"_{args.tag}" if args.tag else "")
    df.to_csv(out_dir / f"denoiser_norm{suffix}.csv", index=False)

    colors = {"two_pass": "tab:blue", "carry": "tab:orange"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for mode, sub in df.groupby("mode"):
        ax.plot(sub.s_star, sub.g_norm, "-", color=colors[mode], label=mode)
        ax.fill_between(sub.s_star, sub.g_lo, sub.g_hi, color=colors[mode], alpha=0.15)
        if mode == "carry":
            ax.plot(sub.s_star, sub.deployed_norm, "--", color=colors[mode], lw=1.2,
                    label="carry, two-pass estimate at its own states")
        ax2.plot(sub.t, sub.g_norm, "-", color=colors[mode], label=mode)
    for a_ in (ax, ax2):
        a_.axhline(1.0, ls=":", color="0.4", lw=1.0, label="fully committed (|G|=1)")
        a_.set_ylabel(r"$\|G\|/\sqrt{d}$   (confidence proxy)")
        a_.set_ylim(-0.02, 1.05)
        a_.legend(fontsize=8, loc="lower right")
    ax.set_xlabel(r"signal level $s^*$  (the network's own input scalar)")
    ax.set_title("vs signal level (shaded: 10-90% across paths)")
    ax2.set_xlabel("t (SL time)")
    ax2.set_title("vs wall-clock t, for comparison")

    fig.suptitle(f"MNIST denoiser confidence along the sampling path  "
                 f"(class {args.cls}, tau={tau:.3f}, n={args.n}/mode, K={K})")
    fig.tight_layout()
    fig.savefig(out_dir / f"denoiser_norm{suffix}.png", dpi=150)
    print(f"\nsaved: {out_dir / f'denoiser_norm{suffix}.csv'}")
    print(f"saved: {out_dir / f'denoiser_norm{suffix}.png'}")


if __name__ == "__main__":
    main()
