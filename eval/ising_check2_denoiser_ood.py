"""Ising check 2: net vs. exact denoiser, across signal level `a` and reward-tilt tau.

Checkpoint : ckpt/ising12_exact10_lambda3p5_30k_jac.pt
Dataset    : data/run_exact10  (d=13, IsingTarget(d=12) + reward exact:10)

For each point on a 20 (a) x 5 (tau) grid:
    - a in (0, 1): the network's own noise-level input (beta(a) = a/(1-a^2)),
      NOT beta itself -- 20 points strictly between 0 and 1 (endpoints
      excluded: beta(0)=0 is fine but beta(1)=inf isn't).
    - tau in [0, sqrt(d)]: the reward-tilt strength, h_extra = tau * e_star.
      tau_max = sqrt(13) ~= 3.606 matches this project's own lambda-focusing
      ceiling (noising.lambda_matrix requires 1.0 <= lambda < sqrt(d)); this
      checkpoint's lambda_max=3.5 was chosen just under that ceiling, so
      sqrt(d) is the natural "how far this network was pushed in training"
      reference point (informal proxy -- tau itself is never a training
      label, only lambda-focusing is).
    - x ~ q_tau (the reward-tilted law), u = VP-noise(x, a), n=100 draws.
    - ground truth: target.natural_mean(beta(a)*u + tau*e_star)  (EXACT,
      via enumeration over the 8192 states).
    - network: net(v*, a*) after build_query folds in h_extra (mode="ood").
    - SSE per (a, tau): mean over the 100 draws of the per-draw sum-of-
      squared-error across all 13 coordinates -- same "sse" convention as
      tilt/evaluate.py::operator_error.

Reuses tilt/evaluate.py::operator_error directly (mode="ood") rather than
reimplementing the ground-truth/query logic -- operator_error already takes
`beta` internally and reports `a` back in each row, so we just invert our
own `a` grid to `beta` via beta_of_a before calling it, one call per tau.

Usage:
    python eval/ising_check2_denoiser_ood.py
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

from tilt.data import load_dataset  # noqa: E402
from tilt.evaluate import operator_error  # noqa: E402
from tilt.load_ckpt import load_tilt_checkpoint  # noqa: E402
from tilt.noising import beta_of_a  # noqa: E402

CKPT = ROOT / "ckpt" / "ising12_exact10_lambda3p5_30k_jac.pt"
DATA = ROOT / "data" / "run_exact10"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--n-a", type=int, default=20)
    p.add_argument("--n-tau", type=int, default=5)
    p.add_argument("--n", type=int, default=100, help="draws per (a, tau) point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "ising"))
    args = p.parse_args()

    _x_train, _frozen, target, _meta = load_dataset(args.data)
    if target is None:
        raise SystemExit(f"{args.data} has no target.json -- exact ground truth unavailable")
    if not hasattr(target, "tilt_direction"):
        raise SystemExit("target has no tilt_direction() -- needs a RewardAugmented target")

    net, info = load_tilt_checkpoint(args.ckpt, map_location="cpu")
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})")

    tau_max = math.sqrt(target.d)
    a_grid = np.linspace(0.0, 1.0, args.n_a + 2)[1:-1]
    tau_grid = np.linspace(0.0, tau_max, args.n_tau)
    beta_grid = beta_of_a(torch.tensor(a_grid)).tolist()

    print(f"d={target.d}  tau_max=sqrt(d)={tau_max:.4f}")
    print(f"a grid ({args.n_a} pts): {a_grid[0]:.4f} .. {a_grid[-1]:.4f}")
    print(f"tau grid ({args.n_tau} pts): {tau_grid}")

    all_rows = []
    for j, tau in enumerate(tau_grid):
        for i, b in enumerate(beta_grid):
            point_seed = args.seed + i * args.n_tau + j
            rows = operator_error(net, target, [b], n=args.n, seed=point_seed,
                                   mode="ood", tau=float(tau))
            row = rows[0]
            row["tau"] = float(tau)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)[["a", "beta", "tau", "sse", "rmse", "max", "bayes_floor"]]
    print(df.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "check2_denoiser_ood.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    for i, tau in enumerate(tau_grid):
        sub = df[df["tau"] == float(tau)].sort_values("a")
        ax.plot(sub["a"], sub["sse"], marker="o", ms=4,
                color=cmap(i / max(len(tau_grid) - 1, 1)), label=f"tau={tau:.3f}")
    ax.set_xlabel("a (signal level)")
    ax.set_ylabel("SSE (net vs. exact operator)")
    ax.set_title(f"Ising d={target.d} denoiser error vs. a, by reward-tilt tau "
                 f"(n={args.n}/point)")
    ax.legend(title="tau")
    fig.tight_layout()
    fig.savefig(out_dir / "check2_denoiser_ood.png", dpi=150)

    print(f"\nsaved: {out_dir / 'check2_denoiser_ood.csv'}")
    print(f"saved: {out_dir / 'check2_denoiser_ood.png'}")


if __name__ == "__main__":
    main()
