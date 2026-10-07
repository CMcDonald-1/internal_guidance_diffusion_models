"""Sudoku 4x4 check 1: net vs. exact denoiser, across signal level `a` and reward-tilt tau.

Checkpoint : ckpt/sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt
             (plain, non-self-conditioned TiltMLP, width=1024, n_blocks=8)
Dataset    : data/sudoku_reward_v2  (d=33, SudokuRewardAugmented(SudokuMixtureTarget(
             SudokuRowsTarget(row_noise=0.02), p_inject=0.05)))

Modeled on eval/ising_check2_denoiser_ood.py's grid/plot, with one deliberate
difference in how puzzles are drawn (see "Independent draws per (a, tau)
point" below). Nothing sudoku-specific is needed in this script itself. The
"sudoku functions for computing the ground-truth denoiser, under tilt and not
under tilt" are exactly:

    target.natural_mean(h)             -- EXACT E[X|H=h] under pi (tau=0)
    target.tilt_direction(tau)         -- h_extra = tau * e_star, the linear
                                           reward tilt
    target.natural_mean(h + h_extra)   -- EXACT E[X|H=h] under q_tau (tilted)

all three inherited unmodified by SudokuRewardAugmented from
targets.RewardAugmented / EnumerableTarget: even though this target's
`.states` is None (the SudokuMixtureTarget base never materializes an
explicit state list), `natural_mean`/`tilt_direction` are still exact via the
S-vs-C mixture decomposition documented in tilt/sudoku.py -- there is nothing sudoku-specific for
tilt/evaluate.py to special-case; `operator_error(..., mode="ood", tau=...)`
(see its module docstring) works unmodified via `_ground_truth`/`_resolve_h_extra`.

Grid:
    - a in (0, 1): the network's own noise-level input (beta(a) = a/(1-a^2)),
      NOT beta itself -- 20 points strictly between 0 and 1.
    - tau in [0, sqrt(d)]: the reward-tilt strength, h_extra = tau * e_star.
      tau_max = sqrt(33) ~= 5.745 mirrors this checkpoint's own lambda_max=5.0
      training ceiling (noising.lambda_matrix requires 1.0 <= lambda < sqrt(d)),
      the same informal "how far this network was pushed in training"
      reference point used for the Ising check.
    - x ~ q_tau (the reward-tilted law), u = VP-noise(x, a), n=100 draws.
    - ground truth: target.natural_mean(beta(a)*u + tau*e_star)  (EXACT).
    - network: net(v*, a*) after build_query folds in h_extra (mode="ood").
    - SSE per (a, tau): mean over the 100 draws of the per-draw sum-of-
      squared-error across all 33 coordinates -- same "sse" convention as
      tilt/evaluate.py::operator_error.

Independent draws per (a, tau) point
-------------------------------------
The Ising script (eval/ising_check2_denoiser_ood.py) calls
`operator_error(net, target, beta_grid, seed=seed, ...)` ONCE per tau, handing
it the whole beta_grid: internally, `_typical_inputs` reseeds a fresh
`torch.Generator().manual_seed(seed)` on every beta iteration, so all 20 `a`
levels for a given tau end up reusing the SAME underlying 100 puzzles (only
the noise scale changes with a). Here, that's deliberately changed: this
script calls `operator_error` once per SINGLE beta with a distinct seed per
(a, tau) point (`seed = base_seed + i*n_tau + j`), so every one of the
20 x 5 = 100 grid points draws its own independent fresh set of 100 puzzles
from q_tau. `tilt/evaluate.py` itself is untouched -- only how this script
drives it differs.

This is a pure denoiser check (single VP-noise step, no SDE trajectory), so
it does not use the sampler's (T, K, schedule) at all -- those only apply to
full sl_sample runs (see eval/common/sample_config.py), not this script.

Usage:
    python eval/sudoku4x4_check1_denoiser_ood.py
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

CKPT = ROOT / "ckpt" / "sudoku_reward_v2_infill_mode1p25_lambda5_w1024b8_30k.pt"
DATA = ROOT / "data" / "sudoku_reward_v2"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--n-a", type=int, default=20)
    p.add_argument("--n-tau", type=int, default=5)
    p.add_argument("--n", type=int, default=100, help="draws per (a, tau) point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku4x4"))
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
    df.to_csv(out_dir / "check1_denoiser_ood.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    for i, tau in enumerate(tau_grid):
        sub = df[df["tau"] == float(tau)].sort_values("a")
        ax.plot(sub["a"], sub["sse"], marker="o", ms=4,
                color=cmap(i / max(len(tau_grid) - 1, 1)), label=f"tau={tau:.3f}")
    ax.set_xlabel("a (signal level)")
    ax.set_ylabel("SSE (net vs. exact operator)")
    ax.set_title(f"Sudoku 4x4 d={target.d} denoiser error vs. a, by reward-tilt tau "
                 f"(n={args.n}/point, independent draws per point)")
    ax.legend(title="tau")
    fig.tight_layout()
    fig.savefig(out_dir / "check1_denoiser_ood.png", dpi=150)

    print(f"\nsaved: {out_dir / 'check1_denoiser_ood.csv'}")
    print(f"saved: {out_dir / 'check1_denoiser_ood.png'}")


if __name__ == "__main__":
    main()
