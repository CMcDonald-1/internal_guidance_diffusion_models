"""Sudoku 6x6 check 1: net vs. exact ROW-MARGINAL denoiser, across signal level `a`, tau=0 only.

Checkpoint : ckpt/sudoku6_v2_l10_xf_sc_s8264_180k.pt
             (TransformerTilt, self-conditioned, d=109 = 108 core + 1 reward
             coordinate, d_model=256, n_layers=4)
Dataset    : data/sudoku6_v2 was used to train it (row_noise=0.02, p_inject=0.05,
             see data/sudoku6_v2/meta.json); NOT loaded here -- this
             check needs only `row_noise` as a number, not the dataset itself
             (see "No target.json" below).

Why this check looks different from Ising / Sudoku 4x4's check1
-----------------------------------------------------------------
Those two problems have a `target.json` (an `EnumerableTarget`/`RewardAugmented`)
whose `natural_mean` is EXACT over the full joint (x_core, x_star), so check1
there sweeps both signal level `a` AND reward-tilt tau. 6x6 sudoku has no such
object -- `Sudoku6Target` (tilt/sudoku6.py) is sample-only, and column/box
constraints break the row-independence trick that would otherwise make an
exact joint tractable.

What IS exact at this scale is the ROW marginal: `tilt.sudoku6.row_marginal_mean`
gives E[x_row | h_row] in closed form under `sample_weak_rows`'s per-row
mixture density (uniform permutation w.p. 1-row_noise, i.i.d. random cells
w.p. row_noise), and `exact_core_mean` applies that independently to each of
the 6 rows' 18-bit slice of a (N, 108) core natural parameter. This is exact
for the 108 CORE coordinates only -- there is still no closed-form denoiser
for the reward coordinate x_star once tilted, and `exact_core_mean` does not
itself account for whole-solution injection (p_inject), which shifts the true
core marginal slightly (see `exact_row_operator_error`'s docstring).

This exact ground truth is only usable at tau=0 with the network's own
reward-coordinate INPUT forced to exactly 0 (`force_star_zero=True`), not at
an arbitrary reward tilt: under VP noising, u_star | x_star=+1 and
u_star | x_star=-1 have EQUAL likelihood at u_star=0 for any signal level a,
so observing u_star=0 carries zero information about x_star (hence zero
information about x_core beyond what h_core already gives) --
E[x_core | h_core, h_star=0] = E[x_core | h_core] exactly. Feeding the network
its own real sampled u_star instead (the default `force_star_zero=False`), or
any tau != 0 (h_star = tau, not 0), would score the network against a ground
truth that no longer matches what it's actually being asked -- see
`tilt.sudoku6.exact_row_operator_error`'s full docstring for the derivation.
So this check is intentionally scoped to tau=0 / force_star_zero=True; a
reward-tilted 6x6 check would need a different (approximate or sample-based)
ground truth and is out of scope here.

Self-conditioning
-----------------
This checkpoint is self-conditioned, so it must be scored via
`net.natural_self_cond` (the two-pass "deployed" estimate), not the raw
one-pass `net.natural` (which would silently default `x_pred` to zero and
understate its real accuracy) -- `tilt.sudoku6.exact_row_operator_error_selfcond`
is the version that does this; it is otherwise identical to
`exact_row_operator_error`. `TransformerTilt` (this checkpoint's arch) exposes
`natural_self_cond` natively, same interface as `SelfCondTiltMLP`, so no
transformer-specific code is needed here.

No sl_sample/K/T at all
------------------------
Like Ising/Sudoku4x4 check1, this is a pure denoiser check (single VP-noise
step at each `a`, no SDE trajectory) -- eval/common/sample_config.py's
(T, K, schedule) don't apply here.

Independent draws per `a` point
---------------------------------
`exact_row_operator_error_selfcond(net, betas, ...)` takes a LIST of betas but
reseeds the SAME `torch.Generator().manual_seed(seed)` at the start of each
beta's own draw internally -- passing the whole grid in one call would reuse
the same underlying puzzles at every `a`, only the VP-noise scale differing
(this is exactly the confound the Ising check2 script's fix
removed). To draw fresh, independent puzzles at each grid point, this script
instead calls it once per SINGLE beta with a distinct seed per point
(`seed = base_seed + i`), mirroring `sudoku4x4_check1_denoiser_ood.py`'s
per-point-seed pattern.

Usage:
    python eval/sudoku6_check1_denoiser_ood.py
"""
from __future__ import annotations

import argparse
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

from tilt.load_ckpt import load_tilt_checkpoint  # noqa: E402
from tilt.noising import beta_of_a  # noqa: E402
from tilt.sudoku6 import exact_row_operator_error_selfcond  # noqa: E402

CKPT = ROOT / "ckpt" / "sudoku6_v2_l10_xf_sc_s8264_180k.pt"
ROW_NOISE = 0.02  # matches data/sudoku6_v2's recipe (meta.json); not read
                   # from a dataset -- there is no target.json / EnumerableTarget for 6x6.


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--row-noise", type=float, default=ROW_NOISE)
    p.add_argument("--n-a", type=int, default=20)
    p.add_argument("--n", type=int, default=100, help="draws per a-level")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "sudoku6"))
    args = p.parse_args()

    net, info = load_tilt_checkpoint(args.ckpt, map_location="cpu")
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- use "
                          "exact_row_operator_error (plain net.natural) instead")

    a_grid = np.linspace(0.0, 1.0, args.n_a + 2)[1:-1]
    beta_grid = beta_of_a(torch.tensor(a_grid)).tolist()

    print(f"row_noise={args.row_noise}  tau=0 (force_star_zero=True)")
    print(f"a grid ({args.n_a} pts): {a_grid[0]:.4f} .. {a_grid[-1]:.4f}")
    print(f"n={args.n}/point, independent draws per point")

    all_rows = []
    for i, b in enumerate(beta_grid):
        point_seed = args.seed + i
        rows = exact_row_operator_error_selfcond(
            net, [b], row_noise=args.row_noise, force_star_zero=True,
            n=args.n, seed=point_seed,
        )
        all_rows.append(rows[0])

    df = pd.DataFrame(all_rows)[["a", "beta", "sse", "rmse", "max", "bayes_floor"]]
    print(df.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "check1_denoiser_ood.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["a"], df["sse"], marker="o", ms=4, color="tab:orange")
    ax.set_xlabel("a (signal level)")
    ax.set_ylabel("SSE (net vs. exact row-marginal operator)")
    ax.set_title(f"Sudoku 6x6 d={net.cfg.d} core denoiser error vs. a, tau=0 "
                 f"(row_noise={args.row_noise}, n={args.n}/point, force_star_zero=True)")
    fig.tight_layout()
    fig.savefig(out_dir / "check1_denoiser_ood.png", dpi=150)

    print(f"\nsaved: {out_dir / 'check1_denoiser_ood.csv'}")
    print(f"saved: {out_dir / 'check1_denoiser_ood.png'}")


if __name__ == "__main__":
    main()
