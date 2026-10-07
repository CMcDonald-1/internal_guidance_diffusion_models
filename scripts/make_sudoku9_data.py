"""Build a 9x9 sudoku dataset (weak row generator + valid-solution injection).

The direct port of `scripts/make_sudoku6_data.py`. Three things differ, all of
them forced by the jump from 6x6 to 9x9.

INJECTION IS NO LONGER EXACTLY UNIFORM
--------------------------------------
`sudoku6_fast` enumerated the 39,168 reduced grids and sampled exactly
uniformly over all 28,200,960. At 9x9 there are 1.84e16 reduced grids and no
efficient exactly-uniform sampler is known. `tilt/sudoku9_fast.py` replaces it
with randomised backtracking (0.25 ms/grid) plus a uniform random element of
the full symmetry group |G| = 1,218,998,108,160, which is exactly uniform
WITHIN each orbit and pushes all residual bias into the orbit label. Read that
module's docstring before quoting any uniformity claim; the bias is real,
measured (z = 10.2 on a G-invariant statistic between two seed generators), and
bounded. `--gen mrv` builds the control dataset with the other generator.

Crucially this does NOT weaken the ground truth. `sudoku9.success_rate_exact`
gives P(valid | tau) in closed form from `p_inject` alone, because the reward
is binary -- it counts the valid set, never samples from it. The tau_eff
diagnostic that carried the 6x6 experiments transfers untouched.

NPY IS NOW THE ONLY SENSIBLE FORMAT
-----------------------------------
A row is 324 core bits plus a reward. As CSV that is ~840 bytes; as
grids.npy (n,81) uint8 + reward.npy (n,) int8 it is 82. At 10M rows: 820 MB
against 8.4 GB. `--format csv` still exists for small smoke sets only.

DIVERSITY BUDGET
----------------
Memorisation is driven by how many times each DISTINCT valid grid is seen. On
sudoku6_v1, 7,511 distinct grids seen ~9,800 times each produced 33.6% verbatim
reproduction (2,522x chance); sudoku6_v2's 248,778 grids at ~296 exposures
measured clean (0.5% exact vs 0.44% chance). The default here, 10M rows at
p_inject=0.05, gives ~500k distinct grids at ~148 exposures over a 180k-step
run at bs=8192 -- twice v2's margin, and affordable only because generation is
no longer the bottleneck it was at 6x6.

p_inject stays at 0.05. The reward must remain rare -- that is the point of
tilting. It is the DIVERSITY behind those 5% that grew.

Usage
-----
    python scripts/make_sudoku9_data.py --n 10000000 --outdir data/sudoku9_v1
    python scripts/make_sudoku9_data.py --n 20000 --outdir data/sudoku9_smoke --format csv
    python scripts/make_sudoku9_data.py --n 10000000 --outdir data/sudoku9_v1_mrv --gen mrv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tilt.sudoku9 import (D, GRID_COLS, GRID_ROWS, decode_grid_batch,  # noqa: E402
                          r_sudoku9_valid, sample_weak_rows,
                          success_rate_exact)
from tilt.sudoku9_fast import sample_valid_solutions_fast  # noqa: E402

CELLS = GRID_ROWS * GRID_COLS  # 81


def _gen_chunk(n, row_noise, p_inject, g, gen):
    """(n, 81) uint8 grids and (n,) int8 rewards for one chunk."""
    core = sample_weak_rows(n, row_noise, g)
    inject = torch.rand(n, generator=g) < p_inject
    n_inject = int(inject.sum())
    if n_inject:
        core[inject] = sample_valid_solutions_fast(n_inject, generator=g, gen=gen)
    star = r_sudoku9_valid(core)
    grids = decode_grid_batch(core).reshape(n, CELLS)
    return grids.to(torch.uint8).numpy(), star.to(torch.int8).numpy()


def make_sudoku9_dataset(n: int, outdir: str | Path, seed: int = 0,
                         row_noise: float = 0.02, p_inject: float = 0.05,
                         beta_lo: float = 0.02, beta_hi: float = 50.0,
                         gen: str = "rowmajor", fmt: str = "auto",
                         chunk: int = 500_000, verbose: bool = True) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    if fmt == "auto":
        fmt = "npy" if n > 1_000_000 else "csv"

    t0 = time.time()
    grids = np.empty((n, CELLS), dtype=np.uint8)
    reward = np.empty(n, dtype=np.int8)
    done = 0
    while done < n:
        m = min(chunk, n - done)
        grids[done:done + m], reward[done:done + m] = _gen_chunk(
            m, row_noise, p_inject, g, gen)
        done += m
        if verbose and n > chunk:
            rate = done / max(time.time() - t0, 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            print(f"  {done:,}/{n:,} rows  ({time.time()-t0:.1f}s, "
                  f"eta {eta/60:.1f} min)", flush=True)

    n_valid = int((reward > 0).sum())
    success_rate = n_valid / n
    distinct_valid = int(len(np.unique(grids[reward > 0], axis=0)))

    if fmt == "npy":
        np.save(outdir / "grids.npy", grids)
        np.save(outdir / "reward.npy", reward)
        written = outdir / "grids.npy"
    else:
        import pandas as pd
        from tilt.sudoku9 import encode_grid_batch
        core = encode_grid_batch(torch.from_numpy(grids.astype(np.int64))
                                 .reshape(n, GRID_ROWS, GRID_COLS))
        x = torch.cat([core, torch.from_numpy(reward.astype(np.float32))
                       .reshape(-1, 1)], dim=1).to(torch.int8).numpy()
        pd.DataFrame({f"x{i}": x[:, i] for i in range(D + 1)}).to_csv(
            outdir / "train.csv", index=False)
        written = outdir / "train.csv"

    meta = {
        "n": n, "d": D + 1, "seed": seed, "freeze_noise": False,
        "beta_lo": beta_lo, "beta_hi": beta_hi,
        "row_noise": row_noise, "p_inject": p_inject,
        "success_rate": success_rate,
        "success_rate_exact_tau0": success_rate_exact(row_noise, p_inject, 0.0),
        "n_valid": n_valid, "distinct_valid": distinct_valid,
        "format": fmt, "board": "9x9", "cells": CELLS, "bits_per_cell": 4,
        "seed_generator": gen,
        "injection": "sudoku9_fast: randomised backtracking + uniform element "
                     "of the full symmetry group (|G| = 1,218,998,108,160)",
        "note": ("empirical-only dataset. UNLIKE sudoku6, the injected valid "
                 "grids are NOT exactly uniform -- no efficient uniform "
                 "sampler exists at 9x9; the draw is exactly uniform within "
                 "each symmetry orbit and biased only in the orbit label (see "
                 "tilt/sudoku9_fast.py). This does not affect "
                 "success_rate_exact, which depends only on p_inject because "
                 "the reward is binary. distinct_valid is what bounds "
                 "memorisation: exposures per distinct valid grid = "
                 "steps * batch_size * p_inject / distinct_valid."),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    if verbose:
        mb = (grids.nbytes + reward.nbytes) / 1e6
        print(f"\nSaved {n:,} rows -> {written}  ({fmt}, {mb:.0f} MB)")
        print(f"  d={D+1}  success_rate={success_rate:.4%}  "
              f"(exact {meta['success_rate_exact_tau0']:.4%})")
        print(f"  valid={n_valid:,}  distinct_valid={distinct_valid:,}  "
              f"({100*distinct_valid/max(n_valid,1):.2f}% of valid rows distinct)")
        for steps, bs in ((180_000, 8192),):
            exp = steps * bs * p_inject / max(distinct_valid, 1)
            print(f"  exposures per distinct valid grid over {steps//1000}k "
                  f"steps at bs={bs}: {exp:,.0f}   "
                  f"(sudoku6_v1 9,816 -> memorised 33.6%; v2 296 -> clean)")
        print(f"  elapsed {time.time()-t0:.1f}s")
    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--outdir", type=str, default="data/sudoku9_smoke")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--row-noise", type=float, default=0.02)
    p.add_argument("--p-inject", type=float, default=0.05)
    p.add_argument("--format", choices=["auto", "npy", "csv"], default="auto")
    p.add_argument("--gen", choices=["rowmajor", "mrv"], default="rowmajor",
                   help="seed-grid generator. 'mrv' (most-constrained-cell "
                        "first) is 3.4x slower and exists as the control arm "
                        "for the orbit-label bias probe -- build the same "
                        "dataset with it and check whether anything moves.")
    p.add_argument("--chunk", type=int, default=500_000)
    args = p.parse_args()
    make_sudoku9_dataset(args.n, args.outdir, seed=args.seed,
                         row_noise=args.row_noise, p_inject=args.p_inject,
                         gen=args.gen, fmt=args.format, chunk=args.chunk)
