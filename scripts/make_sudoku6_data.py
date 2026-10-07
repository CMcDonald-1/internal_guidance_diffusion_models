"""Build a 6x6 sudoku dataset (weak row generator + exact-uniform
valid-solution injection).

No exact enumeration/ground truth is possible at this scale (28,200,960
valid solutions; row/column/box constraints jointly break the
row-independence trick used for the 4x4 case in tilt/sudoku.py) -- follows
the `rhyme_gpt2_v1` precedent: train.csv + meta.json only, no target.json.
See tilt/sudoku6.py for the generator and validity-checking machinery.

Two things changed for the large-dataset regime:

FAST INJECTION (default).  `sudoku6.sample_valid_solutions` costs ~240 ms per
grid, so the injected 5% dominated everything: 150k rows took 30 minutes.
`sudoku6_fast.sample_valid_solutions_fast` is the same distribution -- exactly
uniform over all 28,200,960 -- at ~3 us, via the (39,168 reduced grids) x (720
relabellings) bijection. 5M rows now takes seconds. Pass --slow to use the
original generator (e.g. to reproduce an existing dataset byte for byte).

COMPACT FORMAT (--format npy, default for n > 1M).  A row is 108 core bits
plus a reward, but those bits are a deterministic function of 36 cell values,
so grids.npy (n,36) uint8 + reward.npy (n,) int8 is 37 bytes/row against ~280
of CSV text -- 185 MB rather than 1.4 GB at 5M rows, and seconds rather than
minutes to load on every training job. `load_dataset` picks the format up
automatically via tilt/data_grids.py. --format csv keeps the old dense output.

Why the dataset size matters here: memorisation is driven by how many times
each DISTINCT valid grid is seen. At 150k rows and p_inject=0.05 that was
7,511 grids seen ~9,800 times each over a 180k-step run, and the model
reproduced 33.6% of its valid outputs verbatim (2,522x chance). p_inject is
deliberately unchanged -- the reward should stay rare; it is the DIVERSITY of
the grids behind that 5% that needs to grow.

Usage
-----
    python scripts/make_sudoku6_data.py --n 5000000 --outdir data/sudoku6_v2
    python scripts/make_sudoku6_data.py --n 20000 --outdir data/sudoku6_smoke --format csv
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

from tilt.sudoku6 import (D, GRID_COLS, GRID_ROWS, decode_grid_batch,  # noqa: E402
                          is_valid_sudoku6_batch, r_sudoku6_valid,
                          sample_valid_solutions, sample_weak_rows)


def _gen_chunk(n, row_noise, p_inject, g, fast):
    """(n, 36) uint8 grids and (n,) int8 rewards for one chunk."""
    core = sample_weak_rows(n, row_noise, g)
    inject = torch.rand(n, generator=g) < p_inject
    n_inject = int(inject.sum())
    if n_inject:
        if fast:
            from tilt.sudoku6_fast import sample_valid_solutions_fast
            core[inject] = sample_valid_solutions_fast(n_inject, generator=g)
        else:
            core[inject] = sample_valid_solutions(n_inject, g)
    star = r_sudoku6_valid(core)
    grids = decode_grid_batch(core).reshape(n, GRID_ROWS * GRID_COLS)
    return grids.to(torch.uint8).numpy(), star.to(torch.int8).numpy()


def make_sudoku6_dataset(n: int, outdir: str | Path, seed: int = 0,
                         row_noise: float = 0.02, p_inject: float = 0.05,
                         beta_lo: float = 0.02, beta_hi: float = 50.0,
                         fast: bool = True, fmt: str = "auto",
                         chunk: int = 500_000, verbose: bool = True) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    if fmt == "auto":
        fmt = "npy" if n > 1_000_000 else "csv"

    t0 = time.time()
    grids_parts, reward_parts = [], []
    done = 0
    while done < n:
        m = min(chunk, n - done)
        gr, rw = _gen_chunk(m, row_noise, p_inject, g, fast)
        grids_parts.append(gr)
        reward_parts.append(rw)
        done += m
        if verbose and n > chunk:
            print(f"  {done:,}/{n:,} rows  ({time.time()-t0:.1f}s)", flush=True)
    grids = np.concatenate(grids_parts) if len(grids_parts) > 1 else grids_parts[0]
    reward = np.concatenate(reward_parts) if len(reward_parts) > 1 else reward_parts[0]
    del grids_parts, reward_parts

    n_valid = int((reward > 0).sum())
    success_rate = n_valid / n
    distinct_valid = int(len(np.unique(grids[reward > 0], axis=0)))

    if fmt == "npy":
        np.save(outdir / "grids.npy", grids)
        np.save(outdir / "reward.npy", reward)
        written = outdir / "grids.npy"
    else:
        import pandas as pd
        from tilt.sudoku6 import encode_grid_batch
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
        "n_valid": n_valid, "distinct_valid": distinct_valid,
        "format": fmt, "fast_injection": bool(fast),
        "note": ("empirical-only dataset, no exact ground truth -- "
                 "success_rate is an exact count over THIS realized set, "
                 "not a population-level target quantity. distinct_valid is "
                 "what bounds memorisation: exposures per distinct valid grid "
                 "= steps * batch_size * p_inject / distinct_valid."),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    if verbose:
        mb = (grids.nbytes + reward.nbytes) / 1e6
        print(f"\nSaved {n:,} rows -> {written}  ({fmt}, {mb:.0f} MB)")
        print(f"  d={D+1}  success_rate={success_rate:.3%}  "
              f"valid={n_valid:,}  distinct_valid={distinct_valid:,}")
        for steps, bs in ((180_000, 8192),):
            exp = steps * bs * p_inject / max(distinct_valid, 1)
            print(f"  exposures per distinct valid grid over {steps//1000}k "
                  f"steps at bs={bs}: {exp:,.0f}")
        print(f"  elapsed {time.time()-t0:.1f}s")
    return written


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--outdir", type=str, default="data/sudoku6_smoke")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--row-noise", type=float, default=0.02)
    p.add_argument("--p-inject", type=float, default=0.05)
    p.add_argument("--format", choices=["auto", "npy", "csv"], default="auto")
    p.add_argument("--slow", action="store_true",
                   help="use sudoku6.sample_valid_solutions (~240 ms/grid) "
                        "instead of the exact fast sampler")
    p.add_argument("--chunk", type=int, default=500_000)
    args = p.parse_args()
    make_sudoku6_dataset(args.n, args.outdir, seed=args.seed,
                         row_noise=args.row_noise, p_inject=args.p_inject,
                         fast=not args.slow, fmt=args.format, chunk=args.chunk)
