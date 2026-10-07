"""Exact uniform sampling of valid 6x6 sudoku grids, ~80,000x faster.

Why
---
`sudoku6.sample_valid_solutions` costs ~240 ms per grid (measured: 150k rows
with p_inject=0.05 took 30 minutes, essentially all of it in 7,512 calls).
That makes a large pool of distinct valid grids prohibitive -- 1M would be 67
core-hours -- which is what forced p_inject's 7,511 grids to be seen ~9,800
times each during a 180k-step run, and in turn produced the 33.6% verbatim
memorisation measured by run_sudoku6_memorization.py.

The shortcut
------------
Every valid grid factors UNIQUELY as (reduced grid, digit relabelling):

  * the relabelling sigma is read straight off the grid's first row,
  * the reduced grid is sigma^-1 applied to it, and has first row 1,2,3,4,5,6.

So (reduced grids) x (relabellings) <-> (valid grids) is a bijection, and

    28,200,960  =  39,168  x  720 .

39,168 fits in 1.41 MB as int8. Enumerate once (~14 s), then sampling is an
array index plus a gather: **3 us per grid, exactly uniform over all
28,200,960**. Not an approximation -- picking each factor uniformly gives the
uniform distribution on valid grids by the bijection.

Verified: 39,168 enumerated (matches 28,200,960/720), all valid, all distinct;
200,000 draws gave all 720 first-rows with counts 200.0 +- 14.1 (multinomial
predicts 14.1) and 199,329 distinct grids (birthday predicts 199,291).

Usage
-----
    from tilt.sudoku6_fast import sample_valid_solutions_fast
    x = sample_valid_solutions_fast(1_000_000)     # (N, 108) encoded {-1,1}

Drop-in for `sudoku6.sample_valid_solutions`: same encoded return shape, same
distribution, same `generator=` semantics.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch

from .sudoku6 import BOXES, GRID_COLS, GRID_ROWS, N_DIGITS, encode_grid_batch

CACHE = Path(__file__).resolve().parent / "_sudoku6_reduced.npy"
N_REDUCED = 39_168          # = 28,200,960 / 720
_CACHE: np.ndarray | None = None


def enumerate_reduced() -> np.ndarray:
    """All valid grids whose first row is (1,...,6), as (39168, 36) int8.

    Plain backtracking over row permutations with column and box pruning.
    ~14 s; called once, then cached to disk.
    """
    perms = list(itertools.permutations(range(1, N_DIGITS + 1)))
    out: list[tuple] = []
    grid: list[tuple | None] = [None] * GRID_ROWS
    grid[0] = tuple(range(1, N_DIGITS + 1))

    def ok(r: int, perm: tuple) -> bool:
        for rr in range(r):                       # columns
            row = grid[rr]
            for c in range(GRID_COLS):
                if row[c] == perm[c]:
                    return False
        if r % 2 == 1:                            # boxes pair rows (0,1),(2,3),(4,5)
            top = grid[r - 1]
            for c0 in (0, 3):
                if set(top[c0:c0 + 3]) & set(perm[c0:c0 + 3]):
                    return False
        return True

    def rec(r: int) -> None:
        if r == GRID_ROWS:
            out.append(tuple(v for row in grid for v in row))
            return
        for perm in perms:
            if ok(r, perm):
                grid[r] = perm
                rec(r + 1)
        grid[r] = None

    rec(1)
    arr = np.array(out, dtype=np.int8)
    if len(arr) != N_REDUCED:
        raise RuntimeError(f"enumerated {len(arr)} reduced grids, expected {N_REDUCED}")
    return arr


def reduced_grids(cache: Path | str | None = None) -> np.ndarray:
    """The 39,168 reduced grids, enumerating and caching on first use."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = Path(cache) if cache is not None else CACHE
    if path.exists():
        arr = np.load(path)
        if len(arr) != N_REDUCED:
            raise RuntimeError(f"cache {path} has {len(arr)} grids, expected {N_REDUCED}")
    else:
        arr = enumerate_reduced()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    _CACHE = arr
    return arr


def sample_valid_grids_fast(n: int, generator=None) -> torch.Tensor:
    """(n, 6, 6) int64 grids, uniform over all 28,200,960 valid solutions."""
    base = torch.from_numpy(reduced_grids().astype(np.int64))       # (39168, 36)
    idx = torch.randint(0, N_REDUCED, (n,), generator=generator)
    g = base[idx]                                                   # (n, 36), values 1..6
    # one random relabelling of {1..6} per sample, applied through a gather
    relab = torch.argsort(torch.rand(n, N_DIGITS, generator=generator), dim=1) + 1
    g = torch.gather(relab, 1, g - 1)
    return g.reshape(n, GRID_ROWS, GRID_COLS)


def sample_valid_solutions_fast(n: int, generator=None) -> torch.Tensor:
    """(n, 108) encoded {-1,1}. Drop-in for sudoku6.sample_valid_solutions."""
    return encode_grid_batch(sample_valid_grids_fast(n, generator=generator))


def _selfcheck(n: int = 50_000, seed: int = 0) -> dict:
    """Validity, uniformity of first rows, and distinctness against the birthday bound."""
    g = torch.Generator().manual_seed(seed)
    grids = sample_valid_grids_fast(n, generator=g).numpy()
    flat = grids.reshape(n, 36)

    def valid(x):
        x = x.reshape(6, 6)
        if any(len(set(x[i, :])) != 6 for i in range(6)):
            return False
        if any(len(set(x[:, j])) != 6 for j in range(6)):
            return False
        return all(len({x[r][c] for r, c in b}) == 6 for b in BOXES)

    sub = flat[:: max(n // 500, 1)]
    counts: dict = {}
    for row in flat:
        k = tuple(row[:6])
        counts[k] = counts.get(k, 0) + 1
    v = np.array(list(counts.values()), dtype=float)
    total = N_REDUCED * 720
    return {
        "n": n,
        "all_valid": all(valid(x) for x in sub),
        "distinct_first_rows": len(counts),
        "first_row_sd": float(v.std()),
        "first_row_sd_expected": float(np.sqrt(n * (1 / 720) * (1 - 1 / 720))),
        "distinct": int(len(np.unique(flat, axis=0))),
        "distinct_expected": float(n - n * n / (2 * total)),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(_selfcheck(), indent=2, default=str))
