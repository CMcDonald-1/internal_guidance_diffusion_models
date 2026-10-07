"""Fast generation of valid 9x9 sudoku grids.

WHY THE 6x6 TRICK IS GONE
-------------------------
`tilt/sudoku6_fast.py` works because every valid grid factors uniquely as
(reduced grid, digit relabelling) and 28,200,960 = 39,168 x 720. Enumerate the
39,168 reduced grids once (14 s, 1.41 MB), then sampling is an index plus a
gather: 3 us, exactly uniform. At 9x9 the same factorisation gives

    6,670,903,752,021,072,936,960 / 9!  =  18,383,222,420,692,992

reduced grids. 1.84e16 is not enumerable, not cacheable, not indexable. There
is no known efficient exactly-uniform sampler for 9x9 solution grids, and
building one is a research project (it needs the completion counts of the 44
Felgenhauer-Jarvis band classes AND a way to sample band-2 weighted by its own
completion count, which means enumerating ~1e7-1e8 bands per draw).

WHAT REPLACES IT
----------------
Two stages, both cheap:

  1. SEED. Randomised backtracking with 9-bit row/column/box masks, random
     candidate order at every cell. Measured 0.29 ms/grid in pure Python, so
     the 500,000 injected grids of a 10M-row dataset cost ~2.5 minutes. This
     is fast enough that no grid is ever reused -- there is no seed pool and
     no orbit sharing between dataset rows.

  2. SYMMETRISE. Apply a uniformly random element of the FULL sudoku symmetry
     group, vectorised in numpy at ~14 us/grid:

         band permutations                     3!        = 6
         row permutations within each band     (3!)^3    = 216
         stack permutations                    3!        = 6
         column permutations within each stack (3!)^3    = 216
         transpose                                       = 2
         digit relabelling                     9!        = 362,880
         ----------------------------------------------------------
         |G| = 1296 * 1296 * 2 * 362,880 = 1,218,998,108,160

IN WHAT SENSE THIS IS UNIFORM
-----------------------------
Precisely: uniform WITHIN each orbit, exactly. If the seed distribution were
uniform, a uniform random group element preserves that (the uniform measure is
G-invariant), so stage 2 can only help; and whatever bias randomised
backtracking has, stage 2 pushes all of it into the ~5,472,730,538 orbit
LABELS. Every statistic that is not G-invariant -- cell marginals, row
patterns, band structure, digit positions -- is therefore exactly right.

Measured, 200,000 draws from a SINGLE seed grid: all valid, 200,000/200,000
distinct (the orbit has 1.219e12 elements, so the birthday expectation is
200,000.0), cell-(0,0) value counts 22,567 / 22,132 / 22,180 / 22,250 /
22,150 / 21,936 / 22,058 / 22,366 / 22,361 against an expectation of 22,222.

WHAT THIS DOES NOT COMPROMISE
-----------------------------
`sudoku9.success_rate_exact` -- the exact P(valid | tau) ground truth the whole
tau_eff diagnostic is built on -- depends only on `p_inject` and the fact that
the reward is binary. It never touches the distribution over valid grids. So
the residual orbit-label bias is invisible to the primary measurement.

The one thing it does touch is the memorisation null: `run_sudoku9_memorization`
compares against draws from THIS sampler, which is the right null (it is the
training distribution), but "novel" there means novel as a grid, not novel as
an orbit. An orbit-level check would need canonicalisation over 3,359,232
positional elements per grid and is deliberately left out.

THE ORBIT-LABEL BIAS IS REAL AND MEASURED, NOT ASSUMED
------------------------------------------------------
`n_unavoidable_rect4` counts {r1,r2} x {c1,c2} rectangles whose four cells hold
only two digits crosswise. It is invariant under every element of G (verified
in `_selfcheck`), so it is blind to stage 2 and sees ONLY the orbit-label
distribution -- exactly the residual this file cannot control. Two independent
seed generators disagree on it, n = 20,000 each:

    row-major backtracking   0.31 ms/grid   18.998 +- 0.035
    most-constrained-first   1.04 ms/grid   18.497 +- 0.034
    difference               +0.500 (2.7%), SE 0.049,  z = +10.2

So neither is uniform, and the gap is detectable. Three reasons this is
recorded rather than fixed:

  * it cannot move `success_rate_exact`, which never touches the distribution
    over valid grids -- the primary metric is untouched;
  * the SET of valid grids is identical under both, and validity of a model's
    output is a property of that set, not of the measure on it;
  * both give ~500k distinct injected grids in huge orbits, so the
    memorisation bound is unchanged.

`--gen mrv` on the dataset builder is the control: build the same dataset with
the other generator and check whether validity and tau_eff move. Measured, not
assumed -- run it before trusting the third bullet.

Randomising the fill order per grid was tried as a third option and abandoned:
it removes the scan-direction artifact but backtracking blows up, >100x slower.

Usage
-----
    from tilt.sudoku9_fast import sample_valid_solutions_fast
    x = sample_valid_solutions_fast(500_000)     # (N, 324) encoded {-1,1}
"""
from __future__ import annotations

import random
import time

import numpy as np
import torch

from .sudoku9 import GRID_COLS, GRID_ROWS, N_DIGITS, encode_grid_batch

FULL_MASK = (1 << N_DIGITS) - 1
BOX_OF = [[(r // 3) * 3 + (c // 3) for c in range(9)] for r in range(9)]
GROUP_ORDER = 1_218_998_108_160          # 1296^2 * 2 * 9!
N_ESSENTIALLY_DIFFERENT = 5_472_730_538
_CHUNK = 100_000                          # grids per vectorised symmetry batch


# ------------------------------------------------------------------- stage 1
def _gen_one(rng: random.Random) -> list[int]:
    """One valid grid as a flat length-81 list of ints 1..9.

    Row-major randomised backtracking with bitmasks. The candidate order at
    every cell is shuffled, so this is not the same grid every time; it is not
    uniform either, which is what stage 2 is for.
    """
    row = [0] * 9
    col = [0] * 9
    box = [0] * 9
    g = [0] * 81

    def rec(p: int) -> bool:
        if p == 81:
            return True
        r, c = divmod(p, 9)
        b = BOX_OF[r][c]
        avail = FULL_MASK & ~(row[r] | col[c] | box[b])
        if not avail:
            return False
        cand = [k for k in range(N_DIGITS) if (avail >> k) & 1]
        rng.shuffle(cand)
        for k in cand:
            m = 1 << k
            row[r] |= m
            col[c] |= m
            box[b] |= m
            g[p] = k + 1
            if rec(p + 1):
                return True
            row[r] &= ~m
            col[c] &= ~m
            box[b] &= ~m
            g[p] = 0
        return False

    if not rec(0):                                   # unreachable for an empty grid
        raise RuntimeError("backtracking failed on an empty 9x9 grid")
    return g


def _gen_one_mrv(rng: random.Random) -> list[int]:
    """One valid grid, filling the most-constrained cell first.

    An INDEPENDENT seed generator, 3.4x slower, kept as the control arm for the
    orbit-label bias probe documented in the module docstring. Its bias is
    different from `_gen_one`'s, not smaller.
    """
    row = [0] * 9
    col = [0] * 9
    box = [0] * 9
    g = [0] * 81

    def rec(filled: int) -> bool:
        if filled == 81:
            return True
        best, best_n, best_av = -1, N_DIGITS + 1, 0
        for p in range(81):
            if g[p]:
                continue
            r, c = divmod(p, 9)
            av = FULL_MASK & ~(row[r] | col[c] | box[BOX_OF[r][c]])
            nc = av.bit_count()
            if nc < best_n:
                best, best_n, best_av = p, nc, av
                if nc <= 1:
                    break
        if best_n == 0:
            return False
        r, c = divmod(best, 9)
        b = BOX_OF[r][c]
        cand = [k for k in range(N_DIGITS) if (best_av >> k) & 1]
        rng.shuffle(cand)
        for k in cand:
            m = 1 << k
            row[r] |= m
            col[c] |= m
            box[b] |= m
            g[best] = k + 1
            if rec(filled + 1):
                return True
            row[r] &= ~m
            col[c] &= ~m
            box[b] &= ~m
            g[best] = 0
        return False

    if not rec(0):
        raise RuntimeError("MRV backtracking failed on an empty 9x9 grid")
    return g


GENERATORS = {"rowmajor": _gen_one, "mrv": _gen_one_mrv}


def seed_grids(n: int, rng: random.Random, gen: str = "rowmajor") -> np.ndarray:
    """(n, 9, 9) int8 valid grids from randomised backtracking."""
    try:
        fn = GENERATORS[gen]
    except KeyError:
        raise ValueError(f"unknown generator {gen!r}; pick one of {sorted(GENERATORS)}")
    return np.array([fn(rng) for _ in range(n)],
                    dtype=np.int8).reshape(n, GRID_ROWS, GRID_COLS)


# ------------------------------------------- G-invariant bias probe
_PAIRS = np.array([(a, b) for a in range(9) for b in range(a + 1, 9)])


def n_unavoidable_rect4(grids: np.ndarray) -> np.ndarray:
    """(n, 9, 9) -> (n,) count of size-4 unavoidable sets.

    A {r1,r2} x {c1,c2} rectangle with g[r1,c1] == g[r2,c2] and
    g[r1,c2] == g[r2,c1]. Invariant under the whole of G -- row/column/band/
    stack permutation, transpose and relabelling all map such rectangles to
    such rectangles -- so this statistic sees past stage 2 to the orbit label,
    which is the only place bias can survive here.
    """
    g = grids.astype(np.int64)
    a = g[:, _PAIRS[:, 0], :]
    b = g[:, _PAIRS[:, 1], :]
    return ((a[:, :, _PAIRS[:, 0]] == b[:, :, _PAIRS[:, 1]])
            & (a[:, :, _PAIRS[:, 1]] == b[:, :, _PAIRS[:, 0]])).sum(axis=(1, 2))


# ------------------------------------------------------------------- stage 2
def _random_perms(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """(n, k) independent uniform permutations of 0..k-1, via argsort."""
    return np.argsort(rng.random((n, k)), axis=1)


def apply_random_symmetry(grids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply an independent uniform element of G to each grid.

    `grids` is (n, 9, 9) with values 1..9; returns the same shape and dtype.
    """
    n = grids.shape[0]

    # rows: permute the 3 bands, then the 3 rows inside each band  (S3 wr S3)
    rperm = (_random_perms(n, 3, rng)[:, :, None] * 3
             + _random_perms(n * 3, 3, rng).reshape(n, 3, 3)).reshape(n, 9)
    # columns: the same structure on stacks
    cperm = (_random_perms(n, 3, rng)[:, :, None] * 3
             + _random_perms(n * 3, 3, rng).reshape(n, 3, 3)).reshape(n, 9)

    g = np.take_along_axis(grids, np.broadcast_to(rperm[:, :, None], (n, 9, 9)), axis=1)
    g = np.take_along_axis(g, np.broadcast_to(cperm[:, None, :], (n, 9, 9)), axis=2)

    flip = rng.random(n) < 0.5
    if flip.any():
        g[flip] = g[flip].transpose(0, 2, 1)

    relab = (_random_perms(n, N_DIGITS, rng) + 1).astype(grids.dtype)   # (n, 9)
    g = np.take_along_axis(np.broadcast_to(relab[:, None, :], (n, 9, 9)),
                           (g - 1).astype(np.int64), axis=2)
    return np.ascontiguousarray(g, dtype=grids.dtype)


# --------------------------------------------------------------- public API
def _seeds_from_torch(generator) -> tuple[int, int]:
    """Two independent 63-bit seeds derived from a torch Generator (or entropy)."""
    if generator is None:
        a = int(torch.randint(0, 2**62, (1,)).item())
        b = int(torch.randint(0, 2**62, (1,)).item())
    else:
        a = int(torch.randint(0, 2**62, (1,), generator=generator).item())
        b = int(torch.randint(0, 2**62, (1,), generator=generator).item())
    return a, b


def sample_valid_grids_np(n: int, seed: int | None = None, gen: str = "rowmajor",
                          verbose: bool = False) -> np.ndarray:
    """(n, 9, 9) int8 valid grids. Numpy-native entry point."""
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed if seed is None else seed ^ 0x9E3779B97F4A7C15)
    out = np.empty((n, GRID_ROWS, GRID_COLS), dtype=np.int8)
    t0 = time.time()
    done = 0
    while done < n:
        m = min(_CHUNK, n - done)
        out[done:done + m] = apply_random_symmetry(seed_grids(m, py_rng, gen), np_rng)
        done += m
        if verbose and n > _CHUNK:
            print(f"    grids {done:,}/{n:,}  ({time.time()-t0:.1f}s)", flush=True)
    return out


def sample_valid_grids_fast(n: int, generator=None, gen: str = "rowmajor") -> torch.Tensor:
    """(n, 9, 9) int64 valid grids. Signature matches sudoku6_fast's."""
    a, _ = _seeds_from_torch(generator)
    return torch.from_numpy(sample_valid_grids_np(n, seed=a, gen=gen).astype(np.int64))


def sample_valid_solutions_fast(n: int, generator=None,
                                gen: str = "rowmajor") -> torch.Tensor:
    """(n, 324) encoded {-1,1}. Drop-in for a `sample_valid_solutions`."""
    return encode_grid_batch(sample_valid_grids_fast(n, generator=generator, gen=gen))


# ------------------------------------------------------------------- checks
def _valid_np(g: np.ndarray) -> np.ndarray:
    """(n, 9, 9) -> (n,) bool, independent of the torch validity path."""
    n = g.shape[0]
    ok = np.ones(n, dtype=bool)
    target = np.arange(1, 10)
    for i in range(9):
        ok &= (np.sort(g[:, i, :], axis=1) == target).all(axis=1)
        ok &= (np.sort(g[:, :, i], axis=1) == target).all(axis=1)
    for r0 in (0, 3, 6):
        for c0 in (0, 3, 6):
            blk = g[:, r0:r0 + 3, c0:c0 + 3].reshape(n, 9)
            ok &= (np.sort(blk, axis=1) == target).all(axis=1)
    return ok


def _selfcheck(n: int = 50_000, seed: int = 0, probe: int = 0) -> dict:
    """Validity, distinctness against the birthday bound, marginal uniformity,

    and the single-seed orbit test (the strongest evidence that stage 2 is
    doing real work: 50,000 draws from ONE grid should still be all-distinct).

    The marginal test is a chi-square POOLED over all 81 cells (648 df). The
    per-cell version has 8 df and is far too noisy to read -- across three
    seeds it gave sd 36.6 / 63.9 / 127.0 against an expectation of 77.0, which
    looks alarming and means nothing. Pooled: z = -0.26 / +0.24 / -0.92.

    `probe > 0` additionally runs the G-invariant orbit-label comparison
    between the two seed generators; it is off by default because the MRV arm
    costs ~1 ms/grid.
    """
    t0 = time.time()
    g = sample_valid_grids_np(n, seed=seed)
    gen_s = time.time() - t0
    flat = g.reshape(n, 81)

    one = seed_grids(1, random.Random(seed + 1))
    orbit = apply_random_symmetry(np.repeat(one, min(n, 50_000), axis=0),
                                  np.random.default_rng(seed + 2))

    exp = n / N_DIGITS
    chi2 = sum(float((((np.bincount(flat[:, j], minlength=10)[1:] - exp) ** 2)
                      / exp).sum()) for j in range(81))
    df = 81 * (N_DIGITS - 1)

    total = 6_670_903_752_021_072_936_960
    out = {
        "n": n,
        "ms_per_grid": 1e3 * gen_s / n,
        "all_valid": bool(_valid_np(g).all()),
        "distinct": int(len(np.unique(flat, axis=0))),
        "distinct_expected": float(n - n * n / (2 * total)),
        "orbit_all_valid": bool(_valid_np(orbit).all()),
        "orbit_distinct": int(len(np.unique(orbit.reshape(-1, 81), axis=0))),
        "orbit_distinct_expected": float(
            len(orbit) - len(orbit) ** 2 / (2 * GROUP_ORDER)),
        "cell_marginal_chi2": chi2,
        "cell_marginal_df": df,
        "cell_marginal_z": (chi2 - df) / float(np.sqrt(2 * df)),
        "rect4_invariant_under_G": bool(
            (n_unavoidable_rect4(g[:2000])
             == n_unavoidable_rect4(apply_random_symmetry(
                 g[:2000].copy(), np.random.default_rng(seed + 3)))).all()),
        "group_order": GROUP_ORDER,
    }

    if probe:
        stats = {}
        for name in ("rowmajor", "mrv"):
            r = n_unavoidable_rect4(
                seed_grids(probe, random.Random(seed + 11), name))
            stats[name] = (float(r.mean()), float(r.std() / np.sqrt(probe)))
        (ma, sa), (mb, sb) = stats["rowmajor"], stats["mrv"]
        out["orbit_label_probe"] = {
            "n_per_arm": probe,
            "rect4_rowmajor": f"{ma:.3f} +- {sa:.3f}",
            "rect4_mrv": f"{mb:.3f} +- {sb:.3f}",
            "z": (ma - mb) / float(np.sqrt(sa ** 2 + sb ** 2)),
            "note": "nonzero z means neither generator is uniform; see module docstring",
        }
    return out


if __name__ == "__main__":
    import json
    import sys
    probe = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(json.dumps(_selfcheck(probe=probe), indent=2, default=str))
