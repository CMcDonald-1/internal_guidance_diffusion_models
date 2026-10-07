"""9x9 sudoku puzzles encoded as {-1,1}^324 vectors.

The full-size case: 9 rows, 9 columns, and nine non-overlapping 3x3 boxes,
each required to be a permutation of {1,...,9}. Structurally identical to
`tilt/sudoku6.py`; everything that changed is a consequence of two numbers.

WHAT SCALED AND WHAT DID NOT
----------------------------
`d = 81 * 4 + 1 = 325`, against 109 at 6x6, so `sqrt(d) = 18.0278` and
`lambda_matrix` admits `lambda in [1, 18.0278)`. The sudoku6 experiments measured
that the sampler queries `lambda_eff = sqrt(d)` EXACTLY at `t=0`, and that
raising `LAMBDA_MAX` from 5.0 to 10.0 (0.48 -> 0.96 of `sqrt(109)`) moved
validity 0.605 -> 0.716 at tau=8. The corresponding setting here is
`LAMBDA_MAX = 17.0` -- the same ~94% fraction of `sqrt(d)`. Training below
that draws the hard band with probability exactly zero, which no reweighting
scheme can repair.

Cells now waste 7 of 16 raw 4-bit codes (k=9..15, decoded 10..16) rather than
2 of 8. A uniformly random 4-bit cell is a legal digit only 9/16 of the time,
against 6/8 at 6x6, so a random point in the cube is further from the data
manifold. `sample_weak_rows` deliberately never emits a dead code, exactly as
at 6x6 -- the dead codes are reachable by the SAMPLER, not by the data.

NO EXACT UNIFORM VALID-SOLUTION SAMPLER EXISTS HERE
---------------------------------------------------
At 6x6 `sudoku6_fast` enumerated the 39,168 grids whose first row is 1..6 and
sampled (reduced grid) x (relabelling) as an exact bijection onto all
28,200,960 valid grids. That trick does not survive the jump:

    9x9 valid grids            6,670,903,752,021,072,936,960
    / 9! relabellings                 18,383,222,420,692,992

1.84e16 reduced grids is not enumerable, cacheable, or indexable. See
`tilt/sudoku9_fast.py` for what replaces it (randomised backtracking plus a
uniform draw from the full symmetry group) and for the precise sense in which
the result is and is not uniform.

THE EXACT TARGET SURVIVES ANYWAY
--------------------------------
This matters less than it looks, because the quantity the sudoku6 experiments
actually validated against -- P(valid) as a function of tau -- does not depend
on the injected grids being uniform. The reward is binary, so tau does exactly
one thing, rescale the odds:

    logit P(valid) = logit p_eff + 2 * tau,     p_eff ~= p_inject

and `p_base` here is ~5.1e-29 (nine rows must each independently realise one
specific permutation), so `p_eff = p_inject` to far beyond machine precision.
`success_rate_exact` below is therefore exact ground truth at 9x9 for the same
reason it was at 6x6, and `tau_eff` remains comparable across checkpoints.

Encoding convention: cell values are 1-9. Zero-indexed k = v - 1 in {0,...,8},
split into 4 bits MSB-first, each mapped to {-1,1} via 2*b - 1. Cell (r, c)
occupies vector positions (r*9+c)*4 .. (r*9+c)*4+3. Total d = 324.
"""
from __future__ import annotations

import math

import torch

N_DIGITS = 9
BITS_PER_CELL = 4
GRID_ROWS = 9
GRID_COLS = 9
BOX_ROWS = 3
BOX_COLS = 3
D = GRID_ROWS * GRID_COLS * BITS_PER_CELL  # 324
D_ROW = N_DIGITS * BITS_PER_CELL  # 36 -- bits in a single row
TOTAL_VALID_SOLUTIONS = 6_670_903_752_021_072_936_960

# Number of dead 4-bit codes per cell (k = 9..15, decoded 10..16). Any unit
# containing one is automatically invalid.
N_DEAD_CODES = 2**BITS_PER_CELL - N_DIGITS  # 7

# The nine non-overlapping 3x3 boxes, each a list of (row, col) coordinates.
BOXES = [
    [(r, c) for r in range(r0, r0 + BOX_ROWS) for c in range(c0, c0 + BOX_COLS)]
    for r0 in range(0, GRID_ROWS, BOX_ROWS)
    for c0 in range(0, GRID_COLS, BOX_COLS)
]

# lambda_matrix admits lambda in [1, sqrt(d)); the sampler queries sqrt(d)
# exactly at t=0. See the module docstring.
SQRT_D = math.sqrt(D + 1)  # 18.027756...
LAMBDA_MAX_RECOMMENDED = 17.0


# ------------------------------------------------------------------ encoding
def _bits_from_k(k: torch.Tensor) -> torch.Tensor:
    """(...) ints in 0..15 -> (..., 4) {-1,1}, MSB first."""
    bits = torch.stack(
        [(k >> s) & 1 for s in range(BITS_PER_CELL - 1, -1, -1)], dim=-1
    )
    return 2.0 * bits.float() - 1.0


def encode_grid(grid) -> torch.Tensor:
    """9x9 grid of ints 1-9 -> length-324 {-1,1} tensor (row-major, 4 bits/cell)."""
    g = torch.as_tensor(grid, dtype=torch.int64)
    if g.shape != (GRID_ROWS, GRID_COLS):
        raise ValueError(f"expected a 9x9 grid, got shape {tuple(g.shape)}")
    if ((g < 1) | (g > N_DIGITS)).any():
        raise ValueError("grid entries must be in {1,...,9}")
    return encode_grid_batch(g.unsqueeze(0))[0]


def decode_grid(x: torch.Tensor) -> torch.Tensor:
    """Length-324 {-1,1} tensor -> 9x9 grid of ints (row-major, 4 bits/cell).

    Decoded values can be 1-16: 10-16 correspond to the seven dead 4-bit codes
    (k=9..15) and are never valid digits.
    """
    x = torch.as_tensor(x)
    if x.shape != (D,):
        raise ValueError(f"expected a length-{D} vector, got shape {tuple(x.shape)}")
    return decode_grid_batch(x.unsqueeze(0))[0]


def encode_grid_batch(grids: torch.Tensor) -> torch.Tensor:
    """(N, 9, 9) ints 1-9 -> (N, 324) {-1,1}, 4 bits/cell, row-major."""
    g = torch.as_tensor(grids, dtype=torch.int64)
    if g.shape[-2:] != (GRID_ROWS, GRID_COLS):
        raise ValueError(f"expected (N, 9, 9), got {tuple(g.shape)}")
    return _bits_from_k(g - 1).reshape(g.shape[0], D)


def encode_row_batch(rows: torch.Tensor) -> torch.Tensor:
    """(..., 9) ints 1-9 -> (..., 36) {-1,1}; one row in isolation."""
    r = torch.as_tensor(rows, dtype=torch.int64)
    if r.shape[-1] != N_DIGITS:
        raise ValueError(f"expected (..., 9), got {tuple(r.shape)}")
    return _bits_from_k(r - 1).reshape(*r.shape[:-1], D_ROW)


def encode_cell_batch(vals: torch.Tensor) -> torch.Tensor:
    """(...,) ints 1-9 -> (..., 4) {-1,1}, a single cell's 4-bit code."""
    return _bits_from_k(torch.as_tensor(vals, dtype=torch.int64) - 1)


def decode_grid_batch(x: torch.Tensor) -> torch.Tensor:
    """Batched decode_grid: (N, 324) {-1,1} -> (N, 9, 9) ints 1-16."""
    x = torch.as_tensor(x)
    if x.ndim != 2 or x.shape[1] != D:
        raise ValueError(f"expected shape (N, {D}), got {tuple(x.shape)}")
    bits = (x.reshape(x.shape[0], GRID_ROWS, GRID_COLS, BITS_PER_CELL) > 0).long()
    k = torch.zeros_like(bits[..., 0])
    for i in range(BITS_PER_CELL):
        k = (k << 1) | bits[..., i]
    return k + 1


# --------------------------------------------------------------- validity
def _dup_or_invalid_mask(vals: torch.Tensor) -> torch.Tensor:
    """(N, 9) int values -> (N, 9) bool: True where that cell is a dead code

    (>9) OR repeats elsewhere in the same group of 9.
    """
    n_cells = vals.shape[1]
    eq = vals.unsqueeze(-1) == vals.unsqueeze(-2)
    eq = eq & ~torch.eye(n_cells, dtype=torch.bool, device=vals.device)
    return eq.any(dim=-1) | (vals > N_DIGITS)


def find_conflicts_batch(grids_or_x: torch.Tensor) -> torch.Tensor:
    """Batched conflict mask: (N, 324) or (N, 9, 9) -> (N, 9, 9) bool."""
    t = torch.as_tensor(grids_or_x)
    grids = decode_grid_batch(t) if t.ndim == 2 and t.shape[-1] == D else t
    if grids.shape[-2:] != (GRID_ROWS, GRID_COLS):
        raise ValueError(f"expected (N, 9, 9) grids, got {tuple(grids.shape)}")
    n = grids.shape[0]
    conflicts = torch.zeros(n, GRID_ROWS, GRID_COLS, dtype=torch.bool,
                            device=grids.device)

    for r in range(GRID_ROWS):
        conflicts[:, r, :] |= _dup_or_invalid_mask(grids[:, r, :])
    for c in range(GRID_COLS):
        conflicts[:, :, c] |= _dup_or_invalid_mask(grids[:, :, c])
    for box in BOXES:
        vals = torch.stack([grids[:, r, c] for r, c in box], dim=1)
        dup = _dup_or_invalid_mask(vals)
        for k, (r, c) in enumerate(box):
            conflicts[:, r, c] |= dup[:, k]

    return conflicts


def is_valid_sudoku9_batch(grids_or_x: torch.Tensor) -> torch.Tensor:
    """(N, 324) or (N, 9, 9) -> (N,) bool. True iff every row, column and 3x3

    box is exactly a permutation of {1,...,9}.
    """
    conflicts = find_conflicts_batch(grids_or_x)
    return ~conflicts.reshape(conflicts.shape[0], -1).any(dim=-1)


def r_sudoku9_valid(x: torch.Tensor) -> torch.Tensor:
    """+1 if the decoded 9x9 grid is a valid sudoku solution, else -1."""
    return torch.where(is_valid_sudoku9_batch(x), 1.0, -1.0)


def row_valid_mask(grids_batch: torch.Tensor) -> torch.Tensor:
    """(N, 9, 9) decoded grids -> (N, 9) bool: True where that ROW is a valid

    permutation of 1-9 (no duplicates, no dead codes). The natural granularity
    for the failure decomposition P(all rows valid) x P(valid | all rows valid)
    that separated the two error terms at 6x6.
    """
    vals, _ = grids_batch.sort(dim=-1)
    no_dup = (vals[:, :, 1:] != vals[:, :, :-1]).all(dim=-1)
    in_range = ((grids_batch >= 1) & (grids_batch <= N_DIGITS)).all(dim=-1)
    return no_dup & in_range


def col_valid_mask(grids_batch: torch.Tensor) -> torch.Tensor:
    """(N, 9, 9) -> (N, 9) bool, per column."""
    return row_valid_mask(grids_batch.transpose(1, 2))


def box_valid_mask(grids_batch: torch.Tensor) -> torch.Tensor:
    """(N, 9, 9) -> (N, 9) bool, per 3x3 box (box index = 3*(r0//3) + c0//3)."""
    n = grids_batch.shape[0]
    boxes = torch.stack(
        [torch.stack([grids_batch[:, r, c] for r, c in box], dim=1) for box in BOXES],
        dim=1,
    )  # (N, 9, 9)
    return row_valid_mask(boxes.reshape(n, len(BOXES), N_DIGITS))


# ------------------------------------------------------------ weak generator
def sample_weak_rows(n: int, row_noise: float = 0.02, generator=None) -> torch.Tensor:
    """Weak base generator, the direct 9x9 analogue of sudoku6's.

    Each of the 9 rows, independently, is w.p. (1 - row_noise) a uniformly
    random permutation of {1,...,9} (ignoring column/box constraints), else
    w.p. row_noise nine i.i.d. uniform draws from {1,...,9} WITH repeats.
    Never emits a dead 4-bit code. Returns (n, 324) encoded {-1,1}.
    """
    n_rows = n * GRID_ROWS
    perm_digits = torch.argsort(
        torch.rand(n_rows, N_DIGITS, generator=generator), dim=1) + 1
    noisy_digits = torch.randint(1, N_DIGITS + 1, (n_rows, N_DIGITS),
                                 generator=generator)
    is_noisy = torch.rand(n_rows, 1, generator=generator) < row_noise
    rows = torch.where(is_noisy, noisy_digits, perm_digits)
    return encode_grid_batch(rows.reshape(n, GRID_ROWS, GRID_COLS))


# ------------------------------------------------------------------ target
class Sudoku9Target:
    """Empirical-only 9x9 sudoku target -- same contract as `Sudoku6Target`.

    `.sample(n)` draws core rows from `sample_weak_rows`, then w.p. `p_inject`
    per row replaces the draw with a freshly generated valid solution from
    `sudoku9_fast.sample_valid_solutions_fast`, and appends the reward
    coordinate x* = r(x), giving d = 324 + 1 = 325.

    The injected grids are NOT exactly uniform over all 6.67e21 valid grids
    (no such sampler exists) -- see `tilt/sudoku9_fast.py`. This does not
    affect `success_rate_exact`, which depends only on `p_inject`.
    """

    def __init__(self, row_noise: float = 0.02, p_inject: float = 0.05):
        self.row_noise = float(row_noise)
        self.p_inject = float(p_inject)
        self.d = D + 1

    def sample_core(self, n: int, generator=None) -> torch.Tensor:
        from .sudoku9_fast import sample_valid_solutions_fast

        core = sample_weak_rows(n, self.row_noise, generator)
        inject = torch.rand(n, generator=generator) < self.p_inject
        n_inject = int(inject.sum())
        if n_inject:
            core[inject] = sample_valid_solutions_fast(n_inject, generator=generator)
        return core

    def sample(self, n: int, generator=None) -> torch.Tensor:
        core = self.sample_core(n, generator)
        star = r_sudoku9_valid(core)
        return torch.cat([core, star.unsqueeze(1)], dim=1)

    def tilt_direction(self, tau: float) -> torch.Tensor:
        h = torch.zeros(self.d)
        h[-1] = tau
        return h

    def success_rate(self, n_mc: int = 20_000, generator=None) -> float:
        """Monte Carlo estimate; `success_rate_exact` is the closed form."""
        core = self.sample_core(n_mc, generator)
        return float((r_sudoku9_valid(core) > 0).float().mean())

    def spec(self) -> dict:
        return {"kind": "sudoku9", "row_noise": self.row_noise,
                "p_inject": self.p_inject}


# ------------------------------------------------------- exact success rate
def log_p_base(row_noise: float = 0.02) -> float:
    """Exact log P(sampled grid is one of the 6.67e21 valid solutions) under

    `sample_weak_rows(row_noise)`, no injection, no tilt. All 9 rows must
    independently realise their solution's specific permutation; each does so
    w.p. `(1-row_noise)/9! + row_noise/9^9`, and the events are disjoint
    across solutions.

    Evaluates to about -65.1 (p_base ~ 5.1e-29), against -22.3 (2.0e-10) at
    6x6 -- the weak generator is nine orders of magnitude further from ever
    producing a valid grid by chance, so injection is the only source of
    reward mass at 9x9 even more decisively than it was at 6x6.
    """
    p_row = (1.0 - row_noise) / math.factorial(N_DIGITS) + row_noise / (N_DIGITS**N_DIGITS)
    return math.log(TOTAL_VALID_SOLUTIONS) + GRID_ROWS * math.log(p_row)


def success_rate_exact(row_noise: float = 0.02, p_inject: float = 0.0,
                       tau: float = 0.0) -> float:
    """Exact P(reward=+1) under `Sudoku9Target(row_noise, p_inject)`, under a

    reward tilt `tau`. Injection only affects the untilted base rate
    (`p_eff = (1-p_inject)*p_base + p_inject`); `tau` is then the generic
    logistic reweighting of a Bernoulli indicator by a linear tilt on its own
    coordinate, independent of how `p_eff` arose.

    This is EXACT at 9x9 -- it never touches the set of valid grids, only
    counts it, so the absence of an exactly-uniform sampler is irrelevant
    here.
    """
    p_base = math.exp(log_p_base(row_noise))
    p_eff = (1.0 - p_inject) * p_base + p_inject
    if tau == 0.0:
        return p_eff
    num = p_eff * math.exp(tau)
    den = num + (1.0 - p_eff) * math.exp(-tau)
    return num / den


def tau_eff(p_observed: float, p_inject: float = 0.05,
            row_noise: float = 0.02) -> float:
    """Invert `success_rate_exact`: the tilt the model ACTUALLY delivered.

    `logit p = logit p_eff + 2 * tau_eff`. This is the number that was
    comparable across checkpoints at 6x6 where a raw validity rate was not --
    every MLP variant saturated near 2.0 while only the transformer rose with
    tau (3.09 / 3.30 / 3.55 at tau = 3 / 5 / 8).
    """
    p_base = math.exp(log_p_base(row_noise))
    p_eff = (1.0 - p_inject) * p_base + p_inject
    p = min(max(float(p_observed), 1e-12), 1 - 1e-12)
    return 0.5 * (math.log(p / (1 - p)) - math.log(p_eff / (1 - p_eff)))


if __name__ == "__main__":  # python -m tilt.sudoku9
    import json

    g = torch.Generator().manual_seed(0)
    from .sudoku9_fast import sample_valid_grids_fast

    grids = sample_valid_grids_fast(256, generator=g)
    enc = encode_grid_batch(grids)
    print(json.dumps({
        "d": D + 1,
        "sqrt_d": SQRT_D,
        "lambda_max_recommended": LAMBDA_MAX_RECOMMENDED,
        "dead_codes_per_cell": N_DEAD_CODES,
        "roundtrip_ok": bool((decode_grid_batch(enc) == grids).all()),
        "all_valid": bool(is_valid_sudoku9_batch(enc).all()),
        "weak_valid_rate": float(
            (r_sudoku9_valid(sample_weak_rows(4096, 0.02, g)) > 0).float().mean()),
        "log_p_base": log_p_base(0.02),
        "success_rate_exact": {
            f"tau={t}": success_rate_exact(0.02, 0.05, t) for t in (0, 3, 5, 8)
        },
    }, indent=2))
