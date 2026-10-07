"""6x6 sudoku puzzles encoded as {-1,1}^108 vectors.

A genuine 6x6 sudoku: 6 rows, 6 columns, and six non-overlapping 2-row x
3-column boxes (3 row-bands x 2 col-bands), each required to be a permutation
of {1,...,6}. This is a scaling proof-of-concept beyond the 4x4 case in
tilt/sudoku.py -- the state space here is far too large for exact enumeration
or even the row-factorization trick used there:

  - Raw row combinations: 6^6 per row (vs. 4^4=256 for 4x4).
  - Column and box constraints jointly break row-independence (unlike the 4x4
    base density, which only ever modeled rows as independent).
  - Total valid solutions: 28,200,960 (vs. 288 for 4x4) -- far too many to
    materialize as an explicit state list the way SudokuValidSolutionsTarget
    does for 4x4.

Consequently there is NO EnumerableTarget/exact-algebra hierarchy here (no
exact natural_mean/log_partition/mixture-math) -- this module follows the
`rhyme_gpt2_v1` precedent instead: an empirical-only target object exposing
just `.sample`/`.d`/`.tilt_direction`, no `target.json`, and only
sample-based (Monte Carlo) diagnostics downstream.

Encoding convention: cell values are 1-6. Zero-indexed k = v - 1 in
{0,...,5}, split into 3 bits (hi, mid, lo) via k's binary representation,
each mapped to {-1,1} via 2*b - 1. Cell (r, c) (0-indexed) occupies vector
positions (r*6+c)*3, (r*6+c)*3+1, (r*6+c)*3+2 (hi, mid, lo). Total d = 108.
Two of the 8 raw 3-bit codes per cell (k=6,7, i.e. decoded value 7 or 8) are
"wasted" -- they don't correspond to any of the 6 digits, and any unit
(row/column/box) containing one is automatically invalid.

The exact valid-solution generator (`sample_valid_solution`) is NOT a
heuristic: it's an exactly-uniform sampler over all 28,200,960 valid grids,
via a two-stage construction:

  1. Sample the first band (rows 0-1) by construction: choose a 3-digit
     subset A for the top-left box's row 0 (20 ways; B = complement of A).
     Row 0 = [shuffle(A), shuffle(B)], row 1 = [shuffle(B), shuffle(A)],
     each of the four length-3 groups shuffled independently (6 ways each)
     -- 20 * 6^4 = 25,920 equally likely bands, and the row/column/box
     constraints within the band hold automatically from the A/B
     complementary structure (each box gets A union B = all 6 digits, and
     any column pairs one A-digit with one B-digit, which are automatically
     distinct).
  2. Enumerate every completion of rows 2-5 given the band (exact
     backtracking, exploiting the same box-complement structure recursively
     at each band-pair). The completion count is NOT constant across bands
     -- exhaustively enumerating all 1,296 structurally-distinct shuffle
     combinations (fixing the top-left digit subset A, since digit
     relabeling is itself a symmetry) shows the count takes exactly one of
     three values: 928 (probability 1/3), 1088 (probability 1/2), or 1408
     (probability 1/6). The *average* over bands is exactly 1088
     (25,920 * 1088 = 28,200,960, matching the known total exactly), but
     individual bands are NOT all equal -- the symmetry group has (at
     least) three orbits on bands, not one, so picking a band uniformly and
     then a completion uniformly within it would NOT give a uniform draw
     over all 28.2M grids (bands with fewer completions would be
     over-represented per-grid).

  3. Correct for this via rejection sampling: accept a sampled band with
     probability `count_b / 1408` (1408 being the proven maximum), else
     redraw a fresh band; then pick uniformly among the accepted band's
     completions.

Steps 1-3 compose into an exact uniform draw over all valid grids (~27%
redraw overhead on average, since E[count_b]/1408 = 1088/1408 ~ 0.77).
"""
from __future__ import annotations

import itertools
import math

import torch

from .noising import a_of_beta, beta_of_a, vp_noise

N_DIGITS = 6
BITS_PER_CELL = 3
GRID_ROWS = 6
GRID_COLS = 6
BOX_ROWS = 2
BOX_COLS = 3
D = GRID_ROWS * GRID_COLS * BITS_PER_CELL  # 108
D_ROW = N_DIGITS * BITS_PER_CELL  # 18 -- bits in a single row
TOTAL_VALID_SOLUTIONS = 28_200_960

# The six non-overlapping 2x3 boxes (3 row-bands x 2 col-bands), each a list
# of (row, col) coordinates.
BOXES = [
    [(r, c) for r in range(r0, r0 + BOX_ROWS) for c in range(c0, c0 + BOX_COLS)]
    for r0 in range(0, GRID_ROWS, BOX_ROWS)
    for c0 in range(0, GRID_COLS, BOX_COLS)
]


# ------------------------------------------------------------------ encoding
def encode_grid(grid) -> torch.Tensor:
    """6x6 grid of ints 1-6 -> length-108 {-1,1} tensor (row-major, 3 bits/cell)."""
    g = torch.as_tensor(grid, dtype=torch.int64)
    if g.shape != (GRID_ROWS, GRID_COLS):
        raise ValueError(f"expected a 6x6 grid, got shape {tuple(g.shape)}")
    if ((g < 1) | (g > N_DIGITS)).any():
        raise ValueError("grid entries must be in {1,...,6}")
    return encode_grid_batch(g.unsqueeze(0))[0]


def decode_grid(x: torch.Tensor) -> torch.Tensor:
    """Length-108 {-1,1} tensor -> 6x6 grid of ints (row-major, 3 bits/cell).

    Decoded values can be 1-8: 7 and 8 correspond to the two "wasted" raw
    3-bit codes (k=6,7) and are never valid digits.
    """
    x = torch.as_tensor(x)
    if x.shape != (D,):
        raise ValueError(f"expected a length-{D} vector, got shape {tuple(x.shape)}")
    return decode_grid_batch(x.unsqueeze(0))[0]


def encode_grid_batch(grids: torch.Tensor) -> torch.Tensor:
    """(N, 6, 6) ints 1-6 -> (N, 108) {-1,1}, 3 bits/cell, row-major."""
    g = torch.as_tensor(grids, dtype=torch.int64)
    if g.shape[-2:] != (GRID_ROWS, GRID_COLS):
        raise ValueError(f"expected (N, 6, 6), got {tuple(g.shape)}")
    k = g - 1
    hi = (k >> 2) & 1
    mid = (k >> 1) & 1
    lo = k & 1
    bits = torch.stack([hi, mid, lo], dim=-1).reshape(g.shape[0], D)
    return 2.0 * bits.float() - 1.0


def encode_row_batch(rows: torch.Tensor) -> torch.Tensor:
    """(N, 6) ints 1-6 -> (N, 18) {-1,1}, 3 bits/cell (same bit convention as

    `encode_grid_batch`, but for a single row in isolation -- used by the
    exact row-marginal ground truth below).
    """
    r = torch.as_tensor(rows, dtype=torch.int64)
    if r.shape[-1] != N_DIGITS:
        raise ValueError(f"expected (..., 6), got {tuple(r.shape)}")
    k = r - 1
    hi = (k >> 2) & 1
    mid = (k >> 1) & 1
    lo = k & 1
    bits = torch.stack([hi, mid, lo], dim=-1).reshape(*r.shape[:-1], D_ROW)
    return 2.0 * bits.float() - 1.0


def encode_cell_batch(vals: torch.Tensor) -> torch.Tensor:
    """(...,) ints 1-6 -> (..., 3) {-1,1}, a single cell's 3-bit code."""
    v = torch.as_tensor(vals, dtype=torch.int64)
    k = v - 1
    hi = (k >> 2) & 1
    mid = (k >> 1) & 1
    lo = k & 1
    bits = torch.stack([hi, mid, lo], dim=-1)
    return 2.0 * bits.float() - 1.0


def decode_grid_batch(x: torch.Tensor) -> torch.Tensor:
    """Batched decode_grid: (N, 108) {-1,1} -> (N, 6, 6) ints (1-8, see decode_grid)."""
    x = torch.as_tensor(x)
    if x.ndim != 2 or x.shape[1] != D:
        raise ValueError(f"expected shape (N, {D}), got {tuple(x.shape)}")
    bits = x.reshape(x.shape[0], GRID_ROWS, GRID_COLS, BITS_PER_CELL)
    hi = (bits[..., 0] > 0).long()
    mid = (bits[..., 1] > 0).long()
    lo = (bits[..., 2] > 0).long()
    k = 4 * hi + 2 * mid + lo
    return k + 1


# --------------------------------------------------------------- validity
def _dup_or_invalid_mask(vals: torch.Tensor) -> torch.Tensor:
    """(N, 6) int values -> (N, 6) bool: True where that cell's value is out

    of range (>6, one of the two wasted raw codes) OR repeats elsewhere in
    the same group of 6 (the shared logic behind row/column/box conflicts).
    """
    n_cells = vals.shape[1]
    eq = vals.unsqueeze(-1) == vals.unsqueeze(-2)  # (N, n_cells, n_cells)
    eq = eq & ~torch.eye(n_cells, dtype=torch.bool, device=vals.device)
    dup = eq.any(dim=-1)
    invalid = vals > N_DIGITS
    return dup | invalid


def find_conflicts_batch(grids_or_x: torch.Tensor) -> torch.Tensor:
    """Batched conflict mask: (N, 108) or (N, 6, 6) -> (N, 6, 6) bool.

    True at cells participating in a duplicate value or an out-of-range
    (wasted-code) value within their row, column, or 2x3 box.
    """
    t = torch.as_tensor(grids_or_x)
    grids = decode_grid_batch(t) if t.ndim == 2 and t.shape[-1] == D else t
    if grids.shape[-2:] != (GRID_ROWS, GRID_COLS):
        raise ValueError(f"expected (N, 6, 6) grids, got {tuple(grids.shape)}")
    n = grids.shape[0]
    conflicts = torch.zeros(n, GRID_ROWS, GRID_COLS, dtype=torch.bool, device=grids.device)

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


def is_valid_sudoku6_batch(grids_or_x: torch.Tensor) -> torch.Tensor:
    """Batched validity: (N, 108) or (N, 6, 6) -> (N,) bool. True iff every

    row, column, and 2x3 box is exactly a permutation of {1,...,6}.
    """
    conflicts = find_conflicts_batch(grids_or_x)
    return ~conflicts.reshape(conflicts.shape[0], -1).any(dim=-1)


def r_sudoku6_valid(x: torch.Tensor) -> torch.Tensor:
    """+1 if the decoded 6x6 grid is a valid sudoku solution, else -1.

    Same reward-augmentation convention as `tilt.sudoku.r_sudoku_valid`:
    an arbitrary (nonlinear) function of x, meant to be appended as its own
    coordinate rather than tilted directly.
    """
    return torch.where(is_valid_sudoku6_batch(x), 1.0, -1.0)


# ------------------------------------------------------------ weak generator
def sample_weak_rows(n: int, row_noise: float = 0.02, generator=None) -> torch.Tensor:
    """Weak base generator, mirroring `tilt.sudoku.SudokuRowsTarget`'s

    row_noise design but restricted to always-decodable rows: each of the 6
    rows, independently, is w.p. (1 - row_noise) a uniformly random
    permutation of {1,...,6} (ignoring column/box constraints), else w.p.
    row_noise a row of 6 i.i.d. uniform draws from {1,...,6} WITH repeats
    (never the wasted 3-bit raw codes). Returns (n, 108) encoded {-1,1}.
    """
    n_rows = n * GRID_ROWS
    # Uniformly random permutation of {0,...,5} per row via argsort of iid
    # uniform noise, vectorized over all n*6 rows at once.
    perm_digits = torch.argsort(torch.rand(n_rows, N_DIGITS, generator=generator), dim=1) + 1
    noisy_digits = torch.randint(1, N_DIGITS + 1, (n_rows, N_DIGITS), generator=generator)
    is_noisy = torch.rand(n_rows, 1, generator=generator) < row_noise
    rows = torch.where(is_noisy, noisy_digits, perm_digits)
    grids = rows.reshape(n, GRID_ROWS, GRID_COLS)
    return encode_grid_batch(grids)


# --------------------------------------------------- exact valid-solution generator
_ALL_PERMS6 = list(itertools.permutations(range(1, N_DIGITS + 1)))  # 720 tuples


def sample_valid_band(generator=None) -> list[list[int]]:
    """Uniformly random valid band (rows 0-1) via the exact construction

    described in this module's docstring: 20 * 6^4 = 25,920 equally likely
    bands. Returns [row0, row1], each a length-6 list of ints 1-6.
    """
    all_digits = list(range(1, N_DIGITS + 1))
    idx = torch.randperm(N_DIGITS, generator=generator)[:3].tolist()
    a = [all_digits[i] for i in idx]
    b = [d for d in all_digits if d not in a]

    def shuffle(group: list[int]) -> list[int]:
        order = torch.randperm(len(group), generator=generator).tolist()
        return [group[i] for i in order]

    row0 = shuffle(a) + shuffle(b)
    row1 = shuffle(b) + shuffle(a)
    return [row0, row1]


def _row_perms_avoiding(columns_used: list[set], forced_halves=None):
    """Yield permutation tuples of 1-6 with perm[c] not in columns_used[c]

    for every column c. If `forced_halves = (left_set, right_set)` is given,
    only permutations formed by permuting those specific 3-digit sets into
    cols 0-2 / cols 3-5 respectively are considered (36 raw combinations
    instead of 720) -- used for the "bottom row of a box-pair" case, where
    the box constraint already forces exactly those two digit sets.
    """
    if forced_halves is None:
        for perm in _ALL_PERMS6:
            if all(perm[c] not in columns_used[c] for c in range(N_DIGITS)):
                yield perm
    else:
        left_set, right_set = forced_halves
        for left in itertools.permutations(left_set):
            if any(left[c] in columns_used[c] for c in range(3)):
                continue
            for right in itertools.permutations(right_set):
                if any(right[c] in columns_used[3 + c] for c in range(3)):
                    continue
                yield left + right


def enumerate_band_completions(band: list[list[int]]) -> list[list[list[int]]]:
    """All valid completions of rows 2-5 given `band` = [row0, row1].

    Exact backtracking exploiting the same box-complement structure used to
    construct the band itself, recursively at each of the two remaining
    band-pairs (rows 2-3, rows 4-5), with column-usage tracked across all
    filled rows. Returns a list of completions, each a list of 4 rows
    (rows 2, 3, 4, 5), each row a length-6 list of ints.

    Per this module's docstring, every band should have exactly 1,088
    completions -- worth spot-checking directly, e.g.:

        >>> len(enumerate_band_completions(sample_valid_band())) == 1088
    """
    columns_used = [{band[0][c], band[1][c]} for c in range(N_DIGITS)]
    results: list[list[list[int]]] = []

    def rec(rows_so_far: list[list[int]], columns_used: list[set]) -> None:
        if len(rows_so_far) == 4:
            results.append([list(r) for r in rows_so_far])
            return
        idx = len(rows_so_far)
        if idx % 2 == 0:
            # Top row of a new box-pair: any column-compatible permutation.
            for perm in _row_perms_avoiding(columns_used):
                new_cu = [columns_used[c] | {perm[c]} for c in range(N_DIGITS)]
                rec(rows_so_far + [list(perm)], new_cu)
        else:
            # Bottom row of the current box-pair: forced to the complement
            # of the top row's own two halves (box constraint), further
            # filtered by column usage.
            top = rows_so_far[-1]
            left_set = [d for d in range(1, N_DIGITS + 1) if d not in top[0:3]]
            right_set = [d for d in range(1, N_DIGITS + 1) if d not in top[3:6]]
            for perm in _row_perms_avoiding(columns_used, forced_halves=(left_set, right_set)):
                new_cu = [columns_used[c] | {perm[c]} for c in range(N_DIGITS)]
                rec(rows_so_far + [list(perm)], new_cu)

    rec([], columns_used)
    return results


# Proven exact maximum of enumerate_band_completions(band) over all 25,920
# valid bands (see module docstring: counts are always 928, 1088, or 1408).
_MAX_BAND_COMPLETIONS = 1408


def sample_valid_solution(generator=None) -> torch.Tensor:
    """Exactly-uniform random valid 6x6 sudoku solution.

    sample_valid_band + rejection sampling (to correct for band completion
    counts not being constant, see module docstring) + enumerate_band_completions
    + a uniform choice among the completions. Returns a (6, 6) tensor of
    ints 1-6.
    """
    while True:
        band = sample_valid_band(generator)
        completions = enumerate_band_completions(band)
        accept_prob = len(completions) / _MAX_BAND_COMPLETIONS
        if float(torch.rand((), generator=generator)) < accept_prob:
            break
    i = int(torch.randint(0, len(completions), (1,), generator=generator))
    rows = [band[0], band[1]] + completions[i]
    return torch.tensor(rows, dtype=torch.int64)


def sample_valid_solutions(n: int, generator=None) -> torch.Tensor:
    """n exactly-uniform i.i.d. valid solutions, encoded: (n, 108) {-1,1}."""
    grids = torch.stack([sample_valid_solution(generator) for _ in range(n)])
    return encode_grid_batch(grids)


# --------------------------------------------------------------------- target
class Sudoku6Target:
    """Empirical-only 6x6 sudoku target: no exact enumeration or ground-truth

    operator is available at this scale (see module docstring). Exposes only
    what the rest of the pipeline needs -- mirrors the `rhyme_gpt2_v1`
    dataset's precedent (no `target.json`, no `EnumerableTarget` machinery).

    `.sample(n)` draws core rows from the weak generator
    (`sample_weak_rows`), then w.p. `p_inject` per row replaces the draw with
    a freshly-drawn EXACT uniform valid solution (`sample_valid_solution`) --
    "reward injection" to ensure genuinely valid solutions appear often
    enough in training data, mirroring `sudoku_v1`'s convention for the 4x4
    case. The reward coordinate x* = r(x) (validity indicator) is appended,
    computed exactly per realized row -- giving d = 108 + 1 = 109.
    """

    def __init__(self, row_noise: float = 0.02, p_inject: float = 0.05):
        self.row_noise = float(row_noise)
        self.p_inject = float(p_inject)
        self.d = D + 1

    def sample_core(self, n: int, generator=None) -> torch.Tensor:
        core = sample_weak_rows(n, self.row_noise, generator)
        inject = torch.rand(n, generator=generator) < self.p_inject
        n_inject = int(inject.sum())
        if n_inject:
            core[inject] = sample_valid_solutions(n_inject, generator)
        return core

    def sample(self, n: int, generator=None) -> torch.Tensor:
        core = self.sample_core(n, generator)
        star = r_sudoku6_valid(core)
        return torch.cat([core, star.unsqueeze(1)], dim=1)

    def tilt_direction(self, tau: float) -> torch.Tensor:
        h = torch.zeros(self.d)
        h[-1] = tau
        return h

    def success_rate(self, n_mc: int = 20_000, generator=None) -> float:
        """Monte Carlo ESTIMATE only -- no closed form exists at this scale

        (unlike the 4x4 case's exact SudokuMixtureTarget.success_rate).
        """
        core = self.sample_core(n_mc, generator)
        return float((r_sudoku6_valid(core) > 0).float().mean())

    def spec(self) -> dict:
        return {"kind": "sudoku6", "row_noise": self.row_noise, "p_inject": self.p_inject}


# --------------------------------------- exact row-marginal ground truth
#
# `sample_weak_rows` generates each row independently as a two-component
# mixture: w.p. (1-row_noise) a uniform permutation of {1,...,6}, else w.p.
# row_noise six i.i.d. uniform cell draws (with repeats allowed). Because
# rows are independent and each component is itself either a small (720-state)
# enumeration or a fully-factorized (6 independent 6-way cells) distribution,
# the per-row conditional mean E[x_row | u, a] is exactly computable -- see
# rethinking_sudoku_problem.pdf Section 1 for the derivation this follows
# (generalized here to properly account for the random-cell branch's other-
# cells partition products when mixing the two branches; the PDF's own
# formula for the *mixing weight* is a simplification that only exactly
# cancels in the row_noise=0 special case). This is EXACT for the real
# training measure `sample_weak_rows(row_noise)` -- not the row_noise=0
# "idealized" proxy (`sample_idealized_rows`), and does not require materializing any of the
# 28.2M valid grids.
#
# The reward coordinate's own denoiser still has no closed form once a
# nonzero reward tilt is applied (too many valid solutions to enumerate),
# but the exact untilted/tilted proper-puzzle PROBABILITY (not the denoiser)
# is available in closed form regardless of tau -- see `success_rate_exact`
# below (PDF Sections 1 & 3).

_PERM_ROW_CODES = encode_row_batch(torch.tensor(_ALL_PERMS6, dtype=torch.int64))  # (720, 18)
_CELL_CODES = encode_cell_batch(torch.arange(1, N_DIGITS + 1))  # (6, 3)


@torch.no_grad()
def row_marginal_mean(v_row: torch.Tensor, row_noise: float = 0.02) -> torch.Tensor:
    """Exact E[x_row | v_row] under `sample_weak_rows`'s per-row mixture density.

    `v_row`: (N, 18) natural parameter for one row (6 cells x 3 bits each,
    same layout as `encode_row_batch`). Returns (N, 18) exact conditional
    mean.
    """
    n = v_row.shape[0]
    if row_noise <= 0.0:
        logits = v_row @ _PERM_ROW_CODES.T  # (N, 720)
        w = torch.softmax(logits, dim=1)
        return w @ _PERM_ROW_CODES

    # Branch A: uniform permutation (720 explicit states).
    logits_perm = v_row @ _PERM_ROW_CODES.T  # (N, 720)
    logZ_perm = torch.logsumexp(logits_perm, dim=1)  # (N,)
    mean_A = torch.softmax(logits_perm, dim=1) @ _PERM_ROW_CODES  # (N, 18)

    # Branch B: 6 independent cells, each uniform over {1,...,6}.
    v_cells = v_row.reshape(n, N_DIGITS, BITS_PER_CELL)  # (N, 6, 3)
    cell_logits = torch.einsum("ncb,vb->ncv", v_cells, _CELL_CODES)  # (N, 6, 6)
    logZ_cells_total = torch.logsumexp(cell_logits, dim=2).sum(dim=1)  # (N,) sum of per-cell log-Z
    w_cells = torch.softmax(cell_logits, dim=2)  # (N, 6, 6)
    mean_B = torch.einsum("ncv,vb->ncb", w_cells, _CELL_CODES).reshape(n, D_ROW)  # (N, 18)

    log_mass_A = math.log(1.0 - row_noise) - math.log(720.0) + logZ_perm
    log_mass_B = math.log(row_noise) - N_DIGITS * math.log(float(N_DIGITS)) + logZ_cells_total
    p_A = torch.sigmoid(log_mass_A - log_mass_B).unsqueeze(1)  # (N, 1)

    return p_A * mean_A + (1.0 - p_A) * mean_B


@torch.no_grad()
def exact_core_mean(h_core: torch.Tensor, row_noise: float = 0.02) -> torch.Tensor:
    """Exact E[x_core | h_core] -- `row_marginal_mean` applied independently

    to each of the 6 rows' 18-bit slice of a (N, 108) core natural parameter.
    """
    outs = [
        row_marginal_mean(h_core[:, r * D_ROW : (r + 1) * D_ROW], row_noise)
        for r in range(GRID_ROWS)
    ]
    return torch.cat(outs, dim=1)


@torch.no_grad()
def exact_row_operator_error(
    net,
    betas,
    row_noise: float = 0.02,
    h_extra: torch.Tensor | None = None,
    force_star_zero: bool = False,
    n: int = 2048,
    seed: int = 0,
):
    """Exact per-beta SSE of `net`'s CORE (first 108) predictions against the

    true row-marginal denoiser of `sample_weak_rows(row_noise)` -- mirrors
    `tilt.evaluate.operator_error`'s style/return format, but scoped to the
    108 core coordinates only (the reward coordinate has no closed-form
    denoiser once tilted -- see module note above).

    `h_extra`, if given, is added to the *network's* input natural parameter
    only (e.g. `target.tilt_direction(tau)` to probe reward-tilted inputs).

    `force_star_zero`, if True, overrides the network's reward-coordinate
    input to exactly 0 instead of passing through the real sampled
    `beta(a)*u_star` (mutually exclusive with `h_extra`, which is ignored
    if both are given). This makes `exact_core_mean(h_core)` an EXACT
    (not approximate) ground truth for the query: under VP noising,
    `u_star | x_star=+1 ~ N(a, 1-a^2)` and `u_star | x_star=-1 ~ N(-a,
    1-a^2)` have equal likelihood at `u_star=0` for any `a`, so observing
    `u_star=0` carries zero information about `x_star` (hence, since
    `x_star` is a deterministic function of the core bits, zero
    information about `x_core` beyond what `h_core` already gives) --
    `E[x_core | h_core, h_star=0] = E[x_core | h_core]` exactly. Without
    this (the default, `force_star_zero=False`, `h_extra=None`), the
    network sees the REAL sampled `u_star`, which generally does carry
    some information about core validity that `exact_core_mean` silently
    ignores -- a small pre-existing approximation gap that `force_star_zero`
    removes. (`exact_core_mean` itself remains an approximation in one
    other respect regardless of this flag: it's exact for the pure
    `sample_weak_rows` row model but does not account for whole-solution
    injection, i.e. `p_inject`, which does shift the true core marginal
    slightly whenever a target uses injection.)
    """
    rows = []
    for b in betas:
        g = torch.Generator().manual_seed(seed)
        a = a_of_beta(torch.full((n, 1), float(b)))
        x0_core = sample_weak_rows(n, row_noise, generator=g)
        star = r_sudoku6_valid(x0_core)
        x0_full = torch.cat([x0_core, star.unsqueeze(1)], dim=1)

        u_full = vp_noise(x0_full, a, generator=g)
        h_core = beta_of_a(a) * u_full[:, :D]
        h_full_net = beta_of_a(a) * u_full
        if force_star_zero:
            h_full_net = torch.cat([h_core, torch.zeros(n, 1)], dim=1)
        elif h_extra is not None:
            h_full_net = h_full_net + h_extra.unsqueeze(0)

        gt_core = exact_core_mean(h_core, row_noise)
        pred_core = net.natural(h_full_net)[:, :D]

        err = pred_core - gt_core
        rows.append(
            {
                "beta": b,
                "a": float(a[0, 0]),
                "rmse": float(err.pow(2).mean().sqrt()),
                "max": float(err.abs().max()),
                "sse": float(err.pow(2).sum(1).mean()),
                "bayes_floor": float((1 - gt_core**2).sum(1).mean()),
            }
        )
    return rows


def row_valid_mask(grids_batch: torch.Tensor) -> torch.Tensor:
    """(N, 6, 6) decoded grids -> (N, 6) bool: True where that ROW is a valid

    permutation of 1-6 (no duplicates, no illegal 7/8 codes). Vectorized
    counterpart of `is_valid_sudoku6_batch`, but per-row rather than
    per-cell/whole-grid -- the natural granularity for this pipeline's
    row-only sample-quality checks (e.g. "what fraction of rows came out
    valid" vs. "what fraction of full grids are entirely valid").
    """
    vals, _ = grids_batch.sort(dim=-1)
    no_dup = (vals[:, :, 1:] != vals[:, :, :-1]).all(dim=-1)
    in_range = ((grids_batch >= 1) & (grids_batch <= 6)).all(dim=-1)
    return no_dup & in_range


def find_single_bit_row_fix(row_bits: torch.Tensor) -> int | None:
    """Given one row's 18 hard-decoded {-1,1} bits (an invalid row), check

    whether flipping exactly one of those 18 bits makes the row a valid
    permutation of {1,...,6}. Returns that bit's local index (0-17) if
    there is EXACTLY one such fixing bit, else `None` (covers both "no
    single-bit fix exists" and "more than one single-bit fix exists" --
    both cases mean the row's failure isn't cleanly attributable to one
    bit, so callers should not flag a specific culprit).
    """
    row_bits = torch.as_tensor(row_bits).reshape(-1)
    if row_bits.numel() != D_ROW:
        raise ValueError(f"expected {D_ROW} bits, got {row_bits.numel()}")

    fixing_bits = []
    for i in range(D_ROW):
        flipped = row_bits.clone()
        flipped[i] = -flipped[i]
        cell_bits = flipped.reshape(N_DIGITS, BITS_PER_CELL)
        hi = (cell_bits[:, 0] > 0).long()
        mid = (cell_bits[:, 1] > 0).long()
        lo = (cell_bits[:, 2] > 0).long()
        k = 4 * hi + 2 * mid + lo
        vals = k + 1
        no_dup = len(torch.unique(vals)) == N_DIGITS
        in_range = bool(((vals >= 1) & (vals <= N_DIGITS)).all())
        if no_dup and in_range:
            fixing_bits.append(i)

    if len(fixing_bits) == 1:
        return fixing_bits[0]
    return None


def find_single_bit_grid_fix(grid: torch.Tensor, hard_bits: torch.Tensor | None = None) -> tuple[int, int] | None:
    """Given a decoded (6, 6) grid with exactly one row-invalid row, and

    that row's own 18 hard-decoded bits (either passed explicitly as
    `hard_bits`, an (108,) or (18,) {-1,1} tensor, or re-derived from
    `grid` itself via `encode_grid_batch` when omitted -- exact for any
    row whose decoded values are all in {1,...,6}, which an invalid row
    always is by construction of `decode_grid_batch`), find the single
    bit (as a (row_idx, local_bit_idx) pair, local_bit_idx in 0-17)
    whose flip would make that row a valid permutation of {1,...,6}.

    Returns `None` if zero rows are invalid, more than one row is
    invalid, or the one invalid row's failure isn't fixable by flipping
    exactly one bit (0 or >1 candidate bits) -- i.e. this only fires for
    the clean "single bit conflict in a row" case the caller asked about.
    """
    grid = torch.as_tensor(grid)
    if grid.shape != (GRID_ROWS, GRID_COLS):
        raise ValueError(f"expected a (6, 6) grid, got {tuple(grid.shape)}")

    invalid_rows = (~row_valid_mask(grid.unsqueeze(0)))[0].nonzero().flatten()
    if invalid_rows.numel() != 1:
        return None
    row_idx = int(invalid_rows[0])

    if hard_bits is None:
        # Not `encode_grid_batch` -- it validates values are in {1,...,6} and
        # would raise on the invalid row's own out-of-range (7/8) codes.
        k = grid.reshape(-1) - 1
        hi = (k >> 2) & 1
        mid = (k >> 1) & 1
        lo = k & 1
        bits = torch.stack([hi, mid, lo], dim=-1).reshape(-1)
        hard_bits = 2.0 * bits.float() - 1.0
    hard_bits = torch.as_tensor(hard_bits).reshape(-1)
    if hard_bits.numel() == D:
        row_bits = hard_bits[row_idx * D_ROW : (row_idx + 1) * D_ROW]
    elif hard_bits.numel() == D_ROW:
        row_bits = hard_bits
    else:
        raise ValueError(f"hard_bits must have {D} or {D_ROW} elements, got {hard_bits.numel()}")

    local_bit = find_single_bit_row_fix(row_bits)
    if local_bit is None:
        return None
    return row_idx, local_bit


def row_repeat_counts(grids_batch: torch.Tensor) -> torch.Tensor:
    """(N, 6, 6) decoded grids -> (N,) float: total duplicate-value count

    summed across all 6 rows of each grid (a row with all 6 distinct values
    contributes 0; a row with one repeated pair contributes 1; etc.) --
    a finer-grained severity measure than the boolean `row_valid_mask`,
    useful for e.g. comparing "how broken" invalid samples are on average
    across checkpoints, not just how often they're broken at all.
    """
    n = grids_batch.shape[0]
    counts = torch.zeros(n)
    for r in range(GRID_ROWS):
        row = grids_batch[:, r, :]
        uniq_counts = torch.tensor([len(torch.unique(row[i])) for i in range(n)])
        counts += GRID_COLS - uniq_counts
    return counts


def sudoku6_row_eval_fn(betas, row_noise: float = 0.02, n: int = 2048, seed: int = 0):
    """Build an `eval_fn(net) -> dict` for `tilt.train.train`'s `eval_fn` slot,

    mirroring `tilt.evaluate.sse_eval_fn`'s return shape (`{"beta_<b>":
    {"sse", "rmse"}, ..., "mean_sse": ...}`) so the same wandb
    flattening/sectioning code in `tilt/train.py` handles it unmodified.
    Backed by `exact_row_operator_error`, i.e. real ground truth (unlike
    most 6x6 datasets, which have none) -- appropriate for the step-1,
    non-reward-tilted base row measure only.
    """

    def _fn(net):
        rows = exact_row_operator_error(net, betas, row_noise=row_noise, n=n, seed=seed)
        out = {f"beta_{r['beta']:g}": {"sse": r["sse"], "rmse": r["rmse"]} for r in rows}
        out["mean_sse"] = sum(r["sse"] for r in rows) / len(rows)
        return out

    return _fn


@torch.no_grad()
def exact_row_operator_error_selfcond(
    net,
    betas,
    row_noise: float = 0.02,
    h_extra: torch.Tensor | None = None,
    force_star_zero: bool = False,
    n: int = 2048,
    seed: int = 0,
):
    """Self-conditioned counterpart of `exact_row_operator_error`: scores

    `net.natural_self_cond` (the two-pass "deployed" estimate, see
    `tilt.models_selfcond.SelfCondTiltMLP`) instead of `net.natural` --
    mirrors `tilt.evaluate_selfcond.operator_error_selfcond`'s reasoning:
    `net.natural` alone would silently default `x_pred` to zero and
    understate a self-conditioned network's real (two-pass) accuracy.

    `force_star_zero`: see `exact_row_operator_error`'s docstring -- same
    exact-ground-truth argument applies unchanged here.
    """
    rows = []
    for b in betas:
        g = torch.Generator().manual_seed(seed)
        a = a_of_beta(torch.full((n, 1), float(b)))
        x0_core = sample_weak_rows(n, row_noise, generator=g)
        star = r_sudoku6_valid(x0_core)
        x0_full = torch.cat([x0_core, star.unsqueeze(1)], dim=1)

        u_full = vp_noise(x0_full, a, generator=g)
        h_core = beta_of_a(a) * u_full[:, :D]
        h_full_net = beta_of_a(a) * u_full
        if force_star_zero:
            h_full_net = torch.cat([h_core, torch.zeros(n, 1)], dim=1)
        elif h_extra is not None:
            h_full_net = h_full_net + h_extra.unsqueeze(0)

        gt_core = exact_core_mean(h_core, row_noise)
        pred_core = net.natural_self_cond(h_full_net)[:, :D]

        err = pred_core - gt_core
        rows.append(
            {
                "beta": b,
                "a": float(a[0, 0]),
                "rmse": float(err.pow(2).mean().sqrt()),
                "max": float(err.abs().max()),
                "sse": float(err.pow(2).sum(1).mean()),
                "bayes_floor": float((1 - gt_core**2).sum(1).mean()),
            }
        )
    return rows


def sudoku6_row_eval_fn_selfcond(betas, row_noise: float = 0.02, n: int = 2048, seed: int = 0):
    """Self-conditioned counterpart of `sudoku6_row_eval_fn` -- see there for

    the return-shape rationale. Backed by `exact_row_operator_error_selfcond`.
    """

    def _fn(net):
        rows = exact_row_operator_error_selfcond(net, betas, row_noise=row_noise, n=n, seed=seed)
        out = {f"beta_{r['beta']:g}": {"sse": r["sse"], "rmse": r["rmse"]} for r in rows}
        out["mean_sse"] = sum(r["sse"] for r in rows) / len(rows)
        return out

    return _fn


def log_p_base(row_noise: float = 0.02) -> float:
    """Exact log P(sampled grid is one of the 28,200,960 valid solutions)

    under `sample_weak_rows(row_noise)`, no reward tilt / no injection --
    rethinking_sudoku_problem.pdf Section 1's `p_base`. All 6 rows must
    independently realize their band's specific valid-solution permutation;
    each row does so w.p. `(1-row_noise)/6! + row_noise/6^6` (uniform-
    permutation branch draws it exactly w.p. 1/6!, random-cell branch draws
    that exact ordering w.p. 1/6^6), and the events "grid == solution i" are
    disjoint across the `TOTAL_VALID_SOLUTIONS` solutions, so the total
    probability is `TOTAL_VALID_SOLUTIONS * p_row^6`.
    """
    p_row = (1.0 - row_noise) / math.factorial(N_DIGITS) + row_noise / (N_DIGITS**N_DIGITS)
    return math.log(TOTAL_VALID_SOLUTIONS) + GRID_ROWS * math.log(p_row)


def success_rate_exact(row_noise: float = 0.02, p_inject: float = 0.0, tau: float = 0.0) -> float:
    """Exact P(reward=+1) under `Sudoku6Target(row_noise, p_inject)`'s law,

    optionally under a reward tilt `tau` on the reward coordinate --
    rethinking_sudoku_problem.pdf Sections 1 & 3, closed form (no Monte
    Carlo, unlike `Sudoku6Target.success_rate`). Injection only affects the
    *untilted* base rate (`p_eff = (1-p_inject)*p_base + p_inject`, since
    injection always inserts an actual valid solution); `tau` is then the
    generic logistic reweighting of any Bernoulli reward indicator by a
    linear tilt on its own coordinate, independent of how `p_eff` arose.
    """
    p_base = math.exp(log_p_base(row_noise))
    p_eff = (1.0 - p_inject) * p_base + p_inject
    if tau == 0.0:
        return p_eff
    num = p_eff * math.exp(tau)
    den = num + (1.0 - p_eff) * math.exp(-tau)
    return num / den
