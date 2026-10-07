"""4x4 sudoku puzzles encoded as {-1,1}^32 vectors.

Encoding convention: cell values are 1-4. Zero-index k = v - 1 in {0,1,2,3},
split into 2 bits hi = k >> 1, lo = k & 1, mapped to {-1,1} via 2*b - 1. Cell
(r, c) (0-indexed) occupies vector positions 8*r + 2*c (hi) and 8*r + 2*c + 1
(lo). Total d = 32. Every bit pattern decodes to *some* grid -- the only way
a decoded grid can be "wrong" is a row/column/box duplicate value, never an
invalid bit pattern at the cell level.

This module is intentionally standalone (no dependency on tilt.targets yet):
Phase 1 is just encode/decode/validity/display utilities on top of the
{-1,1}^32 representation. A later phase will add a base-density target class
and hook it into tilt.targets' EnumerableTarget/RewardAugmented machinery.
"""
from __future__ import annotations

import itertools
import math

import torch

from .targets import EnumerableTarget, RewardAugmented

D = 32  # 4x4 grid, 2 bits/cell

# The four non-overlapping 2x2 boxes, each a list of (row, col) coordinates.
BOXES = [
    [(0, 0), (0, 1), (1, 0), (1, 1)],  # top-left
    [(0, 2), (0, 3), (1, 2), (1, 3)],  # top-right
    [(2, 0), (2, 1), (3, 0), (3, 1)],  # bottom-left
    [(2, 2), (2, 3), (3, 2), (3, 3)],  # bottom-right
]


def encode_grid(grid) -> torch.Tensor:
    """4x4 grid of ints 1-4 -> length-32 {-1,1} tensor (row-major, 2 bits/cell)."""
    g = torch.as_tensor(grid, dtype=torch.int64)
    if g.shape != (4, 4):
        raise ValueError(f"expected a 4x4 grid, got shape {tuple(g.shape)}")
    if ((g < 1) | (g > 4)).any():
        raise ValueError("grid entries must be in {1,2,3,4}")
    x = torch.empty(D, dtype=torch.float32)
    k = g - 1
    hi = (k >> 1) & 1
    lo = k & 1
    for r in range(4):
        for c in range(4):
            base = 8 * r + 2 * c
            x[base] = 2.0 * hi[r, c] - 1.0
            x[base + 1] = 2.0 * lo[r, c] - 1.0
    return x


def decode_grid(x: torch.Tensor) -> torch.Tensor:
    """Length-32 {-1,1} tensor -> 4x4 grid of ints 1-4 (row-major, 2 bits/cell)."""
    x = torch.as_tensor(x)
    if x.shape != (D,):
        raise ValueError(f"expected a length-{D} vector, got shape {tuple(x.shape)}")
    grid = torch.empty(4, 4, dtype=torch.int64)
    for r in range(4):
        for c in range(4):
            base = 8 * r + 2 * c
            hi = int(x[base] > 0)
            lo = int(x[base + 1] > 0)
            grid[r, c] = 2 * hi + lo + 1
    return grid


def _grid_of(grid_or_x) -> torch.Tensor:
    """Accept either a 4x4 grid or a length-32 {-1,1} vector; return a 4x4 grid."""
    t = torch.as_tensor(grid_or_x)
    if t.shape == (D,):
        return decode_grid(t)
    if t.shape == (4, 4):
        return t
    raise ValueError(f"expected a 4x4 grid or length-{D} vector, got shape {tuple(t.shape)}")


def find_conflicts(grid_or_x) -> torch.Tensor:
    """4x4 boolean mask: True at cells participating in a duplicate value

    within their row, column, or 2x2 box. Generic over all three unit types
    -- doesn't assume rows are already valid, since arbitrary/generated
    samples elsewhere in the pipeline may not respect that.
    """
    grid = _grid_of(grid_or_x)
    conflicts = torch.zeros(4, 4, dtype=torch.bool)

    def flag_unit(coords):
        vals = [int(grid[r, c]) for r, c in coords]
        dupes = {v for v in vals if vals.count(v) > 1}
        if not dupes:
            return
        for r, c in coords:
            if int(grid[r, c]) in dupes:
                conflicts[r, c] = True

    for r in range(4):
        flag_unit([(r, c) for c in range(4)])
    for c in range(4):
        flag_unit([(r, c) for r in range(4)])
    for box in BOXES:
        flag_unit(box)

    return conflicts


def is_valid_sudoku(grid_or_x) -> bool:
    """True iff find_conflicts(...) is all-False (rows, columns, and boxes

    each contain 1-4 with no repeats).
    """
    return not bool(find_conflicts(grid_or_x).any())


def print_puzzle(x) -> None:
    """Pretty-print a puzzle as a 4x4 grid, with conflicting cells in red.

    Accepts either a length-32 {-1,1} vector or a 4x4 grid. Conflicting
    cells (duplicate within their row/column/box) are wrapped in ANSI red;
    this renders in the Positron console/terminal. If your terminal doesn't
    support ANSI escapes, conflicting cells will look like garbled text --
    let me know and I'll switch to a plain-text marker instead.
    """
    grid = _grid_of(x)
    conflicts = find_conflicts(grid)

    RED = "\033[91m"
    RESET = "\033[0m"

    lines = []
    for r in range(4):
        cells = []
        for c in range(4):
            v = int(grid[r, c])
            s = f"{RED}{v}{RESET}" if conflicts[r, c] else str(v)
            cells.append(s)
        row_str = f"{cells[0]} {cells[1]} | {cells[2]} {cells[3]}"
        lines.append(row_str)
        if r == 1:
            lines.append("-" * 9)
    print("\n".join(lines))


# --------------------------------------------------------------- batched ops
def decode_grid_batch(x: torch.Tensor) -> torch.Tensor:
    """Batched decode_grid: (N, 32) {-1,1} -> (N, 4, 4) ints 1-4.

    Vectorized: the encoding's row-major layout means x.reshape(N, 4, 4, 2)
    already groups (row, col, bit) correctly -- no per-cell Python loop.
    """
    x = torch.as_tensor(x)
    if x.ndim != 2 or x.shape[1] != D:
        raise ValueError(f"expected shape (N, {D}), got {tuple(x.shape)}")
    bits = x.reshape(x.shape[0], 4, 4, 2)
    hi = (bits[..., 0] > 0).long()
    lo = (bits[..., 1] > 0).long()
    return 2 * hi + lo + 1


def _dup_mask(vals: torch.Tensor) -> torch.Tensor:
    """(N, 4) values -> (N, 4) bool, True where that value repeats elsewhere

    in the same group of 4 (the shared logic behind row/column/box conflicts).
    """
    eq = vals.unsqueeze(-1) == vals.unsqueeze(-2)  # (N, 4, 4)
    eq = eq & ~torch.eye(4, dtype=torch.bool, device=vals.device)
    return eq.any(dim=-1)


def find_conflicts_batch(grids_or_x: torch.Tensor) -> torch.Tensor:
    """Batched find_conflicts: (N, 32) or (N, 4, 4) -> (N, 4, 4) bool mask.

    Fully vectorized over N; only loops over the fixed 4 rows / 4 cols / 4
    boxes (each a group of 4 cells), so this scales to large N (e.g. the
    331,776-state sudoku support) without a per-puzzle Python loop.
    """
    t = torch.as_tensor(grids_or_x)
    grids = decode_grid_batch(t) if t.shape[-1] == D and t.ndim == 2 else t
    if grids.shape[-2:] != (4, 4):
        raise ValueError(f"expected (N, 4, 4) grids, got {tuple(grids.shape)}")
    n = grids.shape[0]
    conflicts = torch.zeros(n, 4, 4, dtype=torch.bool, device=grids.device)

    for r in range(4):
        dup = _dup_mask(grids[:, r, :])
        conflicts[:, r, :] |= dup
    for c in range(4):
        dup = _dup_mask(grids[:, :, c])
        conflicts[:, :, c] |= dup
    for box in BOXES:
        vals = torch.stack([grids[:, r, c] for r, c in box], dim=1)
        dup = _dup_mask(vals)
        for k, (r, c) in enumerate(box):
            conflicts[:, r, c] |= dup[:, k]

    return conflicts


def is_valid_sudoku_batch(grids_or_x: torch.Tensor) -> torch.Tensor:
    """Batched is_valid_sudoku: (N, 32) or (N, 4, 4) -> (N,) bool."""
    conflicts = find_conflicts_batch(grids_or_x)
    return ~conflicts.reshape(conflicts.shape[0], -1).any(dim=-1)


# ---------------------------------------------------------- row marginals
# All 4^4 = 256 possible cell-value assignments for a single row (not just
# the 24 permutations) -- the full per-row support once row_noise > 0.
_ROW_VALUES = torch.tensor(list(itertools.product([1, 2, 3, 4], repeat=4)))  # (256, 4)


def _row_states() -> torch.Tensor:
    """(256, 8) {-1,1} encoding of every possible row (2 bits/cell), in the

    same order as _ROW_VALUES -- i.e. row index i corresponds to
    _ROW_VALUES[i] and encodes to base-4 index k0*64+k1*16+k2*4+k3 (k = v-1),
    which is exactly itertools.product's iteration order (first coordinate
    slowest). row_index_of() below relies on this correspondence.
    """
    k = _ROW_VALUES - 1
    hi = (k >> 1) & 1
    lo = k & 1
    codes = torch.empty(256, 8, dtype=torch.float32)
    codes[:, 0::2] = 2.0 * hi.float() - 1.0
    codes[:, 1::2] = 2.0 * lo.float() - 1.0
    return codes


def _row_is_perm() -> torch.Tensor:
    """(256,) bool: True for the 24 rows among the 256 that are permutations

    of {1,2,3,4} (all 4 distinct values).
    """
    return torch.tensor([len(set(row.tolist())) == 4 for row in _ROW_VALUES])


ROW_STATES = _row_states()  # (256, 8)
ROW_IS_PERM = _row_is_perm()  # (256,)


def row_index_of(row_bits: torch.Tensor) -> torch.Tensor:
    """(N, 8) {-1,1} row blocks -> (N,) index into ROW_STATES/ROW_IS_PERM

    (0-255), computed arithmetically (no search) via the same base-4 packing
    ROW_STATES was built with.
    """
    bits = row_bits.reshape(row_bits.shape[0], 4, 2)
    hi = (bits[..., 0] > 0).long()
    lo = (bits[..., 1] > 0).long()
    k = 2 * hi + lo  # (N, 4), zero-indexed cell values
    return k[:, 0] * 64 + k[:, 1] * 16 + k[:, 2] * 4 + k[:, 3]


def row_log_prob(row_noise: float) -> torch.Tensor:
    """(256,) log-probability of each of the 256 row states under: w.p.

    (1 - row_noise) a uniform permutation of {1,2,3,4}; w.p. row_noise a
    uniformly random row (any of the 256 combinations, permutation or not).
    row_noise=0 recovers "uniform over the 24 permutations, zero elsewhere"
    exactly (log(0) = -inf on the other 232 rows).
    """
    is_perm = ROW_IS_PERM.float()
    prob = row_noise / 256.0 + (1.0 - row_noise) / 24.0 * is_perm
    return torch.log(prob)


# --------------------------------------------------------------------- target
class SudokuRowsTarget(EnumerableTarget):
    """Base density: 4 independent rows, each w.p. (1 - row_noise) a uniform

    permutation of {1,2,3,4}, and w.p. row_noise a *completely* random row
    (any of the 256 possible cell-value assignments, which may violate the
    permutation rule) -- so every puzzle in {-1,1}^32 has positive
    probability once row_noise > 0 ("full support").

    row_noise=0 (default) is the original pure-permutation-rows target: d=32
    exceeds MAX_ENUM_DIM=24, but the support is still a small, explicit list
    (24^4 = 331,776 states), so this overrides _build_states() and keeps
    exact .states/.probs available -- this is the branch used everywhere so
    far (success_rate cross-check against the 288 known solutions, etc).

    row_noise>0 blows the true support up to the full 2^32, which can't be
    materialized as an explicit state list -- but rows are independent, so
    every exact quantity (log_Z, mean, natural_mean/G, log_partition,
    effective_support) is still computed exactly via the 256-state *row*
    marginal, never touching a (2^32, 32) or even (331776, 32) tensor. In
    this branch `.states`/`.probs` are not available (set to None); anything
    that needs an explicit enumerated state list (e.g. some scripts that
    index target.states directly) won't work here -- only the documented
    methods below are exact-and-supported when row_noise > 0.
    """

    def __init__(self, row_noise: float = 0.0, device=None):
        self.row_noise = float(row_noise)
        if self.row_noise == 0.0:
            super().__init__(D, device)
        else:
            self.d = D
            self.device = device
            self.states = None
            self.probs = None
            base_row_lp = row_log_prob(self.row_noise).to(device) if device else row_log_prob(self.row_noise)
            # (4, 256): one log-prob vector per row, identical across rows until
            # condition_on_coords masks specific rows differently (see below).
            self._row_log_probs = base_row_lp.unsqueeze(0).repeat(4, 1)
            self._row_states = ROW_STATES.to(device) if device else ROW_STATES

    @classmethod
    def _from_row_log_probs(cls, row_log_probs: torch.Tensor, row_states: torch.Tensor,
                             device=None) -> "SudokuRowsTarget":
        """Build a row_noise>0-style instance directly from per-row log-prob
        vectors, bypassing __init__ -- used by condition_on_coords, whose
        conditioned rows generally differ from each other (unlike the shared
        vector every row starts with)."""
        obj = cls.__new__(cls)
        obj.row_noise = -1.0  # sentinel: always use the row-factorized path below
        obj.d = D
        obj.device = device
        obj.states = None
        obj.probs = None
        obj._row_log_probs = row_log_probs
        obj._row_states = row_states
        return obj

    # -- row_noise == 0 branch: exact explicit enumeration (unchanged) -----
    def _build_states(self, device=None) -> torch.Tensor:
        perms = torch.tensor(list(itertools.permutations([1, 2, 3, 4])))  # (24, 4)
        k = perms - 1
        hi = (k >> 1) & 1
        lo = k & 1
        row_code = torch.empty(24, 8, dtype=torch.float32)
        row_code[:, 0::2] = 2.0 * hi.float() - 1.0
        row_code[:, 1::2] = 2.0 * lo.float() - 1.0

        idx = torch.cartesian_prod(*[torch.arange(24) for _ in range(4)])  # (24^4, 4)
        states = torch.cat([row_code[idx[:, r]] for r in range(4)], dim=1)  # (24^4, 32)
        return states.to(device) if device is not None else states

    def log_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device)

    # -- row_noise > 0 branch: exact row-factorized quantities --------------
    def sample(self, n: int, generator=None) -> torch.Tensor:
        if self.row_noise == 0.0:
            return super().sample(n, generator=generator)
        rows = []
        for r in range(4):
            probs_r = torch.softmax(self._row_log_probs[r], dim=0)
            idx = torch.multinomial(probs_r, n, replacement=True, generator=generator)
            rows.append(self._row_states[idx])
        return torch.cat(rows, dim=1)

    def mean(self) -> torch.Tensor:
        if self.row_noise == 0.0:
            return super().mean()
        means = [torch.softmax(self._row_log_probs[r], dim=0) @ self._row_states
                 for r in range(4)]
        return torch.cat(means)

    @torch.no_grad()
    def natural_mean(self, h: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        if self.row_noise == 0.0:
            return super().natural_mean(h, chunk=chunk)
        h = torch.atleast_2d(h)
        means = []
        for r in range(4):
            h_r = h[:, 8 * r : 8 * r + 8]
            logits = self._row_log_probs[r].unsqueeze(0) + h_r @ self._row_states.T
            means.append(torch.softmax(logits, dim=1) @ self._row_states)
        return torch.cat(means, dim=1)

    @torch.no_grad()
    def log_partition(self, h: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        if self.row_noise == 0.0:
            return super().log_partition(h, chunk=chunk)
        h = torch.atleast_2d(h)
        total = torch.zeros(h.shape[0], device=h.device)
        for r in range(4):
            h_r = h[:, 8 * r : 8 * r + 8]
            logits = self._row_log_probs[r].unsqueeze(0) + h_r @ self._row_states.T
            total = total + torch.logsumexp(logits, dim=1)
        return total

    def G(self, u: torch.Tensor, a: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        from .noising import beta_of_a

        return self.natural_mean(beta_of_a(a) * u, chunk=chunk)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact log pi(x) for an explicit batch of candidate states (N, 32),

        via the row-factorized marginal -- works regardless of row_noise
        (or conditioning), and doesn't require enumerating the full support.
        """
        x = torch.atleast_2d(x)
        rows = x.reshape(x.shape[0], 4, 8)
        row_lps = (self._row_log_probs if self.row_noise != 0.0
                   else row_log_prob(0.0).to(x.device).unsqueeze(0).expand(4, -1))
        total = torch.zeros(x.shape[0], device=x.device)
        for r in range(4):
            idx = row_index_of(rows[:, r, :])
            total = total + row_lps[r][idx]
        return total

    def effective_support(self) -> float:
        if self.row_noise == 0.0:
            return super().effective_support()
        total = 1.0
        for r in range(4):
            p = torch.softmax(self._row_log_probs[r], dim=0)
            total *= float(torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum()))
        return total

    @torch.no_grad()
    def constraint_log_prob(self, coords, values) -> float:
        """log P_self(x_coords = values), the exact marginal log-probability
        this instance's OWN current (possibly already-conditioned) density
        assigns to fixing a subset of coordinates -- computed row-by-row (no
        enumeration of the up-to-24^4 matching states needed). Used by
        SudokuMixtureTarget.condition_on_coords to Bayes-update its mixing
        weight after conditioning.
        """
        if self.row_noise == 0.0:
            return super().constraint_log_prob(coords, values)
        total = 0.0
        for r in range(4):
            in_row = [(c, v) for c, v in zip(coords, values) if int(c) // 8 == r]
            if not in_row:
                continue
            mask = torch.ones(256, dtype=torch.bool, device=self._row_states.device)
            for c, v in in_row:
                sign = 1.0 if float(v) > 0 else -1.0
                mask &= (torch.sign(self._row_states[:, int(c) % 8]) == sign)
            row_lp = self._row_log_probs[r]
            if not torch.isfinite(row_lp[mask]).any():
                return float("-inf")
            total += float(torch.logsumexp(row_lp[mask], dim=0))
        return total

    def condition_on_coords(self, coords, values) -> "SudokuRowsTarget":
        """Exact pi(x | x_coords = values): the true hard-fixed conditional,
        not a soft tilt. Rows are independent, so fixing coordinates within
        row r only ever masks row r's own 256-state marginal to -inf outside
        the matching subset -- untouched rows keep their original marginal.
        No renormalisation is needed here (natural_mean/log_partition/sample
        renormalise internally via softmax), which is also why iterating
        this (conditioning further on an already-conditioned instance) just
        works.
        """
        if self.row_noise == 0.0:
            return super().condition_on_coords(coords, values)
        row_log_probs = self._row_log_probs.clone()
        for c, v in zip(coords, values):
            r, local = divmod(int(c), 8)
            sign = 1.0 if float(v) > 0 else -1.0
            keep = torch.sign(self._row_states[:, local]) == sign
            masked = torch.where(
                keep, row_log_probs[r], torch.full_like(row_log_probs[r], -float("inf"))
            )
            if not torch.isfinite(masked).any():
                raise ValueError(f"row {r}: given constraint has probability zero under this target")
            # Renormalise so this row's log-probs sum to 1 in exp-space (i.e.
            # log_partition(0) == 0 for this row), matching the convention
            # EnumerableTarget._from_filtered uses -- softmax-based methods
            # (natural_mean, sample) are invariant to this uniform shift, but
            # logsumexp-based ones (log_partition, log_prob,
            # constraint_log_prob) are NOT, and several downstream
            # computations (e.g. SudokuMixtureTarget's Bayes-updated mixing
            # weight) rely on every conditioned component being
            # self-normalised this way.
            row_log_probs[r] = masked - torch.logsumexp(masked, dim=0)
            if not torch.isfinite(row_log_probs[r]).any():
                raise ValueError(f"row {r}: given constraint has probability zero under this target")
        return SudokuRowsTarget._from_row_log_probs(row_log_probs, self._row_states, device=self.device)

    def spec(self) -> dict:
        return {"kind": "sudoku_rows", "row_noise": self.row_noise}

    @classmethod
    def build(cls, row_noise: float = 0.0, device=None):
        return cls(row_noise=row_noise, device=device)


# ------------------------------------------------------------ reward target
_VALID_SOLUTIONS_CACHE: torch.Tensor | None = None


def valid_solutions(device=None) -> torch.Tensor:
    """(288, 32) tensor of every valid 4x4 sudoku solution, cached after the

    first call (built once from the row_noise=0 SudokuRowsTarget's 331,776
    states, filtered to those that also satisfy column/box validity).
    """
    global _VALID_SOLUTIONS_CACHE
    if _VALID_SOLUTIONS_CACHE is None:
        base = SudokuRowsTarget(row_noise=0.0)
        mask = is_valid_sudoku_batch(base.states)
        _VALID_SOLUTIONS_CACHE = base.states[mask].clone()
    return _VALID_SOLUTIONS_CACHE.to(device) if device is not None else _VALID_SOLUTIONS_CACHE


def uniform_on_reward_set(n: int, generator=None, device=None) -> torch.Tensor:
    """Draw n samples i.i.d. uniformly from the 288 valid 4x4 sudoku solutions."""
    states = valid_solutions(device)
    idx = torch.randint(0, states.shape[0], (n,), generator=generator)
    return states[idx]


class SudokuValidSolutionsTarget(EnumerableTarget):
    """Uniform distribution over the 288 valid 4x4 sudoku solutions.

    An ordinary EnumerableTarget (only 288 states -- cheap), useful as the
    "reward-injected" component of a mixture (see SudokuMixtureTarget) since
    it reuses all the standard exact machinery (mean, natural_mean, G,
    log_partition, tilted_probs) for free.
    """

    def _build_states(self, device=None) -> torch.Tensor:
        return valid_solutions(device)

    def log_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device)

    def spec(self) -> dict:
        return {"kind": "sudoku_valid_solutions"}


class SudokuMixtureTarget:
    """pi_mix = (1 - p_inject) * base + p_inject * uniform_on_reward_set.

    "Reward injection": mix in extra mass on the 288 valid solutions so they
    appear often enough in a training set even when base's own success rate
    is tiny (e.g. base=SudokuRowsTarget(row_noise=0) puts only 288/331776 ~
    0.09% of its mass there). Every exact quantity below is computed exactly
    from the two components' own exact quantities via the mixture identity
    E_mix[f(X)] = (1-p)*E_A[f(X)] + p*E_B[f(X)] (in particular applied to
    f(x) = exp(<h,x>) and f(x) = x*exp(<h,x>) to get log_partition/
    natural_mean) -- neither component nor the mixture ever needs to
    enumerate the full {-1,1}^32.
    """

    def __init__(self, base: SudokuRowsTarget, p_inject: float, device=None, reward=None):
        self.base = base
        self.p_inject = float(p_inject)
        self.d = base.d
        self.device = device
        self.states = None  # never materialized -- see log_weights/.states note below
        self.probs = None
        # reward defaults to the unconditioned 288-solution target; condition_on_coords
        # passes in an already-conditioned (filtered) one instead, see below.
        self.reward = reward if reward is not None else SudokuValidSolutionsTarget(D, device)

    def sample(self, n: int, generator=None) -> torch.Tensor:
        inject = torch.rand(n, generator=generator) < self.p_inject
        out = torch.empty(n, self.d, device=self.device)
        n_inject = int(inject.sum())
        n_base = n - n_inject
        if n_base:
            out[~inject] = self.base.sample(n_base, generator=generator)
        if n_inject:
            # self.reward.sample(), NOT the module-level uniform_on_reward_set() --
            # the latter always draws from the full unconditioned 288 solutions,
            # silently ignoring a conditioned/filtered self.reward (see
            # condition_on_coords) and defeating the whole point of conditioning.
            # Identical in distribution to the old code for the default
            # (unconditioned) self.reward, since SudokuValidSolutionsTarget is
            # uniform over all 288 -- only the RNG call pattern differs.
            out[inject] = self.reward.sample(n_inject, generator=generator)
        return out

    def mean(self) -> torch.Tensor:
        return (1 - self.p_inject) * self.base.mean() + self.p_inject * self.reward.mean()

    @torch.no_grad()
    def natural_mean(self, h: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        h = torch.atleast_2d(h)
        logZA = self.base.log_partition(h, chunk=chunk)
        logZB = self.reward.log_partition(h, chunk=chunk)
        mA = self.base.natural_mean(h, chunk=chunk)
        mB = self.reward.natural_mean(h, chunk=chunk)
        logwA = math.log(1 - self.p_inject) + logZA
        logwB = math.log(self.p_inject) + logZB
        wts = torch.softmax(torch.stack([logwA, logwB], dim=-1), dim=-1)  # (batch, 2)
        return wts[:, 0:1] * mA + wts[:, 1:2] * mB

    @torch.no_grad()
    def log_partition(self, h: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """log Z_mix(h), exact via the same mixture identity as natural_mean:

        Z_mix(h) = (1-p_inject)*Z_base(h) + p_inject*Z_reward(h), so
        log Z_mix(h) = logsumexp([log(1-p_inject)+logZ_base(h),
        log(p_inject)+logZ_reward(h)]). This -- plus log_prob below -- is what
        lets a SudokuMixtureTarget be passed as `base` into
        SudokuRewardAugmented (see that class's row_noise>0 branch), closing
        the "known gap" noted in the project memory: reward-augmentation and
        reward-injection now compose.
        """
        h = torch.atleast_2d(h)
        logZA = self.base.log_partition(h, chunk=chunk)
        logZB = self.reward.log_partition(h, chunk=chunk)
        logwA = math.log(1 - self.p_inject) + logZA
        logwB = math.log(self.p_inject) + logZB
        return torch.logsumexp(torch.stack([logwA, logwB], dim=-1), dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact log pi_mix(x) for an explicit batch of candidate states
        (N, 32) -- the same mixture identity applied pointwise rather than to
        a partition function: pi_mix(x) = (1-p_inject)*pi_base(x) +
        p_inject*pi_reward(x). Delegates to self.reward.log_prob(x) (rather
        than a hardcoded 1/288-if-valid-else-0 formula) so this is correct
        both for the default unconditioned reward (uniform 1/288 on all 288
        solutions -- identical to the old hardcoded formula) and for a
        conditioned/filtered reward (uniform on just the solutions matching
        some given-cell constraint, -inf elsewhere -- see
        condition_on_coords). Needed by SudokuRewardAugmented's S/C
        decomposition to compute P_mix(valid) =
        exp(logsumexp(log_prob(valid_solutions))).
        """
        x = torch.atleast_2d(x)
        log_pA = self.base.log_prob(x)
        log_pB = self.reward.log_prob(x)
        term_a = math.log(1 - self.p_inject) + log_pA
        term_b = math.log(self.p_inject) + log_pB
        return torch.logsumexp(torch.stack([term_a, term_b], dim=-1), dim=-1)

    @torch.no_grad()
    def constraint_log_prob(self, coords, values) -> float:
        """log P_mix(x_coords = values), via the same mixture identity."""
        log_pA = self.base.constraint_log_prob(coords, values)
        log_pB = self.reward.constraint_log_prob(coords, values)
        log_wA = math.log(1 - self.p_inject) + log_pA
        log_wB = math.log(self.p_inject) + log_pB
        return float(torch.logsumexp(torch.tensor([log_wA, log_wB]), dim=0))

    def condition_on_coords(self, coords, values) -> "SudokuMixtureTarget":
        """Exact pi_mix(x | x_coords = values).

        This is NOT simply SudokuMixtureTarget(base.condition_on_coords(...),
        p_inject, reward=reward.condition_on_coords(...)) with the SAME
        p_inject -- conditioning on evidence Bayes-updates the mixing weight
        too, since the two components generally assign different
        probability to the evidence:

            pi_mix(x | E) = w_A' * pi_A(x | E) + w_B' * pi_B(x | E)
            w_B' = p * P_B(E) / [(1-p) * P_A(E) + p * P_B(E)]

        (derived by expanding pi_mix(x)*1[x in E] and regrouping). Setting
        the new instance's p_inject to this w_B' and its base/reward to each
        component's OWN conditioned (self-renormalised) version makes the
        existing natural_mean/log_partition formulas exactly correct with no
        other changes, since logZ_A'(0) = logZ_B'(0) = 0 for any properly
        self-normalised conditioned component.
        """
        coords = list(coords)
        values = list(values)
        log_pA = self.base.constraint_log_prob(coords, values)
        log_pB = self.reward.constraint_log_prob(coords, values)
        if log_pA == float("-inf") and log_pB == float("-inf"):
            raise ValueError("given constraint has probability zero under this mixture")
        log_wA = math.log(1 - self.p_inject) + log_pA
        log_wB = math.log(self.p_inject) + log_pB
        new_p_inject = float(torch.softmax(torch.tensor([log_wA, log_wB]), dim=0)[1])
        cond_base = self.base.condition_on_coords(coords, values)
        cond_reward = self.reward.condition_on_coords(coords, values)
        return SudokuMixtureTarget(cond_base, new_p_inject, device=self.device, reward=cond_reward)

    def G(self, u: torch.Tensor, a: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        from .noising import beta_of_a

        return self.natural_mean(beta_of_a(a) * u, chunk=chunk)

    def success_rate(self) -> float:
        """P_mix(valid) = (1 - p_inject) * P_base(valid) + p_inject * 1."""
        valid = valid_solutions(self.device)
        log_p_base_valid = torch.logsumexp(self.base.log_prob(valid), dim=0)
        p_base_valid = float(torch.exp(log_p_base_valid))
        return (1 - self.p_inject) * p_base_valid + self.p_inject * 1.0

    def effective_support(self, n_mc: int = 20_000, generator=None) -> float:
        """Monte Carlo estimate of exp(H[pi_mix]) = exp(-E[log pi_mix(X)]).

        Unlike SudokuRowsTarget.effective_support() this is NOT exact: entropy
        doesn't decompose additively once the two components' supports
        overlap (the 288 valid solutions have positive mass under both). This
        is a standard unbiased Monte Carlo entropy estimator (only subject to
        sampling noise, not a systematic approximation) -- fine as a meta.json
        diagnostic, not meant for precise comparisons.
        """
        x = self.sample(n_mc, generator=generator)
        log_pi_mix = self.log_prob(x)
        return float(torch.exp(-log_pi_mix.mean()))

    def spec(self) -> dict:
        return {"kind": "sudoku_mixture", "base": self.base.spec(), "p_inject": self.p_inject}

    def save(self, path) -> None:
        from pathlib import Path
        import json

        Path(path).write_text(json.dumps(self.spec(), indent=2))


# ------------------------------------------------ reward augmentation (d+1)
def r_sudoku_valid(x: torch.Tensor) -> torch.Tensor:
    """+1 if the decoded 4x4 grid is a valid sudoku solution, else -1.

    This is `r(x)` in `targets.RewardAugmented`'s sense: an arbitrary
    (nonlinear) function of x. It does NOT need to be linear -- that's the
    whole point of the augmentation trick (see targets.RewardAugmented's
    docstring): promoting r(x) to its own coordinate x* makes the *tilt*
    linear in the augmented representation, regardless of how r itself is
    computed from x. Sudoku's reward is handled exactly the same way as
    Ising's "majority"/"parity"/"exact:k" rewards -- there is nothing
    sudoku-specific about the mechanism, only about how r(x) is evaluated.
    """
    return torch.where(is_valid_sudoku_batch(x), 1.0, -1.0)


class SudokuRewardAugmented(RewardAugmented):
    """RewardAugmented(SudokuRowsTarget, "sudoku_valid") -- the standard,

    target-agnostic reward-tilt construction from targets.RewardAugmented,
    applied to sudoku exactly as it would be to Ising or Product. The ONLY
    sudoku-specific thing here is overriding `_build_states`: the default
    EnumerableTarget._build_states would call enumerate_cube(base.d + 1) =
    enumerate_cube(33), which exceeds MAX_ENUM_DIM=24. Since only pairs
    (x, x*) with x* == r(x) carry any mass anyway (see
    RewardAugmented.log_weights), and `base.states` is already the explicit
    (331776, 32) list of row-permutation states (row_noise=0), we can build
    the augmented (331776, 33) state list directly -- no full-cube
    enumeration needed, and no other change to RewardAugmented's exact
    machinery (tilt_direction, log_weights, success_rate, natural_mean,
    log_partition all apply completely unmodified).

    At `row_noise>0` the base support is the full 2^32 (base.states=None,
    row-factorized only), so no explicit augmented state list can be built
    either. Rather than falling back to a different mechanism, this is
    still the SAME linear tilt -- it's computed exactly via the S (288
    valid solutions) vs. C (everything else) decomposition:

        pi_base = (1 - p_S) * pi_C + p_S * pi_S,   p_S = P_base(valid)

    Since x* = r(x) is forced to +1 on S and -1 on C, tilting by
    h = (h_core, h_star) gives

        Z(h) = e^{h_star} * p_S * Z_S(h_core) + e^{-h_star} * (1-p_S) * Z_C(h_core)

    -- note the -h_star on the C branch (missing it, i.e. treating the
    reward as a {0,1}-indicator tilt instead of {-1,+1}, is a easy sign
    bug to make and was caught before landing here). Z_C(h_core)/
    m_C(h_core) are recovered algebraically from Z_A(h_core) (base's own
    row-factorized log_partition, 256-state marginals) and Z_S(h_core)
    (SudokuValidSolutionsTarget's 288-state enumeration) via
    alpha = p_S*Z_S/Z_A: m_C = (m_A - alpha*m_S)/(1-alpha),
    logZ_C = logZ_A + log(1-alpha) - log(1-p_S). Never enumerates beyond
    256 (per row) or 288 states, never touches 2^32 or 2^33.
    """

    def __init__(self, base: SudokuRowsTarget, device=None, solutions=None):
        if base.states is not None:
            # Materialized branch: build augmented states/log_w directly from
            # base's, bypassing RewardAugmented.__init__'s call to
            # self.log_weights() (which would call base.log_weights(core) --
            # a formula a conditioned/filtered base generally doesn't define).
            # Using base.log_w directly instead is equivalent: the two differ
            # only by an additive constant (base.log_weights(core) is the
            # PRE-shift log-weight, base.log_w is shifted by -max), and that
            # constant is removed again by this class's own "- lw.max()"
            # below -- so this reproduces the old code path exactly for an
            # unconditioned base, and correctly extends it to a conditioned
            # (filtered, states-still-materialized) one.
            self.base = base
            self.reward_name = "sudoku_valid"
            self.r = r_sudoku_valid
            self.d = base.d + 1
            self.device = device
            core = base.states.to(device) if device is not None else base.states
            star = r_sudoku_valid(core)
            self.states = torch.cat([core, star.unsqueeze(1)], dim=1)
            lw = base.log_w.to(device) if device is not None else base.log_w
            self.log_w = lw - lw.max()
            self.log_Z = torch.logsumexp(self.log_w, 0)
            self.probs = torch.exp(self.log_w - self.log_Z)
            return
        # row_noise > 0 (or a conditioned row-factorized/mixture base): fall
        # back to the exact S/C decomposition below.
        self.base = base
        self.reward_name = "sudoku_valid"
        self.r = r_sudoku_valid
        self.d = base.d + 1
        self.device = device
        self.states = None
        self.probs = None
        # solutions defaults to the unconditioned 288-solution target;
        # condition_on_coords passes in an already-conditioned one instead,
        # so the "S" branch of the S/C decomposition also respects whatever
        # given-cell constraint this instance was conditioned on.
        self.solutions = solutions if solutions is not None else SudokuValidSolutionsTarget(D, device)
        valid = valid_solutions(device)
        self.log_p_S = float(torch.logsumexp(base.log_prob(valid), dim=0))
        self.log_1m_p_S = math.log1p(-math.exp(self.log_p_S))
        self.p_S = math.exp(self.log_p_S)

    def condition_on_coords(self, coords, values) -> "SudokuRewardAugmented":
        """Exact pi_aug(x, x* | x_coords = values) for coords among the 32
        core bits (conditioning on the reward coordinate x* itself isn't
        supported -- it's a deterministic function of x, not an independent
        thing to fix). Conditions self.base and self.solutions separately
        (each via their own condition_on_coords) and rebuilds on top -- the
        S/C decomposition's algebra (log_p_S etc.) is recomputed fresh in
        __init__ from the now-conditioned base/solutions, so both the "S"
        (valid solutions) and "C" (everything else) branches correctly
        respect the given-cell constraint.
        """
        if self.states is not None:
            return super().condition_on_coords(coords, values)
        coords = list(coords)
        values = list(values)
        if any(int(c) >= D for c in coords):
            raise NotImplementedError("conditioning on the reward coordinate x* isn't supported")
        cond_base = self.base.condition_on_coords(coords, values)
        cond_solutions = self.solutions.condition_on_coords(coords, values)
        return SudokuRewardAugmented(cond_base, device=self.device, solutions=cond_solutions)

    def _build_states(self, device=None) -> torch.Tensor:
        core = self.base.states.to(device) if device is not None else self.base.states
        star = r_sudoku_valid(core)
        return torch.cat([core, star.unsqueeze(1)], dim=1)

    @staticmethod
    def _split_h(h: torch.Tensor):
        h = torch.atleast_2d(h)
        return h[:, :-1], h[:, -1]

    @torch.no_grad()
    def _logZ_C_and_alpha(self, h_core: torch.Tensor, chunk: int = 256):
        logZ_A = self.base.log_partition(h_core, chunk=chunk)
        logZ_S = self.solutions.log_partition(h_core, chunk=chunk)
        alpha = torch.exp(self.log_p_S + logZ_S - logZ_A).clamp(max=1.0 - 1e-6)
        logZ_C = logZ_A + torch.log1p(-alpha) - self.log_1m_p_S
        return logZ_C, logZ_S, alpha

    @torch.no_grad()
    def natural_mean(self, h: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        if self.states is not None:
            return super().natural_mean(h, chunk=chunk)
        h_core, h_star = self._split_h(h)
        logZ_C, logZ_S, alpha = self._logZ_C_and_alpha(h_core, chunk=chunk)
        m_A = self.base.natural_mean(h_core, chunk=chunk)
        m_S = self.solutions.natural_mean(h_core, chunk=chunk)
        m_C = (m_A - alpha.unsqueeze(-1) * m_S) / (1.0 - alpha).unsqueeze(-1)

        logit_S = h_star + self.log_p_S + logZ_S
        logit_C = -h_star + self.log_1m_p_S + logZ_C
        w = torch.softmax(torch.stack([logit_C, logit_S], dim=-1), dim=-1)
        m_core = w[:, 0:1] * m_C + w[:, 1:2] * m_S
        m_star = (w[:, 1] - w[:, 0]).unsqueeze(-1)
        return torch.cat([m_core, m_star], dim=-1)

    @torch.no_grad()
    def log_partition(self, h: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        if self.states is not None:
            return super().log_partition(h, chunk=chunk)
        h_core, h_star = self._split_h(h)
        logZ_C, logZ_S, _ = self._logZ_C_and_alpha(h_core, chunk=chunk)
        logit_S = h_star + self.log_p_S + logZ_S
        logit_C = -h_star + self.log_1m_p_S + logZ_C
        return torch.logsumexp(torch.stack([logit_C, logit_S], dim=-1), dim=-1)

    def mean(self) -> torch.Tensor:
        if self.states is not None:
            return super().mean()
        return self.natural_mean(torch.zeros(1, self.d, device=self.device))[0]

    def effective_support(self) -> float:
        """exp(H[pi_aug]).  x* = r(x) is a deterministic function of x, so

        H(X, X*) = H(X) exactly (adding a coordinate with zero conditional
        entropy given the rest doesn't change the entropy) -- this is exact
        for both branches, not just an approximation, and avoids needing
        `.probs` (None when row_noise>0, where the inherited
        EnumerableTarget.effective_support() would otherwise break).
        """
        return self.base.effective_support()

    def success_rate(self) -> float:
        if self.states is not None:
            return super().success_rate()
        return self.p_S

    def sample(self, n: int, generator=None) -> torch.Tensor:
        if self.states is not None:
            return super().sample(n, generator=generator)
        core = self.base.sample(n, generator=generator)
        star = r_sudoku_valid(core)
        return torch.cat([core, star.unsqueeze(1)], dim=1)

    def sample_tilted(self, h: torch.Tensor, n: int, generator=None) -> torch.Tensor:
        """Exact samples from q(x,x*) propto pi_aug(x,x*) exp(<h,(x,x*)>).

        h splits into h_core (tilt on the 32 raw sudoku bits x) and h_star
        (tilt on the reward coordinate x* = r(x)) -- see `_split_h`. Only
        h_core=0 is implemented (raises NotImplementedError otherwise), and
        that's not a missing feature so much as a documented scope match to
        this method's one and only caller:

          - `evaluate.py`'s `_typical_inputs` calls `sample_tilted(h_extra,
            n)` to draw x ~ q_tau *before any noise/observation exists* --
            i.e. samples of the tilted law itself, not a posterior given
            some observed u.
          - h_extra there is always `target.tilt_direction(tau)`, which by
            construction (see RewardAugmented.tilt_direction) is the zero
            vector with tau in the last (reward) slot -- so h_core is zero
            at this call site by construction, not by coincidence.

        Contrast with `natural_mean`, which genuinely needs arbitrary
        (nonzero) h_core: its caller (`_ground_truth`) evaluates it at
        h = beta(a)*u + h_extra, and u is the noisy observation of all 33
        coordinates, so beta(a)*u's first 32 entries are generically
        nonzero. That's a different function computing a posterior mean
        given noise, not a prior sample -- hence it has the full h_core
        algebra (see `_logZ_C_and_alpha`) that this method deliberately
        doesn't need.

        Uses a coin flip with the exact tilted success probability

            P(reward | tau) = p_S*e^tau / (p_S*e^tau + (1-p_S)*e^-tau)

        (NOT p_S*e^tau / (p_S*e^tau + (1-p_S)) -- that omits the e^-tau on
        the non-reward branch, i.e. treats the {-1,+1}-valued reward as a
        {0,1} indicator tilt, which is wrong here), then either samples
        uniformly among the 288 solutions, or draws from the base sampler
        and rejects-and-redraws on the rare (~p_S probability) chance it
        accidentally lands on a solution -- cheap since p_S is typically
        well under 1%.
        """
        if self.states is not None:
            return super().sample_tilted(h, n, generator=generator)
        h = torch.atleast_1d(h).reshape(-1)
        h_core, h_star = h[:-1], h[-1]
        if float(h_core.abs().max()) > 1e-8:
            raise NotImplementedError(
                "sample_tilted with nonzero h_core isn't implemented for the "
                "row_noise>0 branch (only h_core=0, i.e. tilt_direction(tau), "
                "is needed by evaluate.py's mode='ood' sampling step)"
            )
        tau = float(h_star)
        logit_S = tau + self.log_p_S
        logit_C = -tau + self.log_1m_p_S
        w_S = float(torch.softmax(torch.tensor([logit_C, logit_S]), dim=0)[1])

        is_solution = torch.rand(n, generator=generator) < w_S
        x_core = torch.empty(n, self.base.d)
        n_S = int(is_solution.sum())
        if n_S:
            x_core[is_solution] = self.solutions.sample(n_S, generator=generator)
        n_C = n - n_S
        if n_C:
            out = torch.empty(n_C, self.base.d)
            filled = 0
            while filled < n_C:
                cand = self.base.sample(n_C - filled, generator=generator)
                keep = cand[~is_valid_sudoku_batch(cand)]
                take = min(keep.shape[0], n_C - filled)
                out[filled : filled + take] = keep[:take]
                filled += take
            x_core[~is_solution] = out
        star = torch.where(is_solution, torch.ones(n), -torch.ones(n)).unsqueeze(1)
        return torch.cat([x_core, star], dim=1)
