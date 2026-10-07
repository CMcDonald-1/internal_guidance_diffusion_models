"""Exact distance from a generated 9x9 grid to the nearest VALID sudoku solution.

Direct 9x9 port of `tilt/sudoku6_distance.py` -- see that module's docstring
for the full motivation (the validity reward is binary and saturates near
zero, hiding how CLOSE a failing grid actually is to a solution). Everything
here is the same idea at N=9: `cell_distance` (minimum cells that must change)
and `bit_distance` (minimum of the 324 analog bits that must flip, under
`tilt.sudoku9`'s 4-bits-per-cell encoding -- not simply 4 x cell_distance,
since changing a cell costs 1-4 bits depending on which digit it moves to).

Both are solved exactly with CP-SAT over the classic 9x9 all-different
row/column/box constraints -- the standard sudoku CSP, unlike 6x6's
non-square-box variant, so this is expected to solve at least as fast
per-grid as the 6x6 version despite the larger board.

Out-of-alphabet codes
---------------------
`decode_grid_batch` (see `tilt/sudoku9.py`) can emit 10-16: the seven unused
4-bit codes (k=9..15). A cell holding one of these can never agree with a
valid grid (always costs an edit), same as 6x6's 7/8 -- tracked separately as
`n_bad_symbol`.
"""
from __future__ import annotations

import numpy as np
from ortools.sat.python import cp_model

N = 9
N_DIGITS = 9
BOX_SIZE = 3
BOXES = [
    [(r, c) for r in range(r0, r0 + BOX_SIZE) for c in range(c0, c0 + BOX_SIZE)]
    for r0 in range(0, N, BOX_SIZE)
    for c0 in range(0, N, BOX_SIZE)
]

# 4-bit code for digit d is (d-1) in binary, per tilt.sudoku9.encode_grid_batch.
_POPCOUNT = [bin(k).count("1") for k in range(16)]


def _bit_cost(from_val: int, to_val: int) -> int:
    """Bits that differ between the 4-bit codes of two cell values.

    from_val may be 1..16 (10..16 are the seven wasted codes); to_val is 1..9.
    """
    return _POPCOUNT[(from_val - 1) ^ (to_val - 1)]


def _add_sudoku_constraints(m, y):
    for i in range(N):
        m.AddAllDifferent([y[i][j] for j in range(N)])
    for j in range(N):
        m.AddAllDifferent([y[i][j] for i in range(N)])
    for box in BOXES:
        m.AddAllDifferent([y[r][c] for r, c in box])


def min_distance(grid, metric: str = "cell", time_limit: float = 10.0):
    """Exact minimum distance from `grid` to a valid 9x9 sudoku solution.

    Parameters
    ----------
    grid : (9, 9) array of ints, values 1..16 (10..16 = wasted codes)
    metric : "cell" or "bit"
    time_limit : per-grid solver cap in seconds

    Returns
    -------
    dict with keys: distance, witness (9x9 array), n_bad_symbol, status
    """
    g = np.asarray(grid, dtype=int)
    if g.shape != (N, N):
        raise ValueError(f"expected (9, 9), got {g.shape}")
    if metric not in ("cell", "bit"):
        raise ValueError("metric must be 'cell' or 'bit'")

    n_bad = int(((g < 1) | (g > N_DIGITS)).sum())

    m = cp_model.CpModel()
    y = [[m.NewIntVar(1, N_DIGITS, f"y_{i}_{j}") for j in range(N)] for i in range(N)]
    _add_sudoku_constraints(m, y)

    cost_terms = []
    for i in range(N):
        for j in range(N):
            v = int(g[i][j])
            if metric == "cell":
                if 1 <= v <= N_DIGITS:
                    differ = m.NewBoolVar(f"d_{i}_{j}")
                    m.Add(y[i][j] == v).OnlyEnforceIf(differ.Not())
                    m.Add(y[i][j] != v).OnlyEnforceIf(differ)
                    cost_terms.append(differ)
                else:
                    # out-of-alphabet code can never match: constant cost 1.
                    pass
            else:
                costs = [_bit_cost(v, t) for t in range(1, N_DIGITS + 1)]
                idx = m.NewIntVar(0, N_DIGITS - 1, f"i_{i}_{j}")
                m.Add(idx == y[i][j] - 1)
                cvar = m.NewIntVar(min(costs), max(costs), f"c_{i}_{j}")
                m.AddElement(idx, costs, cvar)
                cost_terms.append(cvar)

    const_cost = n_bad if metric == "cell" else 0
    m.Minimize(sum(cost_terms))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = 1  # caller parallelises over grids
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"distance": None, "witness": None, "n_bad_symbol": n_bad,
                "status": s.StatusName(st)}

    w = np.array([[s.Value(y[i][j]) for j in range(N)] for i in range(N)], dtype=int)
    return {
        "distance": int(s.ObjectiveValue()) + const_cost,
        "witness": w,
        "n_bad_symbol": n_bad,
        "status": "OPTIMAL" if st == cp_model.OPTIMAL else "FEASIBLE",
    }


# ------------------------------------------------------------------ fast paths
def constraint_violations(grid) -> dict:
    """Cheap O(1) violation counts per constraint family. No solver.

    Useful as a screen (distance 0 iff all three are 0) and as the
    family-level breakdown: which of rows/cols/boxes a tilt actually fixes.
    """
    g = np.asarray(grid, dtype=int)

    def miss(vals):
        return N_DIGITS - len(set(int(v) for v in vals if 1 <= v <= N_DIGITS))

    rows = sum(miss(g[i, :]) for i in range(N))
    cols = sum(miss(g[:, j]) for j in range(N))
    boxes = sum(miss([g[r][c] for r, c in box]) for box in BOXES)
    return {"row_viol": rows, "col_viol": cols, "box_viol": boxes,
            "total_viol": rows + cols + boxes,
            "valid": rows == 0 and cols == 0 and boxes == 0}


def score_grid(grid, metrics=("cell",), time_limit: float = 10.0) -> dict:
    """Violation counts plus exact distance(s). Skips the solver on valid grids."""
    out = constraint_violations(grid)
    if out["valid"]:
        for mt in metrics:
            out[f"{mt}_distance"] = 0
        out["n_bad_symbol"] = 0
        out["status"] = "VALID"
        return out
    for mt in metrics:
        r = min_distance(grid, metric=mt, time_limit=time_limit)
        out[f"{mt}_distance"] = r["distance"]
        out["n_bad_symbol"] = r["n_bad_symbol"]
        out["status"] = r["status"]
    return out
