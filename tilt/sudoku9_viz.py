"""9x9 sudoku infill-sample visualization: the partial ("given cells only")
puzzle next to one generated completion, side by side.

Direct 9x9 analogue of the plotting conventions already used at smaller scale
in `tilt/sudoku_viz.py` (4x4 fuzzy-puzzle) and `tilt/sudoku6_viz.py` (6x6
terminal-flag grids) -- same visual language (bold digits, orange = conflict),
adapted for this scale's purpose: showing what an infill test in
`tilt/sudoku9_infill.py` actually produced, not a confidence/fixation
trajectory.
"""
from __future__ import annotations

import torch

from .sudoku9 import GRID_COLS, GRID_ROWS, D, decode_grid_batch, find_conflicts_batch

SURFACE, INK, INK_MUTED, GRID_COLOR = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d8d4"
GIVEN_BG = "#ddeeff"
ORANGE = "#eb6834"


def _draw_grid(ax, grid, given_mask=None, conflict_mask=None, title=None):
    import matplotlib.pyplot as plt

    ax.set_facecolor(SURFACE)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            v = int(grid[r, c])
            given = bool(given_mask[r, c]) if given_mask is not None else False
            bad = bool(conflict_mask[r, c]) if conflict_mask is not None else False
            if given:
                ax.add_patch(plt.Rectangle((c, GRID_ROWS - 1 - r), 1, 1,
                                            facecolor=GIVEN_BG, edgecolor="none",
                                            zorder=0))
            if v != 0:
                txt = str(v) if 1 <= v <= 9 else "?"
                ax.text(c + 0.5, GRID_ROWS - 1 - r + 0.5, txt, ha="center",
                        va="center", fontsize=11,
                        color=ORANGE if bad else INK,
                        fontweight="bold" if (bad or given) else "normal")
    for i in range(GRID_ROWS + 1):
        lw = 1.6 if i % 3 == 0 else 0.6
        ax.plot([i, i], [0, GRID_ROWS], color=GRID_COLOR if i % 3 else INK_MUTED, lw=lw)
        ax.plot([0, GRID_COLS], [i, i], color=GRID_COLOR if i % 3 else INK_MUTED, lw=lw)
    ax.set_xlim(-0.05, GRID_COLS + 0.05)
    ax.set_ylim(-0.05, GRID_ROWS + 0.05)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, color=INK, pad=6)


def plot_infill_sample(given_cells, sol: torch.Tensor, hard_sample: torch.Tensor,
                       path: str | None = None, titles=None):
    """Two-panel figure: left = the partial puzzle (only `given_cells` filled
    in, values taken from `sol`), right = one generated completion.

    Args:
        given_cells: iterable of (row, col) 0-indexed cells that were fixed
            (the same list passed to `tilt.sudoku9_infill.build_infill_tilt`).
        sol: length->=D {-1,1} tensor -- the reference solution supplying the
            given cells' revealed values.
        hard_sample: length->=D {-1,1} tensor -- one decoded sample, e.g. one
            row of `infill_success_rate(...)["hard"]`.
        path: if given, saves the figure there (parent dirs created).
        titles: optional (left_title, right_title) override.

    Returns (fig, (ax_left, ax_right)).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    given_cells = list(given_cells)
    given_mask = torch.zeros(GRID_ROWS, GRID_COLS, dtype=torch.bool)
    for r, c in given_cells:
        given_mask[r, c] = True

    sol_core = sol[:D]
    gen_core = hard_sample[:D]

    sol_grid = decode_grid_batch(sol_core.unsqueeze(0))[0]
    puzzle_grid = torch.where(given_mask, sol_grid, torch.zeros_like(sol_grid))

    gen_grid = decode_grid_batch(gen_core.unsqueeze(0))[0]
    conflicts = find_conflicts_batch(gen_core.unsqueeze(0))[0]

    honors_per_cell = (
        torch.sign(gen_core.reshape(GRID_ROWS, GRID_COLS, 4))
        == torch.sign(sol_core.reshape(GRID_ROWS, GRID_COLS, 4))
    ).all(dim=-1)
    valid = bool(~conflicts.any())
    n_honored = int(honors_per_cell[given_mask].sum()) if given_cells else 0

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.6, 4.0), facecolor=SURFACE)
    lt, rt = titles or (
        f"given puzzle ({len(given_cells)}/{GRID_ROWS * GRID_COLS} cells)",
        f"generated ({'valid' if valid else 'INVALID'}, "
        f"given honored: {n_honored}/{len(given_cells)})",
    )
    _draw_grid(axl, puzzle_grid, given_mask=given_mask, title=lt)
    _draw_grid(axr, gen_grid, given_mask=given_mask, conflict_mask=conflicts, title=rt)
    fig.tight_layout()
    if path:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    return fig, (axl, axr)
