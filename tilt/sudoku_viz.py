"""Sudoku-specific visualizations of a tilt-mean operator's confidence.

Reconstructed from Section 6.3-6.4 of the writeup (the "fuzzy predict"
function and the coordinate-trajectory/fixation-event plot) -- the
originals were one-off console code. Everything here is built on top of
tilt.sudoku's {-1,1}^32 encoding (hi/lo bit per cell, see that module's
docstring) and tilt.sampler.sl_sample's trajectory support.

Two pieces:

  plot_fuzzy_puzzle       One snapshot: a 4x4 grid where a cell shows a
                          single bold digit if the network is confident
                          (|G| >= threshold) on both of its bits, or small
                          "pencil mark" candidate digits (one per bit
                          pattern still consistent with whichever bit(s)
                          are unresolved) otherwise.
  plot_fuzzy_trajectory   Runs one SL-sampler path, tracks all 32
                          coordinate trajectories of net.natural(h + x_t)
                          over time, and marks/plots the "fixation event"
                          for each cell -- the first time both of its bits
                          cross the confidence threshold together -- with
                          a fuzzy-puzzle snapshot at each such event.

As the paper notes, this premature per-step decoding is a diagnostic/fun
trick, not how the sampler is meant to be read -- only the final X_T is a
proper draw. It is also not "logical" in the human-solver sense: a cell
being fixed does not retroactively narrow the candidates of other cells
in its box/row/column, since the network never enforces that.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from .sampler import sl_sample
from .sudoku import D

_CONFIDENT_COLOR = "black"
_PENCIL_COLOR = "steelblue"
_DEFAULT_THRESHOLD = 0.75


def _digit_slot(v: int) -> tuple[int, int]:
    """digit (1-4) -> (hi, lo) bit pattern, i.e. (sub-row, sub-col) within a cell."""
    k = v - 1
    return (k >> 1) & 1, k & 1


def _cell_candidates(hi_val: float, lo_val: float, threshold: float) -> list[int]:
    """Digits (1-4) consistent with whichever bits are confidently resolved."""
    hi_bit = 1 if hi_val > 0 else 0
    lo_bit = 1 if lo_val > 0 else 0
    hi_confident = abs(hi_val) >= threshold
    lo_confident = abs(lo_val) >= threshold
    out = []
    for v in range(1, 5):
        vh, vl = _digit_slot(v)
        if hi_confident and vh != hi_bit:
            continue
        if lo_confident and vl != lo_bit:
            continue
        out.append(v)
    return out


def _cell_bits(r: int, c: int) -> tuple[int, int]:
    """(row, col) -> (index of hi bit, index of lo bit) in the length-32 vector."""
    base = 8 * r + 2 * c
    return base, base + 1


def _plot_fuzzy_cell(ax, r: int, c: int, hi_val: float, lo_val: float, threshold: float) -> None:
    cands = _cell_candidates(hi_val, lo_val, threshold)
    cx, cy = c + 0.5, 3 - r + 0.5
    if len(cands) == 1:
        ax.text(cx, cy, str(cands[0]), ha="center", va="center",
                fontsize=20, fontweight="bold", color=_CONFIDENT_COLOR)
        return
    for v in cands:
        vh, vl = _digit_slot(v)
        x = c + 0.22 + 0.56 * vl
        y = (3 - r + 1) - 0.22 - 0.56 * vh
        ax.text(x, y, str(v), ha="center", va="center", fontsize=10, color=_PENCIL_COLOR)


def plot_fuzzy_puzzle(G: torch.Tensor, ax=None, threshold: float = _DEFAULT_THRESHOLD,
                       title: str | None = None):
    """Draw one fuzzy-puzzle snapshot from a length->=32 confidence vector.

    `G` is any per-coordinate value in [-1, 1] -- typically
    net.natural(h + x) at some point x along (or at the end of) a sampler
    trajectory, or the exact operator's G for comparison. Only the first 32
    coordinates are used (the sudoku bits; any reward coordinate is
    ignored). Cells with both bits confident (|G| >= threshold) show a bold
    digit; otherwise the still-consistent candidate digits are shown as
    small pencil marks, positioned within the cell by their own (hi, lo)
    bit pattern.
    """
    G = torch.as_tensor(G).reshape(-1)[:D]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    for r in range(4):
        for c in range(4):
            hi_idx, lo_idx = _cell_bits(r, c)
            _plot_fuzzy_cell(ax, r, c, float(G[hi_idx]), float(G[lo_idx]), threshold)

    for i in range(5):
        lw = 2.5 if i % 2 == 0 else 0.75
        ax.axhline(i, color="black", lw=lw)
        ax.axvline(i, color="black", lw=lw)

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def _cell_fixation_step(Gs: torch.Tensor, r: int, c: int, threshold: float) -> int | None:
    """First step index (into Gs, shape (K+1, D)) where both of cell (r,c)'s

    bits are simultaneously confident, or None if that never happens along
    this trajectory.
    """
    hi_idx, lo_idx = _cell_bits(r, c)
    confident = (Gs[:, hi_idx].abs() >= threshold) & (Gs[:, lo_idx].abs() >= threshold)
    nz = confident.nonzero(as_tuple=True)[0]
    return int(nz[0]) if len(nz) else None


def fuzzy_fixation_trajectory(net, h: torch.Tensor, threshold: float = _DEFAULT_THRESHOLD,
                               T: float = 20.0, K: int = 200, schedule: str = "late",
                               seed: int = 0):
    """Run one SL-sampler path and track per-cell "fixation events".

    Returns (times, Gs, events):
      times  list of K+1 floats, t=0 through t=T (t=0 prepended to
             sl_sample's own K return-path entries).
      Gs     tensor (K+1, D), net.natural(h + x_k)[:D] at every step.
      events list of (step_idx, time, [(r, c, digit), ...]) -- one entry
             per distinct step at which at least one new cell first became
             confident on both bits, cells grouped by the step they first
             fixed at, in row-major order. A final synthetic event at the
             last step is always appended (labeled separately by the
             plotting function) so callers can show the fully-decoded grid
             even if no cell fixes exactly at t=T.
    """
    d = h.shape[-1]
    with torch.no_grad():
        _, path = sl_sample(net.natural, d=d, n=1, h=h, T=T, K=K,
                             schedule=schedule, seed=seed, return_path=True)
        times = [0.0] + [t for t, _ in path]
        Gs = torch.stack([net.natural(h.unsqueeze(0))[0]]
                          + [net.natural(h.unsqueeze(0) + x)[0] for _, x in path])
        Gs = Gs[:, :D].detach()

    fix_step = {}
    for r in range(4):
        for c in range(4):
            step = _cell_fixation_step(Gs, r, c, threshold)
            if step is not None:
                fix_step[(r, c)] = step

    by_step: dict[int, list[tuple[int, int]]] = {}
    for (r, c), step in fix_step.items():
        by_step.setdefault(step, []).append((r, c))

    events = []
    for step in sorted(by_step):
        cells = sorted(by_step[step])
        digits = [(r, c, int(torch.tensor(_cell_candidates(
            float(Gs[step, _cell_bits(r, c)[0]]), float(Gs[step, _cell_bits(r, c)[1]]),
            threshold)[0]))) for r, c in cells]
        events.append((step, times[step], digits))

    return times, Gs, events


def _wrap_fixed_label(digits: list[tuple[int, int, int]], per_line: int = 2) -> str:
    """"(r,c)=v, ..." entries, wrapped a few per line so long lists (many

    cells fixing at the same step) don't overflow into neighboring panels.
    """
    pairs = [f"({r},{c})={v}" for r, c, v in digits]
    lines = [", ".join(pairs[i : i + per_line]) for i in range(0, len(pairs), per_line)]
    return "\n".join(lines)


def _ema_smooth(G: torch.Tensor, t: torch.Tensor, tau: float) -> torch.Tensor:
    """Time-aware EMA: G_smooth[k] = G_smooth[k-1] + (1 - exp(-dt/tau)) * (G[k] -
    G_smooth[k-1]), using the ACTUAL dt between steps k-1 and k rather than
    assuming a uniform grid -- needed because "sstar" (this project's
    canonical schedule) is deliberately non-uniform, dense near t=0.

    Display-only: this is never used for fixation-event detection or the
    fuzzy-puzzle snapshots (both stay on the raw, unsmoothed `Gs`, which
    reflect the network's actual instantaneous confidence) -- only for how
    the top trajectory panel's lines are drawn, purely a visual smoothing of
    an otherwise fully-accurate high-K trajectory, chosen to visually match
    a reference figure from an earlier (lost) version of this plot.
    """
    out = torch.empty_like(G)
    out[0] = G[0]
    for k in range(1, len(t)):
        dt = float(t[k] - t[k - 1])
        w = 1.0 - torch.exp(torch.tensor(-dt / tau))
        out[k] = out[k - 1] + w * (G[k] - out[k - 1])
    return out


def plot_fuzzy_trajectory(net, h: torch.Tensor, threshold: float = _DEFAULT_THRESHOLD,
                           T: float = 20.0, K: int = 200, schedule: str = "late",
                           seed: int = 0, max_snapshots: int = 6,
                           smooth_tau: float | None = 0.05,
                           line_alpha: float = 0.35, line_width: float = 0.8):
    """Reproduce the "coordinate trajectories + cell fixation events" figure.

    Top panel: all 32 coordinate trajectories of net.natural(h + x_t) over
    the sampler path, with dashed lines at +/-threshold and a dotted
    vertical line at each cell-fixation event. Bottom row: a fuzzy-puzzle
    snapshot right after each event (capped to `max_snapshots`, spread
    evenly across the events if there are more than that), always ending
    with the final state at t=T labeled as fully solved/stable.

    As in the paper, this is a diagnostic/fun trick: fixation events are
    detected independently per cell and are not "logical" in the human-
    solver sense (a fixed cell does not constrain its box/row/column
    neighbors here) -- only the final X_T is a properly decoded draw.

    `smooth_tau` (display only, see `_ema_smooth`): EMA time-constant applied
    to the top panel's plotted lines, to visually match a reference figure
    from an earlier (lost) version of this plot without sacrificing sampling
    accuracy -- pass `None` to disable and draw the raw trajectories.
    `line_alpha`/`line_width` are the top panel's per-line style.
    """
    times, Gs, events = fuzzy_fixation_trajectory(
        net, h, threshold=threshold, T=T, K=K, schedule=schedule, seed=seed)

    # Pick which events to render as snapshots, capped at max_snapshots - 1
    # (the last slot is always reserved for the final t=T state).
    n_event_panels = max(max_snapshots - 1, 0)
    if len(events) > n_event_panels:
        idx = torch.linspace(0, len(events) - 1, n_event_panels).round().long().tolist()
        idx = sorted(set(idx))
    else:
        idx = list(range(len(events)))
    shown_events = [events[i] for i in idx]
    n_panels = len(shown_events) + 1

    fig = plt.figure(figsize=(3.3 * n_panels, 7.5))
    gs = fig.add_gridspec(2, n_panels, height_ratios=[1.3, 1.0], hspace=0.55, wspace=0.35)

    ax_traj = fig.add_subplot(gs[0, :])
    t_arr = torch.tensor(times)
    Gs_drawn = _ema_smooth(Gs, t_arr, smooth_tau) if smooth_tau is not None else Gs
    for i in range(D):
        ax_traj.plot(t_arr, Gs_drawn[:, i], lw=line_width, alpha=line_alpha)
    ax_traj.axhline(threshold, color="gray", lw=1, ls="--")
    ax_traj.axhline(-threshold, color="gray", lw=1, ls="--")
    for _, t_event, _ in shown_events:
        ax_traj.axvline(t_event, color="gray", lw=0.8, ls=":")
    ax_traj.set_ylim(-1.05, 1.05)
    ax_traj.set_xlabel("t")
    ax_traj.set_ylabel("net G value")
    ax_traj.set_title("All 32 coordinate trajectories (dotted lines = fixation events)")

    cum_fixed: dict[tuple[int, int], int] = {}
    for panel, (step, t_event, digits) in enumerate(shown_events):
        for r, c, v in digits:
            cum_fixed[(r, c)] = v
        ax = fig.add_subplot(gs[1, panel])
        plot_fuzzy_puzzle(Gs[step], ax=ax, threshold=threshold)
        ax.set_title(f"t={t_event:.3f}\nfixed: {_wrap_fixed_label(digits)}",
                     fontsize=7, pad=8)

    ax = fig.add_subplot(gs[1, -1])
    plot_fuzzy_puzzle(Gs[-1], ax=ax, threshold=threshold)
    ax.set_title(f"FINAL  t={times[-1]:.2f}\n(fully solved, stable)", fontsize=7, pad=8)

    fig.suptitle("Coordinate trajectories + cell fixation events, one SL SDE path")
    return fig, (ax_traj,)
