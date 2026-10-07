"""Generic 'does the reward tilt hit the right log-odds at each tau' check.

Ground truth differs by problem, so this module only provides the sweep/report
scaffolding -- each eval_<target>.py supplies its own ground-truth callable and
sampling loop (the decoded quantity being scored differs: a reward bit's sign,
a full-grid validity predicate, or a CNN class prediction):

  - Ising / Sudoku 4x4: EXACT via `target.natural_mean(target.tilt_direction(tau))`
    -- both are RewardAugmented targets whose `natural_mean` is exact
    algebraically even when `target.states` is None (see the
    exact-conditioning and Sudoku-target notes in tilt/targets.py and tilt/sudoku.py) -- `exact_reward_rate_from_natural_mean`
    below.
  - Sudoku 6x6 / 9x9: EXACT closed form,
    `tilt.sudoku6.success_rate_exact` / `tilt.sudoku9.success_rate_exact`
    (row_noise, p_inject, tau) -- no enumeration, no Monte Carlo.
  - MNIST: EXACT closed form for the pattern-tilt class marginal,
    `scripts/run_mnist_reward_tilt_check.py::predicted_class_prob`.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta


def exact_reward_rate_from_natural_mean(target, tau: float) -> float:
    """P(x* = +1 | h = tilt_direction(tau)) from an EXACT natural_mean.

    natural_mean(h)[-1] is E[x* | h] in {-1,1}; P(x*=+1) = (1+E[x*])/2. Exact
    whenever `target.natural_mean` is exact (RewardAugmented over an
    EnumerableTarget, or a Sudoku4x4 SudokuMixtureTarget/SudokuRewardAugmented
    base), even when `target.states` is None (no state enumeration happened).

    `natural_mean` takes a BATCHED (n, d) natural parameter and returns a
    batched (n, d) mean (see tilt/targets.py::EnumerableTarget.natural_mean),
    so a single query is wrapped/unwrapped as a length-1 batch here.
    """
    h = target.tilt_direction(tau)
    m = target.natural_mean(h.unsqueeze(0))[0]
    return float((1.0 + m[-1]) / 2.0)


def logit(p: float, n: int | None = None) -> float:
    """Logit, clamped away from 0/1 by half a count if `n` (sample size) is given."""
    if n:
        lo, hi = 0.5 / n, 1.0 - 0.5 / n
        p = min(max(p, lo), hi)
    else:
        p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1.0 - p))


def logit_ci(p_lo: float, p_hi: float) -> tuple[float, float]:
    """TRUE log-odds of a probability CI's own bounds -- NOT clamped away from

    0/1 the way the point-estimate `logit()` above is. `p_lo=0` legitimately
    gives `-inf` and `p_hi=1` legitimately gives `+inf`: this is a real,
    sample-size-INDEPENDENT property of the Clopper-Pearson bound whenever
    x=0 or x=n (Beta(a, 0) / Beta(0, b) are degenerate distributions with all
    their mass at 1 / 0, so their quantile function is identically that value
    regardless of n -- see Brown/Cai/DasGupta 2001), not an approximation
    artifact to paper over with a fake finite number. Silently reusing the
    continuity-corrected `logit(p, n)` for a CI BOUND (rather than a point
    estimate) is wrong: it overwrites "the data cannot rule out p=1" with a
    fabricated "the data says p=0.9998", which is a different and false claim.

    Callers that need a finite number to actually draw a plot must clip the
    infinite endpoint themselves (e.g. to the axis's own y-limit) -- this
    function does not do that clipping, so the returned values can be
    +-inf and should be stored/printed as such; only the DRAWN error bar
    should be visually truncated, never the recorded CI.
    """
    lo = -math.inf if p_lo <= 0.0 else math.log(p_lo / (1.0 - p_lo))
    hi = math.inf if p_hi >= 1.0 else math.log(p_hi / (1.0 - p_hi))
    return lo, hi


def finite_axis_range(*arrays, pad_frac: float = 0.08) -> tuple[float, float]:
    """(y_min, y_max) spanning every FINITE value across all given arrays,

    padded by `pad_frac` of the span on each side. Use this to fix a
    log-odds axis's limits BEFORE calling `clip_logit_yerr`, rather than
    reading `ax.get_ylim()` back after plotting just one reference curve:
    a data point's own continuity-corrected estimate (not just its CI bound)
    can exceed whatever range that one curve happened to establish (e.g. the
    exact-operator sampler's point estimate landing above the exact
    closed-form theory curve's own endpoint) -- clipping against too small a
    range then makes `hi_clipped < center`, a negative "yerr" matplotlib
    rejects outright. Passing every series that will actually be plotted
    (the theory curve plus every point-estimate series) avoids that.
    """
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * pad_frac if hi > lo else max(abs(hi), 1.0) * pad_frac
    return lo - pad, hi + pad


def clip_logit_yerr(center, lo, hi, y_max: float, y_min: float | None = None):
    """Build a matplotlib-ready (2, N) `yerr` array from possibly +-infinite

    CI bounds (see `logit_ci`), clipping to `[y_min, y_max]` so a drawn error
    bar terminates visibly at the plot's own edge instead of trying to extend
    to a literal infinity matplotlib can't render. This ONLY affects what
    gets drawn -- callers should keep storing/printing the true (possibly
    infinite) `lo`/`hi` values separately (e.g. in a results CSV) rather than
    overwriting them with this function's clipped output, which is a display
    device, not a corrected estimate.

    `y_min` defaults to `-y_max` (symmetric clipping), which is fine for
    log-odds plots centered near 0; pass an explicit value otherwise.
    """
    center = np.asarray(center, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if y_min is None:
        y_min = -y_max
    lo_clipped = np.maximum(lo, y_min)
    hi_clipped = np.minimum(hi, y_max)
    return np.vstack([center - lo_clipped, hi_clipped - center])


def clopper_pearson_ci(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI: (p_lo, p_hi) such that P(X>=x|p_lo)=P(X<=x|p_hi)=alpha/2.

    Inverts the binomial CDF via the Beta-binomial identity, so coverage is
    guaranteed >= 1-alpha for any n, p -- unlike a Wald interval
    (p +- z*sqrt(p(1-p)/n)), which under-covers badly as p -> 0 or 1, exactly
    the regime high-tau reward-tilt points sit in. Needs the raw hit count
    `x`, not just the rate `x/n` -- callers should keep the count around
    rather than only the derived probability.
    """
    lo = 0.0 if x == 0 else float(_beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(_beta.ppf(1 - alpha / 2, x + 1, n - x))
    return lo, hi


def build_tau_grid(p0: float, tau_max: float, target_p: float = 0.98,
                    n_before: int = 5, n_after: int = 5):
    """Two-segment tau grid anchored at tau_98, not a

    plain `linspace(0, tau_max, n)`: a flat grid over-samples the tail, where
    the exact success rate is already near 0 or 1 and this check's resolution
    is limited by sample size (see logit_ci's docstring) rather than by tau
    itself. `tau_98` is the tau at which the EXACT rate first reaches
    `target_p` (default 98%), via the closed-form log-odds inversion
    `tau_98 = (logit(target_p) - logit(p0)) / 2` -- exact for any reward
    tilt obeying `logit(p_tau) = logit(p0) + 2*tau` (every reward-augmented
    target in this project: Ising/Sudoku4x4 via
    `exact_reward_rate_from_natural_mean`, Sudoku6/9 via
    `tilt.sudoku6/9.success_rate_exact`).

    Returns `(taus, tau_98_or_None)`: `n_before` evenly-spaced points on
    `[0, tau_98]` (the informative rising region) plus `n_after` more on
    `(tau_98, tau_max]` (the tail) -- `n_before + n_after` points total.
    Falls back to a single `linspace(0, tau_max, n_before + n_after)` (and
    returns `tau_98=None`, so callers know not to draw a tau_98 marker) if
    `tau_98 >= tau_max`, which happens whenever `p0` is high enough / `d` is
    small enough that 98% is reached past this problem's own `tau_max`
    ceiling -- e.g. Ising's `floor(sqrt(13))=3 < tau_98~=3.48` -- since the
    two-segment split would otherwise degenerate (an empty or backwards
    "after" segment).
    """
    tau_98 = (logit(target_p) - logit(p0)) / 2.0
    if tau_98 >= tau_max:
        return np.linspace(0.0, tau_max, n_before + n_after), None
    before = np.linspace(0.0, tau_98, n_before)
    after = np.linspace(tau_98, tau_max, n_after + 1)[1:]
    return np.concatenate([before, after]), tau_98


def run_tau_sweep(taus: list[float], measure_fn: Callable[[float], dict],
                   sep: float = 2.0) -> pd.DataFrame:
    """Run `measure_fn(tau) -> {"exact": p, "model": p, "n": n, ...}` at each tau.

    Adds `gap = model - exact` and `tau_eff = (logit(model) - logit(model@tau=0))
    / sep` -- comparable across checkpoints in a way a raw rate is not, and the
    diagnostic signature of a broken/saturated tilt is `tau_eff` flat in tau
    (see tilt/sudoku6.py / tilt/sudoku9.py's `tau_eff` and this project's
    sudoku6/9 reward-tilt-check scripts). `sep` is the log-odds scaling of the
    tilt actually used: 2.0 for a single-axis reward tilt on a {-1,1}
    coordinate (Ising, Sudoku 4x4/6x6/9x9), 4.0 for MNIST's one-hot PATTERN
    tilt (see tilt/mnist.py's module docstring).
    """
    rows = []
    base_logit = None
    for tau in taus:
        row = measure_fn(tau)
        row["tau"] = tau
        row["gap"] = row["model"] - row["exact"]
        n = row.get("n")
        lg = logit(row["model"], n)
        if tau == 0.0 or base_logit is None:
            base_logit = lg
            row["tau_eff"] = 0.0
        else:
            row["tau_eff"] = (lg - base_logit) / sep
        rows.append(row)
        print(f"  tau={tau:>5g}  exact={row['exact']:.4f}  model={row['model']:.4f}  "
              f"gap={row['gap']:+.4f}  tau_eff={row['tau_eff']:.3f}")
    return pd.DataFrame(rows)
