"""Canonical sampling-path parameters, shared by every problem's eval script.

Past evals in this repo used inconsistent (T, K, schedule) combinations
across problems and even across checks for the same problem (the
K-sweep findings motivated standardizing on "sstar" + a large K). This module is the single place that pins down the
values every eval/*.py script should import, so nothing drifts silently
between problems or between a script written this month and one written
next month.

T=20.0 was already the uniform default across every problem's existing
scripts (run_sudoku6/9_reward_tilt_check.py, run_mnist_reward_tilt_check.py,
etc.) -- it's a pure sampling-horizon parameter, not tied to how any
checkpoint was trained (no dataset's meta.json stores T; only beta_lo/
beta_hi, which are a training-time concern). SCHEDULE is what varied
inconsistently before; standardizing on "sstar" fixes that going forward.

K (Euler-Maruyama step count) is deliberately NOT one shared constant here:
harder/higher-dimensional problems (e.g. 9x9 sudoku) may need a larger K
than the Ising check uses, per the project's own K-sweep findings (the
sudoku6 K-sweep -- discretization error near t=T can
dominate at small K, and the K needed to reach the discretization-limited
ceiling grew with problem difficulty). K_DEFAULTS below records, per
problem, the K actually used for that problem's eval pass -- so it's still
looked up from one shared place (nothing silently drifts within a problem
across scripts/checks) without forcing every problem to the same value.
"""
from __future__ import annotations

T: float = 20.0
SCHEDULE: str = "sstar"

K_DEFAULTS: dict[str, int] = {
    "ising": 1000,
    "sudoku4x4": 1000,
    "sudoku6": 1000,
    "sudoku9": 1000,
    "mnist": 1000,
}
"""K used for each problem's eval pass. Add an entry here (don't hardcode K
inline in a new eval/*.py script) when starting a new problem, and update it
if that problem's own K is later revisited.

sudoku9=1000: `eval/sudoku9_k_sweep.py` swept K in
{1000,2000,4000,8000} at tau=3,5 on the premier checkpoint, under BOTH
self_cond_mode="two_pass" and "carry" (see tilt/sampler_selfcond.py and
Appendix B.4 of the paper for what that flag means). Neither mode showed a meaningful validity gain past K=1000:
carry's own actual-grid-validity plateaus almost immediately (K=1000->2000
+0.03, then flat through K=8000); two_pass stays flat-to-mildly-declining
across the whole range. This SUPERSEDES the older "9x9 may need K well past
8000" note that used to live here -- that claim traces back to
an earlier 9x9 study, which predates the carry->two_pass
fix and was almost certainly measured under the buggy carry default, so it
is not a trustworthy K recommendation independent of the direct sweep.

mnist=1000: MEASURED by `eval/mnist_k_sweep.py` (K in {25, 50,
100, 250, 500, 1000, 2000}, tau in {0, 0.55, 1.0}, classes {1, 9}, n=256,
two_pass). The elbow sits at K~250, far below the sudoku problems' -- as
predicted by the fact that their K-sensitivity lived in joint constraint
coordination (columns/boxes), which MNIST does not have. K=1000 is retained
rather than lowered to the elbow: it is comfortably past it, costs little on
the checks that matter, and the tau=0.55 curve had not quite flattened by 250.

The finding that matters for anyone reading MNIST numbers: THE CLASSIFIER
METRIC IS BLIND TO K. `actual` (P(scorer says c)) is flat within its CIs from
K=25 to K=2000 at every tau -- on that metric alone K=25 would look fine. The
scorer-free per-pixel-mean statistic says otherwise: measured in units of its
own real-data finite-sample floor, the generated batch's excess runs 4.4 SD
(K=25), 3.0 (50), 1.8 (100), ~0 (K>=250) at tau=1.0, with the same shape at
tau=0.55. Low K does not break class conditioning; it distorts the pixel
distribution, and it does so invisibly to the CNN. The classifier rate also
OVERSHOOTS the exact rate at low K (+0.07 at K=50, tau=0.55, decaying to
+0.02 by K=1000), consistent with a low-K sampler that is over-concentrated on
prototypical class members -- which is exactly the kind of error an accuracy
metric rewards.

Caveat: one batch per (K, tau, class) point, so the RMSE curve has no error
bar of its own -- the tau=0.55 point at K=2000 bounces back to 2.4 SD, which
is almost certainly that noise rather than a real regression. A multi-seed
rerun would be needed to call anything finer than "elbow near 250"."""
