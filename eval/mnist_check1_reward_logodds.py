"""MNIST check 1: class-conditional reward-tilt log-odds vs. tau.

Checkpoint : ckpt/mnist_v1_l24_d256x4_xf_sc_s8264_30k.pt
             (TransformerTilt, self-conditioned, d=6282 = 6272 Gray-coded
             pixel bits + a 10-bit one-hot class block)
Data       : data/mnist_v1 (60k train images/labels + 10k holdout, .npy --
             there is no train.csv and no target.json for this problem)

This is the MNIST analogue of ising_check3 / sudoku4x4_check2 / sudoku6_check2
/ sudoku9_check2. Four things differ from every previous problem's version of
this check, and all four change how the numbers must be read.


1. THE TILT IS A PATTERN, NOT AN AXIS -- so the log-odds slope is 4, not 2
-----------------------------------------------------------------------------
The reward block is a 10-coordinate one-hot class code in {-1,+1}, and
block-Lambda training stretches all ten coordinates together, so the tilt is
`h = tau * r_c` over the whole block (tilt.mnist.reward_tilt), never
`tau * e_c`. Over the support of pi the inner product <h, x> takes exactly two
values -- 10*tau when class(x) == c (<r_c, r_c> = n_extra) and 6*tau otherwise
(<r_c, r_c'> = n_extra - 4) -- so the tilted class marginal is EXACT and
closed-form, with no free parameters:

    logit pi_tau(c) = logit pi(c) + 4*tau            (`exact_class_prob`)

with pi(c) the training-set class frequency (meta.json's `class_counts`). The
factor 4 is a property of the tilt direction, not of the model; a single-axis
tilt would give 2. Consequence for this script: `common/reward_check.py::
build_tau_grid` hard-codes the slope-2 inversion `tau_98 = (logit(p) -
logit(p0)) / 2`, so it CANNOT be called directly here. It is reused anyway,
exactly and without modification, via the identity `tau_pattern =
tau_axis / 2`: ask it for a grid on [0, 2*tau_max] and halve the result. That
is not an approximation -- both the anchor and the two linspace segments are
linear in tau, so halving maps the slope-2 grid onto the slope-4 one exactly.


2. THE GROUND TRUTH IS MEASURED THROUGH AN IMPERFECT INSTRUMENT
-----------------------------------------------------------------------------
No exact operator exists at d=6282, so "did the sample land in class c" is
answered by an independently trained CNN (ckpt/mnist_cnn_scorer.pt,
98.74% holdout accuracy) rather than by a validity predicate. A classifier is
a noisy instrument, and at the tau values this check cares about its error
rate is the dominant term. Writing s_c = P(yhat=c | y=c) (sensitivity) and
e_c = P(yhat=c | y!=c) (off-class false-positive rate), what is observed is

    m = p * s_c + (1 - p) * e_c

for a true class-c rate p. Two things follow, both measured on the 10k
holdout by `scorer_confusion` below rather than assumed:

  * THE CURVE CEILINGS AT s_c, not at 1. Per class, logit(s_c) runs from 3.24
    (class 9, s=0.9623) to 5.93 (class 1, s=0.9974) -- a spread of 2.7 nats,
    so a curve pooled over classes without per-class correction would be
    comparing different ceilings. The exact theory crosses class 9's ceiling
    at tau ~ 1.36 and class 6's at tau ~ 1.57.
  * THE CEILING IS ITSELF UNCERTAIN. Each s_c comes from only ~1000 holdout
    images, giving a logit-scale SE of 0.17-0.58 -- comparable to, and at
    high tau larger than, the Clopper-Pearson binomial SE on the sampled
    points (~0.3 at n=256, m=0.95). Reporting only a CP interval here would
    understate the real uncertainty, because CP covers sampling noise in the
    generated batch and nothing else.

This script therefore FORWARD-MAPS the theory through the measured confusion
-- comparing observed m against `p_exact * s_c + (1 - p_exact) * e_c` -- in
preference to de-attenuating the data. The inverse (Rogan-Gladen,
`p_hat = (m - e_c) / (s_c - e_c)`) is recorded as a secondary column
(`p_rogan_gladen`, deliberately NOT clipped, so values above 1 stay visible as
the diagnostic they are) but is not plotted as primary: near the ceiling it
amplifies variance and routinely exceeds 1, which reads as a model pathology
when it is really an instrument artifact.

The two uncertainty sources are kept SEPARATE and never merged into one bar:
the sampled points carry a Clopper-Pearson CI (binomial noise), and the
forward-mapped theory curve carries its own band from the holdout counts for
(s_c, e_c) (instrument noise, `--n-boot` Jeffreys-posterior draws with common
random numbers across tau so the band is a coherent curve rather than
independent wiggle). A single combined bar would hide which term dominates,
which is the main thing this check needs to communicate at high tau.

CAVEAT worth stating plainly: s_c and e_c are measured on REAL MNIST holdout
images, while the thing being scored is generated and therefore
off-distribution. The forward map assumes the scorer's error rates transport
from one to the other. It corrects a known, measured bias; it does not
guarantee the corrected comparison is unbiased.


3. TWO CURVES, ONLY ONE OF WHICH IS SCORER-LIMITED
-----------------------------------------------------------------------------
Following sudoku9_check2's convention, two rates come off the SAME drawn batch
at no extra sampling cost:

  `model`  = P(argmax of the network's own 10-bit reward block == c). Read
             directly off the sample, so it has NO scorer ceiling and stays
             informative at every tau on the grid.
  `actual` = P(the CNN says the decoded image is class c). The headline, and
             the only one subject to (2) above.

Their divergence is the MNIST form of the "reward-hacking-like" decoupling
this project has documented for 6x6 and 9x9 sudoku (the network asserting a
reward it did not actually deliver); `xstar_agree` records it per draw, and
`onehot_ok` records how often the reward block is even a clean one-hot.


4. THE TAU GRID STOPS AT tau_95, NOT tau_98
-----------------------------------------------------------------------------
Because of (2), the informative window closes earlier here than for any other
problem in this folder. tau_95 = 1.285 and tau_98 = 1.522 (at the mean class
prior), while the lowest per-class ceiling is already crossed at tau ~ 1.36 --
so anchoring at 98% would place the grid's own reference point essentially on
top of the instrument's resolution limit. The grid is therefore `--n-before`
points on [0, tau_95] plus `--n-after` points on (tau_95, --tau-max], and
every point is flagged `scorer_limited` in the CSV when the forward-mapped
theory comes within 2 ceiling-SEs of logit(s_c). Those flagged points are
retained rather than dropped because they remain fully informative for the
`model` curve and for `onehot_ok`, neither of which involves the classifier;
they are just not evidence about `actual` one way or the other.

`logp_target_mean` (the scorer's mean log-softmax probability on the requested
class) is carried alongside as the auxiliary that keeps resolving after argmax
accuracy saturates -- see scripts/run_mnist_reward_tilt_check.py::classify's
note, which found the pilot pinned at 0.985 against a 0.9874 scorer ceiling.


self_cond_mode is a required argument
-----------------------------------------------------------------------------
Same reasoning as sudoku9_check2: the deprecated non-Markov "carry" mode
substantially outperformed the corrected "two_pass" default on the premier 9x9
checkpoint, and this folder's README is explicit that the finding must not be
assumed to transfer. This script is meant to be run under BOTH modes and the
results compared; nothing here presumes either is correct for MNIST.

Usage:
    python eval/mnist_check1_reward_logodds.py --self-cond-mode two_pass
    python eval/mnist_check1_reward_logodds.py --self-cond-mode carry
    # cheap end-to-end smoke test:
    python eval/mnist_check1_reward_logodds.py --self-cond-mode two_pass \\
        --n 32 --K 100 --n-before 2 --n-after 1 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # standalone subprocess run: no GUI backend available
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.checkpoints import load_premier  # noqa: E402
from common.reward_check import (  # noqa: E402
    build_tau_grid,
    clip_logit_yerr,
    clopper_pearson_ci,
    finite_axis_range,
    logit,
    logit_ci,
)
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.mnist import CORE, D, N_CLASSES, decode_bits, reward_tilt  # noqa: E402

# SmallCNN and classify are imported from the script that TRAINED the cached
# scorer artifact, so the architecture definition stays single-sourced with the
# state_dict it has to load (a local re-definition that drifts by one layer
# fails as a shape error at best, and silently mis-scores at worst).
from scripts.run_mnist_reward_tilt_check import SmallCNN, classify  # noqa: E402

CKPT = ROOT / "ckpt" / "mnist_v1_l24_d256x4_xf_sc_s8264_30k.pt"
DATA = ROOT / "data" / "mnist_v1"
SCORER = ROOT / "ckpt" / "mnist_cnn_scorer.pt"

SEP = 4.0  # log-odds slope of the one-hot PATTERN tilt (see docstring section 1)


# ---------------------------------------------------------------------------
# exact tilted class marginal
# ---------------------------------------------------------------------------
def exact_class_prob(prior: float, tau: float, sep: float = SEP) -> float:
    """EXACT class marginal of the tau-pattern-tilted target: no free parameters.

    logit pi_tau(c) = logit pi(c) + sep*tau, with sep=4 for the one-hot
    +/-1 pattern tilt (sep=2 would be a single-axis tilt). Derivation in this
    module's docstring, section 1; the same formula appears as
    scripts/run_mnist_reward_tilt_check.py::predicted_class_prob.
    """
    odds = prior / (1.0 - prior) * math.exp(sep * tau)
    return odds / (1.0 + odds)


def forward_map(p: float | np.ndarray, s_c: float, e_c: float):
    """True class rate p -> the rate an imperfect scorer with (s_c, e_c) reports.

    m = p*s_c + (1-p)*e_c. Monotone in p and bounded above by s_c, which is
    exactly the ceiling behaviour this check has to respect.
    """
    return p * s_c + (1.0 - p) * e_c


def rogan_gladen(m: float, s_c: float, e_c: float) -> float:
    """Inverse of `forward_map` -- the de-attenuated prevalence estimate.

    Recorded for reference only, and deliberately NOT clipped to [0, 1]: when
    the observed rate exceeds the scorer's own sensitivity this returns > 1,
    and that is informative (it means the measurement is at the instrument's
    resolution limit), whereas a clipped 1.000 would look like a real result.
    """
    denom = s_c - e_c
    return float("nan") if denom <= 0 else (m - e_c) / denom


# ---------------------------------------------------------------------------
# the measuring instrument
# ---------------------------------------------------------------------------
def scorer_confusion(scorer, datadir: Path, device) -> pd.DataFrame:
    """Per-class (s_c, e_c) with the raw counts they came from.

    Measured on the 10k HOLDOUT split, which the scorer was not trained on
    (see scripts/run_mnist_reward_tilt_check.py::get_scorer: it trains on
    images.npy and reports accuracy on holdout_images.npy). The raw counts are
    kept, not just the rates, because the bootstrap band needs them.
    """
    imgs = np.load(datadir / "holdout_images.npy")
    labels = np.load(datadir / "holdout_labels.npy").astype(np.int64)
    preds, _ = classify(scorer, imgs.reshape(len(imgs), -1), device)
    rows = []
    for c in range(N_CLASSES):
        pos = labels == c
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        k_tp = int((preds[pos] == c).sum())
        k_fp = int((preds[~pos] == c).sum())
        s_c, e_c = k_tp / n_pos, k_fp / n_neg
        rows.append({
            "cls": c, "s_c": s_c, "e_c": e_c,
            "n_pos": n_pos, "k_tp": k_tp, "n_neg": n_neg, "k_fp": k_fp,
            "ceiling_logit": logit(s_c, n_pos),
            # delta-method SE of logit(s_c): 1/sqrt(n p (1-p))
            "ceiling_logit_se": float(1.0 / math.sqrt(n_pos * s_c * (1.0 - s_c))),
        })
    return pd.DataFrame(rows)


def bootstrap_forward_band(p_exact: np.ndarray, row: pd.Series, n_boot: int,
                            rng: np.random.Generator, alpha: float = 0.05):
    """(lo, hi) on the forward-mapped theory curve from instrument uncertainty.

    Draws (s_c, e_c) from independent Jeffreys posteriors Beta(k+1/2,
    n-k+1/2) -- a proper interval at the k=n boundary, where a Wald SE would
    give 0 -- and pushes the whole tau curve through each draw with COMMON
    RANDOM NUMBERS (one (s, e) pair per bootstrap replicate, applied at every
    tau). Per-tau independent resampling would give a band that wiggles
    incoherently across tau, implying more information than the ~1000 holdout
    images per class actually carry.
    """
    s = rng.beta(row["k_tp"] + 0.5, row["n_pos"] - row["k_tp"] + 0.5, size=n_boot)
    e = rng.beta(row["k_fp"] + 0.5, row["n_neg"] - row["k_fp"] + 0.5, size=n_boot)
    draws = p_exact[None, :] * s[:, None] + (1.0 - p_exact[None, :]) * e[:, None]
    return (np.quantile(draws, alpha / 2, axis=0),
            np.quantile(draws, 1 - alpha / 2, axis=0))


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_class(df_c: pd.DataFrame, conf: pd.Series, prior: float, tau_anchor,
                args, K: int, out_path: Path, n_boot: int, rng,
                subfig=None) -> None:
    """One class's two-panel figure: log-odds scale and probability scale.

    `subfig`, if given, is a matplotlib SubFigure to draw into instead of a
    fresh standalone figure (and nothing is saved -- the caller owns the parent
    figure). This is what lets the 10-class montage and the 10 standalone PNGs
    come out of ONE plotting implementation: stitching the saved PNGs into a
    grid instead would have rasterised them and left two definitions of "what
    this panel looks like" to drift apart.
    """
    c = int(conf["cls"])
    tau_dense = np.linspace(0.0, float(df_c["tau"].max()), 400)
    p_dense = np.array([exact_class_prob(prior, float(t)) for t in tau_dense])
    m_dense = forward_map(p_dense, conf["s_c"], conf["e_c"])
    band_lo, band_hi = bootstrap_forward_band(p_dense, conf, n_boot, rng)

    logit_p_dense = np.array([logit(p) for p in p_dense])
    logit_m_dense = np.array([logit(m) for m in m_dense])
    logit_band_lo = np.array([logit(v) for v in band_lo])
    logit_band_hi = np.array([logit(v) for v in band_hi])

    standalone = subfig is None
    fig = plt.figure(figsize=(13, 5)) if standalone else subfig
    ax_l, ax_p = fig.subplots(1, 2)

    for ax, dense_exact, dense_fwd, blo, bhi, ceil in (
            (ax_l, logit_p_dense, logit_m_dense, logit_band_lo, logit_band_hi,
             conf["ceiling_logit"]),
            (ax_p, p_dense, m_dense, band_lo, band_hi, conf["s_c"])):
        ax.plot(tau_dense, dense_exact, "--", color="tab:blue", lw=1.2,
                label="exact tilted class marginal")
        ax.plot(tau_dense, dense_fwd, "-", color="tab:blue",
                label="exact, forward-mapped through scorer")
        ax.fill_between(tau_dense, blo, bhi, color="tab:blue", alpha=0.18,
                        label="scorer uncertainty (95%)")
        ax.axhline(ceil, ls=":", color="tab:red", lw=1.2,
                   label=f"scorer ceiling $s_c$={conf['s_c']:.4f}")

    y_lo, y_hi = finite_axis_range(logit_p_dense, df_c["logit_model"],
                                    df_c["logit_actual"], [conf["ceiling_logit"]])
    ax_l.set_ylim(y_lo, y_hi)
    ax_l.errorbar(df_c["tau"], df_c["logit_model"],
                  yerr=clip_logit_yerr(df_c["logit_model"], df_c["logit_model_ci_lo"],
                                       df_c["logit_model_ci_hi"], y_max=y_hi, y_min=y_lo),
                  fmt="o", capsize=3, color="tab:orange",
                  label="model: P(reward block argmax = c)")
    ax_l.errorbar(df_c["tau"], df_c["logit_actual"],
                  yerr=clip_logit_yerr(df_c["logit_actual"], df_c["logit_actual_ci_lo"],
                                       df_c["logit_actual_ci_hi"], y_max=y_hi, y_min=y_lo),
                  fmt="s", capsize=3, color="tab:green",
                  label="actual: P(scorer says c)")

    ax_p.errorbar(df_c["tau"], df_c["model"],
                  yerr=np.vstack([df_c["model"] - df_c["model_ci_lo"],
                                  df_c["model_ci_hi"] - df_c["model"]]),
                  fmt="o", capsize=3, color="tab:orange",
                  label="model: P(reward block argmax = c)")
    ax_p.errorbar(df_c["tau"], df_c["actual"],
                  yerr=np.vstack([df_c["actual"] - df_c["actual_ci_lo"],
                                  df_c["actual_ci_hi"] - df_c["actual"]]),
                  fmt="s", capsize=3, color="tab:green",
                  label="actual: P(scorer says c)")
    ax_p.set_ylim(-0.02, 1.02)

    # shade the region the classifier cannot resolve, per this class's own ceiling
    lim = df_c.loc[df_c["scorer_limited"], "tau"]
    if len(lim):
        for ax in (ax_l, ax_p):
            ax.axvspan(float(lim.min()), float(df_c["tau"].max()), color="0.85",
                       zorder=0, label="scorer-limited (actual curve only)")
    if tau_anchor is not None:
        for ax in (ax_l, ax_p):
            ax.axvline(tau_anchor, ls="--", color="gray", lw=1.0,
                       label=f"$\\tau_{{95}}$={tau_anchor:.3f}")

    ax_l.set_xlabel("tau (pattern-tilt strength)")
    ax_l.set_ylabel("log-odds  logit(P)")
    ax_l.set_title("log-odds scale (theory slope = 4)")
    ax_l.legend(fontsize=7, loc="lower right")
    ax_p.set_xlabel("tau (pattern-tilt strength)")
    ax_p.set_ylabel("P")
    ax_p.set_title("probability scale")
    ax_p.legend(fontsize=7, loc="lower right")

    if standalone:
        fig.suptitle(f"MNIST d={D} class {c} reward tilt  (prior={prior:.4f}, "
                     f"n={args.n}/tau, K={K}, self_cond_mode={args.self_cond_mode!r})")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    else:
        # the montage's own suptitle already carries n/K/mode, so the per-cell
        # heading only needs to say which class this is
        fig.suptitle(f"class {c}  (prior={prior:.4f})", fontsize=13)


def plot_pooled(df: pd.DataFrame, conf: pd.DataFrame, args, K: int,
                 tau_anchor, out_path: Path) -> None:
    """All ten classes pooled -- a summary panel, not a substitute for the per-class ones.

    Pooling is over an equal number of draws per class, so the pooled theory is
    the plain mean of the ten per-class forward-mapped rates (each class keeps
    its own s_c, so the ceiling spread is handled rather than averaged away
    before the fact). The pooled CP interval treats the 10*n draws as one
    binomial, which slightly understates width because the ten per-class rates
    are not identical -- adequate for a summary, which is why the per-class
    figures remain the primary output.
    """
    g = df.groupby("tau", as_index=False).agg(
        actual_hits=("actual_hits", "sum"), model_hits=("model_hits", "sum"),
        n=("n", "sum"), exact_fwd=("exact_fwd", "mean"), exact=("exact", "mean"))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    tau_dense = np.linspace(0.0, float(df["tau"].max()), 400)
    # only the classes actually swept -- `--classes` may be a subset, and
    # averaging in a class that was never drawn would bias the theory curve
    # against a measured rate that does not contain it.
    priors = df.drop_duplicates("cls").set_index("cls")["prior"]
    classes = sorted(int(c) for c in df["cls"].unique())
    fwd = np.mean([[forward_map(exact_class_prob(float(priors[c]), float(t)),
                                float(conf.loc[c, "s_c"]), float(conf.loc[c, "e_c"]))
                    for t in tau_dense] for c in classes], axis=0)
    raw = np.mean([[exact_class_prob(float(priors[c]), float(t)) for t in tau_dense]
                   for c in classes], axis=0)
    ax.plot(tau_dense, raw, "--", color="tab:blue", lw=1.2,
            label=f"exact (mean over {len(classes)} classes)")
    ax.plot(tau_dense, fwd, "-", color="tab:blue", label="exact, forward-mapped")
    for hits, color, marker, lab in (("model_hits", "tab:orange", "o", "model: reward-block argmax"),
                                      ("actual_hits", "tab:green", "s", "actual: scorer")):
        rate = g[hits] / g["n"]
        ci = np.array([clopper_pearson_ci(int(h), int(n)) for h, n in zip(g[hits], g["n"])])
        ax.errorbar(g["tau"], rate, yerr=np.vstack([rate - ci[:, 0], ci[:, 1] - rate]),
                    fmt=marker, capsize=3, color=color, label=lab)
    if tau_anchor is not None:
        ax.axvline(tau_anchor, ls="--", color="gray", lw=1.0, label=f"$\\tau_{{95}}$={tau_anchor:.3f}")
    ax.set_xlabel("tau (pattern-tilt strength)")
    ax.set_ylabel("P")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"MNIST reward tilt, all 10 classes pooled  (n={args.n}/class/tau, "
                 f"K={K}, {args.self_cond_mode})")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def finalize(df: pd.DataFrame, conf: pd.DataFrame, priors: np.ndarray, args,
              K: int, tau_anchor, out_dir: Path, suffix: str) -> None:
    """Everything after sampling: tau_eff, CSV rewrite, figures, summary.

    Split out of `main` so `--replot-only` can reach it with no GPU and no
    resampling. This step needs nothing but the results CSV and the scorer's
    confusion table, yet in the original single-block `main` it ran only after
    the FULL tau loop -- so the first `carry` run, killed during its 8th tau,
    left a perfectly good 70-row CSV with no figures at all and no `tau_eff`
    column. Any interrupted sweep can now be finished into publishable output
    without re-drawing a single sample.
    """
    base = df[df["tau"] == 0.0].set_index("cls")["logit_actual"].to_dict()
    df["tau_eff"] = [(r.logit_actual - base[r.cls]) / SEP if r.cls in base else np.nan
                     for r in df.itertuples()]
    df.to_csv(out_dir / f"check1_reward_logodds{suffix}.csv", index=False)

    classes = sorted(int(c) for c in df["cls"].unique())
    for c in classes:
        plot_class(df[df["cls"] == c].reset_index(drop=True), conf.loc[c],
                   float(priors[c]), tau_anchor, args, K,
                   out_dir / f"check1_reward_logodds{suffix}_cls{c}.png",
                   args.n_boot, np.random.default_rng(args.seed + c))
    plot_pooled(df, conf, args, K, tau_anchor,
                out_dir / f"check1_reward_logodds{suffix}_pooled.png")

    # all classes in one sheet: 2 columns x ceil(n/2) rows of SubFigures, each
    # holding that class's own two panels (so 5x2 cells = 20 axes for 10
    # classes). constrained_layout, not tight_layout -- the latter does not
    # handle nested subfigures.
    ncol = 2
    nrow = math.ceil(len(classes) / ncol)
    big = plt.figure(figsize=(13 * ncol, 5 * nrow), constrained_layout=True)
    cells = big.subfigures(nrow, ncol, squeeze=False)
    for i, c in enumerate(classes):
        plot_class(df[df["cls"] == c].reset_index(drop=True), conf.loc[c],
                   float(priors[c]), tau_anchor, args, K, None, args.n_boot,
                   np.random.default_rng(args.seed + c),
                   subfig=cells[i // ncol][i % ncol])
    for j in range(len(classes), nrow * ncol):        # blank any unused cell
        cells[j // ncol][j % ncol].subplots(1, 1).axis("off")
    big.suptitle(f"MNIST d={D} reward tilt, all {len(classes)} classes  "
                 f"(n={args.n}/tau, K={K}, self_cond_mode={args.self_cond_mode!r})",
                 fontsize=18)
    big.savefig(out_dir / f"check1_reward_logodds{suffix}_all.png", dpi=100)
    plt.close(big)

    print(f"\nsaved: {out_dir / f'check1_reward_logodds{suffix}.csv'}")
    print(f"saved: {len(classes)} per-class PNGs + "
          f"check1_reward_logodds{suffix}_pooled.png + "
          f"check1_reward_logodds{suffix}_all.png ({nrow}x{ncol} montage)")
    piv = df.pivot_table(index="tau", values=["exact", "exact_fwd", "actual", "model",
                                               "gap_fwd", "tau_eff", "xstar_agree",
                                               "onehot_ok", "logp_target_mean"],
                         aggfunc="mean")
    print("\nmean over classes:")
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    if df["scorer_limited"].any():
        lim = df.loc[df["scorer_limited"], "tau"].min()
        print(f"\nNOTE tau >= {lim:.3f} is scorer-limited for at least one class: the "
              f"`actual` column there cannot separate this checkpoint from a perfect "
              f"one. `model`, `onehot_ok` and `logp_target_mean` are still readable.")


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--scorer", default=str(SCORER))
    p.add_argument("--self-cond-mode", required=True, choices=["two_pass", "carry"],
                   help="which self-conditioning mode to sample with -- run BOTH and "
                        "compare, per this script's module docstring")
    p.add_argument("--target-p", type=float, default=0.95,
                   help="exact rate defining the grid anchor (0.95, not the other "
                        "problems' 0.98 -- see docstring section 4)")
    p.add_argument("--n-before", type=int, default=6, help="points on [0, tau_95]")
    p.add_argument("--n-after", type=int, default=3, help="points on (tau_95, tau_max]")
    p.add_argument("--tau-max", type=float, default=2.5,
                   help="upper end of the grid. NOT floor(sqrt(d))=79: the pattern "
                        "tilt's admissible per-coordinate magnitude is "
                        "sqrt(d/n_extra)=25.06 (tilt.mnist.LAMBDA_MAX_CEILING), and "
                        "the classifier stops resolving an order of magnitude before "
                        "even that")
    p.add_argument("--tau-list", type=float, nargs="+", default=None,
                   help="explicit tau values instead of the tau_95-anchored grid")
    p.add_argument("--classes", type=int, nargs="+", default=list(range(N_CLASSES)))
    p.add_argument("--n", type=int, default=256, help="draws per (tau, class)")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="bootstrap replicates for the scorer-uncertainty band")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--K", type=int, default=None,
                   help="override K_DEFAULTS['mnist']; filenames get a _K<K> suffix")
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "mnist"))
    p.add_argument("--tag", default=None, help="extra output-filename suffix")
    p.add_argument("--replot-only", action="store_true",
                   help="rebuild the figures (and any missing tau_eff column) from "
                        "the results CSV already in --out-dir, without drawing a "
                        "single sample. Use after an interrupted sweep, or to "
                        "restyle a figure. Same convention as "
                        "sudoku9_nearest_puzzle_cdf.py's --replot-only.")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K_default = K_DEFAULTS["mnist"]
    K = args.K if args.K is not None else K_default
    suffix = (f"_{args.self_cond_mode}"
              + ("" if K == K_default else f"_K{K}")
              + (f"_{args.tag}" if args.tag else ""))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datadir = Path(args.data)

    # --- the measuring instrument, before anything is sampled -----------------
    if not Path(args.scorer).exists():
        raise SystemExit(
            f"scorer not found at {args.scorer}. This eval deliberately does NOT "
            f"train one on the fly (scripts/run_mnist_reward_tilt_check.py will, "
            f"but a headline number should not silently depend on a freshly "
            f"trained instrument) -- point --scorer at the premier artifact.")
    ck = torch.load(args.scorer, map_location=device, weights_only=False)
    scorer = SmallCNN().to(device)
    scorer.load_state_dict(ck["state_dict"])
    scorer.eval()
    conf = scorer_confusion(scorer, datadir, device).set_index("cls", drop=False)
    conf.to_csv(out_dir / "scorer_confusion.csv", index=False)
    print(f"scorer: {Path(args.scorer).name}  holdout acc={ck['test_acc']:.4f}")
    print(conf[["cls", "s_c", "e_c", "ceiling_logit", "ceiling_logit_se"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    meta = json.loads((datadir / "meta.json").read_text())
    counts = np.asarray(meta["class_counts"], dtype=np.float64)
    priors = counts / counts.sum()

    if args.replot_only:
        csv = out_dir / f"check1_reward_logodds{suffix}.csv"
        if not csv.exists():
            raise SystemExit(f"--replot-only needs {csv}, which does not exist")
        df = pd.read_csv(csv)
        # take n/K/classes from the CSV rather than the CLI defaults: a replot
        # must describe the run that actually happened, not whatever the
        # current argument defaults happen to be
        args.n, K = int(df["n"].iloc[0]), int(df["K"].iloc[0])
        _, anchor_axis = build_tau_grid(float(priors.mean()), 2.0 * args.tau_max,
                                         target_p=args.target_p,
                                         n_before=args.n_before, n_after=args.n_after)
        tau_anchor = None if anchor_axis is None else anchor_axis / 2.0
        print(f"replot-only: {len(df)} rows, {df['tau'].nunique()} taus, "
              f"{df['cls'].nunique()} classes, n={args.n}, K={K}, "
              f"mode={df['self_cond_mode'].iloc[0]!r}")
        finalize(df, conf, priors, args, K, tau_anchor, out_dir, suffix)
        return

    # --- the model ------------------------------------------------------------
    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"\nloaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  "
          f"self_cond_mode={args.self_cond_mode!r}  device={device}")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- expected the "
                          "premier MNIST transformer/self-cond checkpoint")
    if shim.net.cfg.d != D:
        raise SystemExit(f"checkpoint d={shim.net.cfg.d} != tilt.mnist.D={D}")

    # --- tau grid (slope-4 via the slope-2 helper, see docstring section 1) ----
    if args.tau_list is not None:
        taus, tau_anchor = np.array(args.tau_list, dtype=float), None
    else:
        taus_axis, anchor_axis = build_tau_grid(
            float(priors.mean()), 2.0 * args.tau_max, target_p=args.target_p,
            n_before=args.n_before, n_after=args.n_after)
        taus = taus_axis / 2.0
        tau_anchor = None if anchor_axis is None else anchor_axis / 2.0
        print(f"\ntau_{int(args.target_p * 100)} (mean prior {priors.mean():.4f}) = "
              f"{tau_anchor:.4f}" if tau_anchor is not None else "\n(linspace fallback)")
    print(f"tau grid ({len(taus)} pts): {np.round(taus, 4)}")
    print(f"sampling: T={T}, K={K}, schedule={SCHEDULE!r}, n={args.n} per (tau, class), "
          f"{len(args.classes)} classes -> {len(taus) * len(args.classes)} batches")

    rng = np.random.default_rng(args.seed + 917)
    rows = []
    for i, tau in enumerate(taus):
        tau = float(tau)
        for c in args.classes:
            t0 = time.time()
            h0 = reward_tilt(tau, c)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                raw = sl(drift_arg, d=D, n=args.n, h=h0, T=T, K=K, schedule=SCHEDULE,
                         seed=args.seed + 1000 * c + i,
                         self_cond_mode=args.self_cond_mode)
            hard = decode(raw, drift_arg, h0)
            imgs, xstar_cls = decode_bits(hard)
            preds, probs = classify(scorer, imgs.reshape(len(imgs), -1), device)

            model_hits = int((xstar_cls == c).sum())
            actual_hits = int((preds == c).sum())
            model = model_hits / args.n
            actual = actual_hits / args.n
            m_lo, m_hi = clopper_pearson_ci(model_hits, args.n)
            a_lo, a_hi = clopper_pearson_ci(actual_hits, args.n)

            s_c, e_c = float(conf.loc[c, "s_c"]), float(conf.loc[c, "e_c"])
            p_exact = exact_class_prob(float(priors[c]), tau)
            exact_fwd = forward_map(p_exact, s_c, e_c)
            b_lo, b_hi = bootstrap_forward_band(np.array([p_exact]), conf.loc[c],
                                                 args.n_boot, np.random.default_rng(args.seed + c))
            # the classifier cannot separate the model from a perfect one once
            # the forward-mapped theory is within 2 ceiling-SEs of the ceiling
            scorer_limited = bool(
                logit(exact_fwd) > conf.loc[c, "ceiling_logit"]
                - 2.0 * conf.loc[c, "ceiling_logit_se"])

            rows.append({
                "tau": tau, "cls": c, "n": args.n, "K": K,
                "self_cond_mode": args.self_cond_mode, "prior": float(priors[c]),
                "exact": p_exact, "exact_fwd": exact_fwd,
                "exact_fwd_ci_lo": float(b_lo[0]), "exact_fwd_ci_hi": float(b_hi[0]),
                "model": model, "model_hits": model_hits,
                "model_ci_lo": m_lo, "model_ci_hi": m_hi,
                "actual": actual, "actual_hits": actual_hits,
                "actual_ci_lo": a_lo, "actual_ci_hi": a_hi,
                "gap_fwd": actual - exact_fwd,
                "p_rogan_gladen": rogan_gladen(actual, s_c, e_c),
                "s_c": s_c, "e_c": e_c, "scorer_limited": scorer_limited,
                "logit_exact": logit(p_exact), "logit_exact_fwd": logit(exact_fwd),
                "logit_model": logit(model, args.n),
                "logit_model_ci_lo": logit_ci(m_lo, m_hi)[0],
                "logit_model_ci_hi": logit_ci(m_lo, m_hi)[1],
                "logit_actual": logit(actual, args.n),
                "logit_actual_ci_lo": logit_ci(a_lo, a_hi)[0],
                "logit_actual_ci_hi": logit_ci(a_lo, a_hi)[1],
                # tau_eff: measured log-odds shift per unit tau, on the SEP=4
                # scale -- should equal tau for a faithful sampler. Filled in
                # after the sweep, once each class's own tau=0 row exists.
                "logp_target_mean": float(np.log(np.clip(probs[:, c], 1e-6, 1.0)).mean()),
                "xstar_agree": float((xstar_cls == preds).mean()),
                "onehot_ok": float((hard[:, CORE:].sum(1) == -(N_CLASSES - 2))
                                   .float().mean()),
                "mean_intensity": float(imgs.mean()),
                "scorer_acc": float(ck["test_acc"]), "seconds": time.time() - t0,
            })
            r = rows[-1]
            print(f"  [{i + 1}/{len(taus)}] tau={tau:6.3f} cls={c}  "
                  f"model={model:.3f} [{m_lo:.3f},{m_hi:.3f}]  "
                  f"actual={actual:.3f} [{a_lo:.3f},{a_hi:.3f}]  "
                  f"exact_fwd={exact_fwd:.3f}  gap={r['gap_fwd']:+.3f}  "
                  f"agree={r['xstar_agree']:.3f}  onehot={r['onehot_ok']:.3f}"
                  f"{'  [scorer-limited]' if scorer_limited else ''}  "
                  f"({r['seconds']:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / f"check1_reward_logodds{suffix}.csv",
                                  index=False)

    finalize(pd.DataFrame(rows), conf, priors, args, K, tau_anchor, out_dir, suffix)


if __name__ == "__main__":
    main()
