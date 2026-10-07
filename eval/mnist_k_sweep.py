"""MNIST: sweep the Euler-Maruyama step count K before fixing a canonical K.

Checkpoint : ckpt/mnist_v1_l24_d256x4_xf_sc_s8264_30k.pt
             (TransformerTilt, self-conditioned, d=6282)

`common/sample_config.py::K_DEFAULTS["mnist"]` is currently 1000, INHERITED
from sudoku9 and from `scripts/run_mnist_reward_tilt_check.py`'s own default
rather than established by measurement. Every other entry in that dict is
backed by a `<problem>_k_sweep.py`; this script exists so MNIST's is too.
The stakes are set by 6x6, where K=200 -> 1000 roughly TRIPLED reward-tilt
validity and halved the distance to the nearest valid solution: a whole
investigation's worth of "the tilt doesn't work" numbers turned out to be
partly discretization error. A K that is too small does not announce itself.


The ladder runs DOWN from the sudoku defaults, not up
-----------------------------------------------------------------------------
K in {25, 50, 100, 250, 500, 1000, 2000} -- an 80x span whose top is only 2x
the provisional default, where sudoku9's ladder started AT 1000 and ran to
16000. The expectation being tested is that MNIST tolerates a much smaller K
than either sudoku, and the reason is structural rather than a hunch about
image data being "easier": both sudoku sweeps localized their K-sensitivity to
JOINT CONSTRAINT COORDINATION specifically (9x9's `P(valid | all rows valid)`
moved 0.923 -> 0.941 from K=4000->8000 while the row channel sat flat), and
MNIST has essentially no such channel -- the Gray-code pixel map is a
bijection, so the only coordinates under a joint constraint at all are the 10
reward bits that must land on a one-hot. If that diagnosis is right, the
elbow here should sit far below 1000; if the curves are instead still climbing
at K=2000, the diagnosis is wrong and the ladder needs extending upward.

Anchoring the bottom at 25 is deliberate: it is far enough below anything this
project has used successfully that it SHOULD look visibly deficient. A flat
curve across a ladder whose bottom rung was never plausibly too small would be
evidence that the metrics are insensitive, not that the sampler has converged
-- so the low end is there to prove the measurement can detect a problem.


What to measure, given that MNIST has no validity predicate
-----------------------------------------------------------------------------
This is the structural difference from every previous K sweep in this folder.
Sudoku gave a free, exact, scorer-independent quality signal -- a decoded grid
either satisfies the constraints or it does not -- and sudoku9_k_sweep.py
could sharpen it further (`P(valid | all rows valid)`) to isolate the channel
where the K-sensitivity actually lived. MNIST has no such predicate: the
Gray-code map is a per-pixel BIJECTION onto 0..255, so every one of the 2^6272
core bit patterns decodes to a perfectly legal image. There is nothing to
check for internal consistency. Four metrics are therefore tracked, chosen so
that no single failure mode of any one of them can carry the conclusion:

1. `actual` -- P(the CNN scorer says class c). The direct analogue of grid
   validity, and the quantity check 1 reports, but measured through an
   imperfect instrument that ceilings at the per-class sensitivity s_c
   (0.9623-0.9974; see mnist_check1_reward_logodds.py's docstring section 2).
   The tau grid below is chosen to stay well clear of that ceiling.
2. `onehot_ok` -- fraction of draws whose 10-bit reward block decodes to a
   clean one-hot (sum == -8). The closest thing MNIST has to a structural
   constraint: it involves no classifier, and it is exactly the kind of
   joint coordination among a small set of coordinates that the 9x9 sweep
   found to be where discretization error concentrates. `xstar_agree` (does
   the block's argmax match what the scorer reads off the image?) is tracked
   beside it as the self-consistency counterpart.
3. `pixel_mean_rmse` -- a fully SCORER-FREE distributional check: RMSE, in
   0-255 intensity units, between the generated batch's per-pixel mean image
   and the exact tilted target's own per-pixel mean image. The reference is
   not hand-picked; it is
       sum_k pi_tau(k) * (real per-pixel mean of class k)
   with pi_tau the EXACT tilted class marginal (`tilted_class_dist`), so it is
   correct at every tau including tau=0, where the batch is a class mixture
   and the "requested class" is meaningless. Reported alongside
   `pixel_mean_rmse_null`, the same statistic computed on a batch of n REAL
   images drawn from the same pi_tau -- the finite-sample floor this
   statistic cannot go below. A ratio near 1 means the generated batch's
   first-order pixel statistics are indistinguishable from real data at this
   n; large ratios mean they are not. (First-order only: this says nothing
   about whether individual images are coherent, which is exactly why it is
   reported together with the scorer metrics rather than instead of them.)
4. `logp_target_mean` -- the scorer's mean log-probability on the requested
   class. Keeps resolving after argmax accuracy saturates.


Choosing tau: stay in the rising region
-----------------------------------------------------------------------------
sudoku9_k_sweep.py's reasoning applies with extra force here. The exact tilted
class marginal is `logit pi_tau(c) = logit pi(c) + 4*tau`, so with pi(c) ~ 0.1
it passes 0.95 by tau ~ 1.29 and is pinned against the scorer's own ceiling
shortly after. A sweep run up there would be measuring the instrument, not K.
The defaults are tau in {0.0, 0.55, 1.0}, i.e. exact rates of ~0.10, ~0.50 and
~0.86: tau=0 measures BASE sample quality with no tilt confound at all (where
metrics 2 and 3 do all the work), tau=0.55 sits at the point of maximum
binomial resolution, and tau=1.0 is high enough to stress the tilt while still
leaving ~0.14 of headroom below the exact target and more below s_c.

Classes default to {1, 9} -- the two extremes of measured scorer sensitivity
(s_1=0.9974, s_9=0.9623) -- purely to keep the sweep inside an hour. This is a
2-of-10 subsample, so it establishes K for THIS checkpoint under the assumption
that the K-elbow is not strongly class-dependent; `--classes` widens it if that
assumption is ever worth testing.


Cost and pairing
-----------------------------------------------------------------------------
Cost is ~n*K forward passes per point (2x for two_pass), so `_ProgressTracker`
(borrowed from sudoku9_k_sweep.py) weights its ETA by K rather than by point
count. Note that runs at different K are NOT paired: `sl_sample_selfcond`
draws fresh noise per step, so changing K changes the Brownian path entirely
even at a fixed seed. No pairing is needed for validity, though -- the n draws
within a batch are i.i.d., so the Clopper-Pearson interval is a correct
interval for the sampler's true rate at that K, and there is no extra
batch-level variance component beyond it.

Usage:
    python eval/mnist_k_sweep.py
    python eval/mnist_k_sweep.py --self-cond-mode carry
    python eval/mnist_k_sweep.py --K-list 250 500 1000 2000 4000 --n 256
    # cheap end-to-end smoke test:
    python eval/mnist_k_sweep.py --K-list 25 50 --taus 0 0.55 --n 32 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timedelta
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
from common.reward_check import clopper_pearson_ci, logit  # noqa: E402
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.mnist import CORE, D, N_CLASSES, decode_bits, reward_tilt  # noqa: E402

# Single-sourced from check 1 rather than re-derived: the exact tilted class
# marginal and the scorer's forward map must be the SAME functions both checks
# use, or a K recommendation could be justified against a subtly different
# ground truth than the check it is meant to serve.
from mnist_check1_reward_logodds import (  # noqa: E402
    SEP,
    exact_class_prob,
    forward_map,
    scorer_confusion,
)
from scripts.run_mnist_reward_tilt_check import SmallCNN, classify  # noqa: E402

CKPT = ROOT / "ckpt" / "mnist_v1_l24_d256x4_xf_sc_s8264_30k.pt"
DATA = ROOT / "data" / "mnist_v1"
SCORER = ROOT / "ckpt" / "mnist_cnn_scorer.pt"


class _ProgressTracker:
    """Per-point timer plus a K-weighted ETA (borrowed from sudoku9_k_sweep.py).

    Cost is roughly linear in K at fixed n, and this sweep spans a 16x range of
    K within one run, so an ETA based on the fraction of POINTS completed would
    be badly wrong early on. The running "seconds per K-unit" rate is not.
    """

    def __init__(self, points: list[tuple]):
        self.points = points
        self.total_weight = sum(pt[0] for pt in points)
        self.done_weight = 0
        self.t_start = time.time()
        print(f"planned: {len(points)} (K, tau, cls) points, "
              f"K-weighted work = {self.total_weight} K-units\n")

    def report(self, K: int, i: int, elapsed_this_point: float) -> None:
        self.done_weight += K
        elapsed_total = time.time() - self.t_start
        rate = elapsed_total / self.done_weight
        eta_s = (self.total_weight - self.done_weight) * rate
        pct = 100.0 * self.done_weight / self.total_weight
        finish = datetime.now() + timedelta(seconds=eta_s)
        print(f"    [{i}/{len(self.points)} points, {pct:5.1f}% of K-weighted work]  "
              f"this point={elapsed_this_point:.1f}s  elapsed={elapsed_total / 60:.1f}min  "
              f"ETA={eta_s / 60:.1f}min  (est. finish {finish:%H:%M:%S})")


def tilted_class_dist(priors: np.ndarray, tau: float, c: int,
                       sep: float = SEP) -> np.ndarray:
    """EXACT class marginal pi_tau over all 10 classes under the pattern tilt.

    `exact_class_prob` gives only the requested class's own probability; the
    per-pixel reference image below needs the whole distribution, since the
    other nine classes keep a combined mass of 1 - pi_tau(c) split in their
    original relative proportions. Weights are pi(k) * exp(sep * [k == c]),
    which reproduces `exact_class_prob` exactly at index c.
    """
    w = priors * np.exp(sep * tau * (np.arange(N_CLASSES) == c))
    return w / w.sum()


def pixel_mean_stats(gen_imgs: np.ndarray, class_mean: np.ndarray,
                      dist: np.ndarray, real_by_class: list[np.ndarray],
                      rng: np.random.Generator, n_null: int = 20) -> dict:
    """Scorer-free first-order distributional check against the exact target.

    Returns the RMSE (0-255 units) between the generated batch's per-pixel mean
    and the tilted target's own per-pixel mean, plus the SAME statistic for
    batches of real images of the same size drawn from the same class
    distribution. The null is not decoration: with n=256 images the per-pixel
    mean is itself noisy, so an RMSE of several intensity units can be entirely
    finite-sample and mean nothing about the model. Only the RATIO of the two
    is interpretable.

    The null is averaged over `n_null` independent real batches rather than
    taken from one. A single null draw is itself a random variable of the same
    order as the thing it is calibrating -- in a smoke run at n=32 it ranged
    over 7.3 to 14.3 across points, which would have made the ratio swing by
    2x for reasons having nothing to do with K. Its SD across replicates is
    returned too, so the ratio can be read against the noise in its own
    denominator.
    """
    n = len(gen_imgs)
    ref = (dist[:, None] * class_mean).sum(0)                      # (784,)
    gen = gen_imgs.reshape(n, -1).astype(np.float64).mean(0)

    nulls = []
    for _ in range(n_null):
        draw = rng.choice(N_CLASSES, size=n, p=dist)
        real = np.stack([real_by_class[k][rng.integers(len(real_by_class[k]))]
                         for k in draw]).astype(np.float64).mean(0)
        nulls.append(np.sqrt(((real - ref) ** 2).mean()))
    null = float(np.mean(nulls))

    rmse = float(np.sqrt(((gen - ref) ** 2).mean()))
    return {"pixel_mean_rmse": rmse, "pixel_mean_rmse_null": null,
            "pixel_mean_rmse_null_sd": float(np.std(nulls, ddof=1)) if n_null > 1 else float("nan"),
            "pixel_mean_rmse_ratio": rmse / null if null > 0 else float("nan"),
            "pixel_mean_r": float(np.corrcoef(gen, ref)[0, 1])}


def elbow(sub: pd.DataFrame, col: str, lo: str, hi: str) -> int | float:
    """Smallest K whose CI already contains the largest-K point estimate.

    A deliberately crude, stated-in-the-open rule for "where does this stop
    improving", not a convergence proof: it answers "is the extra compute
    buying anything this sweep can resolve at n draws", which is the actual
    decision K_DEFAULTS encodes. It can be fooled by a metric that drifts
    non-monotonically, so read it next to the plotted curve rather than alone.
    """
    sub = sub.sort_values("K")
    target = float(sub[col].iloc[-1])
    for _, r in sub.iterrows():
        if r[lo] <= target <= r[hi]:
            return int(r["K"])
    return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--scorer", default=str(SCORER))
    p.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.55, 1.0],
                   help="PATTERN-dial taus, chosen to sit in the rising region "
                        "well below the scorer ceiling -- see module docstring")
    p.add_argument("--K-list", type=int, nargs="+",
                   default=[25, 50, 100, 250, 500, 1000, 2000],
                   help="runs DOWN from the sudoku defaults on purpose -- see the "
                        "module docstring's ladder section")
    p.add_argument("--classes", type=int, nargs="+", default=[1, 9])
    p.add_argument("--n", type=int, default=256, help="draws per (K, tau, cls) point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--self-cond-mode", default="two_pass", choices=["two_pass", "carry"],
                   help="'carry' is deprecated (non-Markov drift, Appendix B.4) "
                        "but beat two_pass substantially on the premier 9x9 checkpoint, "
                        "and that finding has never been tested on MNIST. Output files "
                        "get a '_carry' suffix so the two never collide.")
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "mnist"))
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    suffix = ("" if args.self_cond_mode == "two_pass" else f"_{args.self_cond_mode}"
              ) + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datadir = Path(args.data)

    # --- instrument + real-data references ------------------------------------
    if not Path(args.scorer).exists():
        raise SystemExit(f"scorer not found at {args.scorer} (see "
                          f"mnist_check1_reward_logodds.py for why this is not "
                          f"trained on the fly)")
    ck = torch.load(args.scorer, map_location=device, weights_only=False)
    scorer = SmallCNN().to(device)
    scorer.load_state_dict(ck["state_dict"])
    scorer.eval()
    conf = scorer_confusion(scorer, datadir, device).set_index("cls", drop=False)

    priors = np.asarray(json.loads((datadir / "meta.json").read_text())["class_counts"],
                        dtype=np.float64)
    priors = priors / priors.sum()
    train_imgs = np.load(datadir / "images.npy")
    train_labels = np.load(datadir / "labels.npy").astype(np.int64)
    real_by_class = [train_imgs[train_labels == k].reshape(-1, 784) for k in range(N_CLASSES)]
    class_mean = np.stack([b.astype(np.float64).mean(0) for b in real_by_class])  # (10, 784)

    # --- model ----------------------------------------------------------------
    shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
    print(f"loaded {info['kind']} (arch={info['arch']}, selfcond={info['selfcond']})  "
          f"device={device}  self_cond_mode={args.self_cond_mode!r}")
    if not info["selfcond"]:
        raise SystemExit("this checkpoint is not self-conditioned -- expected the "
                          "premier MNIST transformer/self-cond checkpoint")
    if shim.net.cfg.d != D:
        raise SystemExit(f"checkpoint d={shim.net.cfg.d} != tilt.mnist.D={D}")
    print(f"d={D}  taus={args.taus}  K_list={args.K_list}  classes={args.classes}  "
          f"n={args.n}/point  T={T}, schedule={SCHEDULE!r}  "
          f"(K_DEFAULTS['mnist'] is currently {K_DEFAULTS['mnist']}, provisional)")
    for tau in args.taus:
        ex = [exact_class_prob(float(priors[c]), tau) for c in args.classes]
        print(f"  exact class rate at tau={tau}: "
              + ", ".join(f"cls{c}={e:.4f}" for c, e in zip(args.classes, ex)))

    # K outer so the cheap points run first (keeps the running rate meaningful)
    points = [(K, tau, c) for K in args.K_list for tau in args.taus for c in args.classes]
    tracker = _ProgressTracker(points)
    rng = np.random.default_rng(args.seed + 4471)

    rows = []
    for i, (K, tau, c) in enumerate(points, start=1):
        t0 = time.time()
        h0 = reward_tilt(tau, c)
        with warnings.catch_warnings():
            # the "carry is deprecated" warning is the point of the flag, not a surprise
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            raw = sl(drift_arg, d=D, n=args.n, h=h0, T=T, K=K, schedule=SCHEDULE,
                     seed=args.seed, self_cond_mode=args.self_cond_mode)
        hard = decode(raw, drift_arg, h0)
        imgs, xstar_cls = decode_bits(hard)
        preds, probs = classify(scorer, imgs.reshape(len(imgs), -1), device)

        actual_hits = int((preds == c).sum())
        actual = actual_hits / args.n
        a_lo, a_hi = clopper_pearson_ci(actual_hits, args.n)
        model_hits = int((xstar_cls == c).sum())
        model = model_hits / args.n
        m_lo, m_hi = clopper_pearson_ci(model_hits, args.n)
        oh_hits = int((hard[:, CORE:].sum(1) == -(N_CLASSES - 2)).sum())
        onehot_ok = oh_hits / args.n
        o_lo, o_hi = clopper_pearson_ci(oh_hits, args.n)

        s_c, e_c = float(conf.loc[c, "s_c"]), float(conf.loc[c, "e_c"])
        p_exact = exact_class_prob(float(priors[c]), tau)
        dist = tilted_class_dist(priors, tau, c)
        px = pixel_mean_stats(imgs, class_mean, dist, real_by_class, rng)

        elapsed = time.time() - t0
        rows.append({
            "K": K, "tau": tau, "cls": c, "n": args.n,
            "self_cond_mode": args.self_cond_mode,
            "exact": p_exact, "exact_fwd": forward_map(p_exact, s_c, e_c),
            "actual": actual, "actual_ci_lo": a_lo, "actual_ci_hi": a_hi,
            "model": model, "model_ci_lo": m_lo, "model_ci_hi": m_hi,
            "onehot_ok": onehot_ok, "onehot_ci_lo": o_lo, "onehot_ci_hi": o_hi,
            "xstar_agree": float((xstar_cls == preds).mean()),
            "logp_target_mean": float(np.log(np.clip(probs[:, c], 1e-6, 1.0)).mean()),
            "conf_max_mean": float(probs.max(axis=1).mean()),
            "logit_actual": logit(actual, args.n), "logit_exact": logit(p_exact),
            "mean_intensity": float(imgs.mean()),
            "s_c": s_c, "e_c": e_c, "elapsed_s": elapsed, **px,
        })
        r = rows[-1]
        print(f"  K={K:5d} tau={tau:4.2f} cls={c}  actual={actual:.4f} "
              f"[{a_lo:.4f},{a_hi:.4f}] (exact_fwd={r['exact_fwd']:.4f})  "
              f"onehot={onehot_ok:.4f}  agree={r['xstar_agree']:.4f}  "
              f"px_rmse={r['pixel_mean_rmse']:6.2f} (null {r['pixel_mean_rmse_null']:5.2f}, "
              f"ratio {r['pixel_mean_rmse_ratio']:4.1f})  logp={r['logp_target_mean']:7.3f}")
        tracker.report(K, i, elapsed)
        # written after every point: this sweep runs for the better part of an
        # hour, so an interruption should not cost the whole run
        pd.DataFrame(rows).to_csv(out_dir / f"k_sweep{suffix}.csv", index=False)

    df = pd.DataFrame(rows)

    # --- figure: 4 metric rows x one column per tau ---------------------------
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_tau = len(args.taus)
    fig, axes = plt.subplots(4, n_tau, figsize=(5.2 * n_tau, 13), sharex=True, squeeze=False)
    for col, tau in enumerate(args.taus):
        for j, c in enumerate(args.classes):
            sub = df[(df["tau"] == tau) & (df["cls"] == c)].sort_values("K")
            color = colors[j % len(colors)]

            ax = axes[0, col]
            ax.errorbar(sub["K"], sub["actual"],
                        yerr=np.vstack([sub["actual"] - sub["actual_ci_lo"],
                                        sub["actual_ci_hi"] - sub["actual"]]),
                        fmt="o-", capsize=3, color=color, label=f"cls {c}")
            ax.axhline(float(sub["exact_fwd"].iloc[0]), ls="--", lw=1.0, color=color,
                       label=f"cls {c} exact (fwd-mapped)")

            ax1 = axes[1, col]
            ax1.errorbar(sub["K"], sub["onehot_ok"],
                         yerr=np.vstack([sub["onehot_ok"] - sub["onehot_ci_lo"],
                                         sub["onehot_ci_hi"] - sub["onehot_ok"]]),
                         fmt="s-", capsize=3, color=color, label=f"cls {c} one-hot ok")
            ax1.plot(sub["K"], sub["xstar_agree"], "^:", color=color, alpha=0.6,
                     label=f"cls {c} xstar agree")

            ax2 = axes[2, col]
            ax2.plot(sub["K"], sub["pixel_mean_rmse"], "o-", color=color, label=f"cls {c}")
            ax2.plot(sub["K"], sub["pixel_mean_rmse_null"], ":", color=color, alpha=0.6,
                     label=f"cls {c} real-data floor")

            axes[3, col].plot(sub["K"], sub["logp_target_mean"], "o-", color=color,
                              label=f"cls {c}")

        for row in range(4):
            axes[row, col].set_xscale("log")
        axes[0, col].set_title(f"tau={tau}"
                               + ("  (untilted: read rows 2-3)" if tau == 0.0 else ""))
        axes[0, col].set_ylim(-0.02, 1.02)
        axes[1, col].set_ylim(-0.02, 1.02)
        axes[3, col].set_xlabel("K (Euler steps)")
        for row in range(4):
            axes[row, col].legend(fontsize=7)

    axes[0, 0].set_ylabel("P(scorer says c)")
    axes[1, 0].set_ylabel("reward-block structure")
    axes[2, 0].set_ylabel("per-pixel mean RMSE (0-255)")
    axes[3, 0].set_ylabel("mean log p(target class)")
    fig.suptitle(f"MNIST K-sweep  (n={args.n}/point, T={T}, schedule={SCHEDULE!r}, "
                 f"self_cond_mode={args.self_cond_mode!r})")
    fig.tight_layout()
    fig.savefig(out_dir / f"k_sweep{suffix}.png", dpi=150)

    # --- where does it stop improving -----------------------------------------
    print("\nsmallest K whose 95% CI already covers the largest-K estimate:")
    for tau in args.taus:
        for c in args.classes:
            sub = df[(df["tau"] == tau) & (df["cls"] == c)]
            k_a = elbow(sub, "actual", "actual_ci_lo", "actual_ci_hi")
            k_o = elbow(sub, "onehot_ok", "onehot_ci_lo", "onehot_ci_hi")
            print(f"  tau={tau:4.2f} cls={c}:  actual -> K={k_a}   onehot_ok -> K={k_o}")
    print("\nThis is a resolution statement at n=%d, not a convergence proof: a real "
          "but small K effect can hide inside these CIs. Read it with the plotted "
          "curves, and see `elbow`'s docstring." % args.n)
    print(f"\nsaved: {out_dir / f'k_sweep{suffix}.csv'}")
    print(f"saved: {out_dir / f'k_sweep{suffix}.png'}")


if __name__ == "__main__":
    main()
