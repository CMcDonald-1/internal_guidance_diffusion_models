"""MNIST: draw a grid of class-conditional samples for visual inspection.

The qualitative companion to `mnist_check1_reward_logodds.py`, in the same
spirit as `eval/sudoku9_infill_figure.py`: check 1 reports whether the class
rate matches the exact tilted marginal, which says nothing at all about
whether the images look like digits. A 5x5 grid answers the question check 1
structurally cannot.

Each panel is labelled with the CNN scorer's own reading of that image, in
green when it matches the requested class and red when it does not -- so the
figure doubles as a per-sample view of the same `actual` rate check 1
aggregates, and a miss is visible as a picture rather than a number. Where
the network's own reward block disagrees with the scorer, the panel also
shows what the block claimed ("x*=k"), which is the per-sample form of check
1's `xstar_agree` column.

DEFAULT TAU IS THAT CLASS'S OWN tau_95, not a round number. The exact tilted
marginal is `logit pi_tau(c) = logit pi(c) + 4*tau` (see
mnist_check1_reward_logodds.py's docstring for why the slope is 4 and not 2),
so the tau delivering a 95% class rate depends on the class's own prior and
is `(logit 0.95 - logit pi(c)) / 4` -- about 1.27-1.30 across the ten
classes. Anything much beyond that buys no additional class purity and is
past where the scorer can verify the result anyway.

self_cond_mode defaults to two_pass here, unlike check 1 where it is
required: `eval/mnist_check1_reward_logodds.py` run under both modes found
two_pass tracks the exact marginal closely (tau_eff within 0.02-0.05 of tau
up to tau~1) while carry overshoots by ~25% through the rising region. That
makes two_pass the right default for an illustrative figure. `--self-cond-mode
carry` still works, and is worth a look precisely because carry's samples
were MORE self-consistent (xstar_agree ~0.98 vs ~0.91-0.96) despite being
distributionally wrong -- whether that shows up visually is an open question
this script can answer.

Usage:
    python eval/mnist_sample_figure.py --cls 7
    python eval/mnist_sample_figure.py --cls 7 --tau 2.0 --rows 4 --cols 8
    python eval/mnist_sample_figure.py --cls 3 --self-cond-mode carry
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # standalone subprocess run: no GUI backend available
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.checkpoints import load_premier  # noqa: E402
from common.sample_config import K_DEFAULTS, SCHEDULE, T  # noqa: E402

from tilt.mnist import CORE, D, N_CLASSES, decode_bits, reward_tilt  # noqa: E402

from mnist_check1_reward_logodds import exact_class_prob  # noqa: E402
from scripts.run_mnist_reward_tilt_check import SmallCNN, classify  # noqa: E402

CKPT = ROOT / "ckpt" / "mnist_v1_l24_d256x4_xf_sc_s8264_30k.pt"
DATA = ROOT / "data" / "mnist_v1"
SCORER = ROOT / "ckpt" / "mnist_cnn_scorer.pt"


def tau_for_rate(prior: float, target_p: float = 0.95, sep: float = 4.0) -> float:
    """Exact inversion of the tilted class marginal: tau reaching `target_p`.

    Closed form, no search: logit p_tau = logit prior + sep*tau.
    """
    return (math.log(target_p / (1 - target_p)) - math.log(prior / (1 - prior))) / sep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(CKPT))
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--scorer", default=str(SCORER))
    p.add_argument("--cls", type=int, required=True, choices=range(N_CLASSES),
                   help="digit class to condition on")
    p.add_argument("--tau", type=float, default=None,
                   help="pattern-tilt strength; default is this class's own tau_95")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--self-cond-mode", default="two_pass", choices=["two_pass", "carry"])
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results" / "mnist"))
    p.add_argument("--tag", default=None)
    p.add_argument("--real", action="store_true",
                   help="draw REAL holdout images of this class instead of sampling "
                        "the model, rendered through the identical code path -- the "
                        "reference panel for judging generated grids")
    p.add_argument("--save-npy", action="store_true",
                   help="also write the raw uint8 images next to the figure")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = args.K if args.K is not None else K_DEFAULTS["mnist"]
    n = args.rows * args.cols
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    priors = np.asarray(json.loads((Path(args.data) / "meta.json").read_text())
                        ["class_counts"], dtype=np.float64)
    priors = priors / priors.sum()
    tau = args.tau if args.tau is not None else tau_for_rate(float(priors[args.cls]))

    ck = torch.load(args.scorer, map_location=device, weights_only=False)
    scorer = SmallCNN().to(device)
    scorer.load_state_dict(ck["state_dict"])
    scorer.eval()

    if args.real:
        # Real holdout digits, deliberately rendered by the SAME code below --
        # identical layout, cmap, vmin/vmax, scorer labelling and border colouring.
        # A reference panel drawn by a separate ad hoc script would differ in some
        # styling detail, and a side-by-side "do the generated ones look like
        # these?" judgement would then be partly a judgement about matplotlib.
        him = np.load(Path(args.data) / "holdout_images.npy")
        hlb = np.load(Path(args.data) / "holdout_labels.npy").astype(np.int64)
        pool = np.flatnonzero(hlb == args.cls)
        pick = np.random.default_rng(args.seed).choice(pool, size=n, replace=False)
        imgs = him[pick].reshape(n, 28, 28)
        xstar_cls = None          # real data has no reward block to disagree with
        print(f"class {args.cls}  REAL holdout images  n={n} drawn from "
              f"{len(pool)} available  (seed={args.seed})")
        preds, probs = classify(scorer, imgs.reshape(len(imgs), -1), device)
        hit = preds == args.cls
        print(f"scorer says class {args.cls}: {hit.sum()}/{n} = {hit.mean():.3f}  "
              f"-- this is the SCORER's own error rate on real data, i.e. the "
              f"ceiling any generated grid is being judged against")
    else:
        shim, info, sl, decode, needs_net, drift_arg = load_premier(args.ckpt, device)
        if shim.net.cfg.d != D:
            raise SystemExit(f"checkpoint d={shim.net.cfg.d} != tilt.mnist.D={D}")

        print(f"class {args.cls}  tau={tau:.4f}"
              f"{'' if args.tau is not None else ' (this class'+chr(39)+'s tau_95)'}  "
              f"n={n}  K={K}  mode={args.self_cond_mode!r}  device={device}")
        h0 = reward_tilt(tau, args.cls)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            raw = sl(drift_arg, d=D, n=n, h=h0, T=T, K=K, schedule=SCHEDULE,
                     seed=args.seed, self_cond_mode=args.self_cond_mode)
        hard = decode(raw, drift_arg, h0)
        imgs, xstar_cls = decode_bits(hard)
        preds, probs = classify(scorer, imgs.reshape(len(imgs), -1), device)

        hit = preds == args.cls
        onehot = (hard[:, CORE:].sum(1) == -(N_CLASSES - 2)).numpy()
        expected = exact_class_prob(float(priors[args.cls]), tau)
        print(f"scorer says class {args.cls}: {hit.sum()}/{n} = {hit.mean():.3f}  "
              f"(exact tilted marginal {expected:.3f}, and this n is far too small for "
              f"that comparison to be evidence -- see check 1 for the measured version)")
        print(f"reward block a clean one-hot: {onehot.sum()}/{n}   "
              f"block argmax agrees with scorer: {(xstar_cls == preds).sum()}/{n}")

    fig, axes = plt.subplots(args.rows, args.cols,
                             figsize=(1.45 * args.cols, 1.62 * args.rows))
    for i, ax in enumerate(np.ravel(axes)):
        ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        label = f"{preds[i]}"
        if xstar_cls is not None and xstar_cls[i] != preds[i]:
            # the network's own reward block claimed a different class than the
            # scorer reads off the image it drew -- check 1's xstar_agree, per sample
            label += f"  (x*={xstar_cls[i]})"
        ax.set_title(label, fontsize=8, color=("tab:green" if hit[i] else "tab:red"),
                     pad=2)
        for sp in ax.spines.values():
            sp.set_edgecolor("tab:green" if hit[i] else "tab:red")
            sp.set_linewidth(1.6)

    head = (f"REAL MNIST holdout, class {args.cls}  (n={n}, seed={args.seed})"
            if args.real else
            f"MNIST samples, class {args.cls}  (tau={tau:.3f}, K={K}, "
            f"{args.self_cond_mode})")
    fig.suptitle(f"{head}\nscorer agrees on {hit.sum()}/{n}; "
                 f"panel title = scorer's reading", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = ((f"samples_cls{args.cls}_real" if args.real else
             f"samples_cls{args.cls}_tau{tau:.3f}_{args.self_cond_mode}")
            + (f"_{args.tag}" if args.tag else ""))
    fig.savefig(out_dir / f"{stem}.png", dpi=160)
    print(f"\nsaved: {out_dir / f'{stem}.png'}")
    if args.save_npy:
        np.save(out_dir / f"{stem}.npy", imgs)
        print(f"saved: {out_dir / f'{stem}.npy'}")


if __name__ == "__main__":
    main()
