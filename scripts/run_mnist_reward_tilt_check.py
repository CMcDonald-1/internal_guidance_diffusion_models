"""Score class-conditional MNIST sampling: tau ladder x 10 classes.

No exact operator exists at d=6282, so the exact-TV scoring of the sudoku
scripts is replaced by the guidance-literature standard: an independently
trained CNN classifier scores whether tilted samples land in the requested
class. The three diagnostics mirror run_sudoku9_reward_tilt_check.py:

  cls_acc       P(classifier says c | tilted toward c) -- the headline.
  tau_eff       Pattern-tilt separation is e^{4 tau} per unit tau, so
                tau_eff = (logit p_c(tau) - logit p_c(0)) / 4, per class,
                using the SAME checkpoint's tau=0 run as the base rate. The
                broken-run signature is tau_eff flat in tau (sudoku finding).
  xstar_agree   Does the model's OWN reward block (argmax) agree with the
                class the classifier sees in the grid it drew? The 6x6
                "asserts reward over an invalid grid" decoupling, ported.
  onehot_ok     Fraction of draws whose reward block decodes to a clean
                one-hot (sum == -8 after sign rounding).

THE TILT IS THE PATTERN, tau * r_c (all ten coordinates), never tau * e_c --
block-Lambda training never manufactures asymmetric reward-block geometry.
tilt.mnist.reward_tilt builds it.

The scorer trains itself on first use (small CNN, ~99% test acc, minutes) and
is cached at --scorer. Its TEST accuracy is printed and stored in the CSV so
a bad scorer can't silently masquerade as a bad diffusion model.

Defaults carry the sudoku9 lessons: --K 1000 (K was the dominant error term),
--schedule sstar (the drift varies along s*, not t).

Usage:
    python scripts/run_mnist_reward_tilt_check.py \
        --ckpt ckpt/mnist_v1_pilot_l24_xf49_d256x4_s8264_30k.pt \
        --data data/mnist_v1 --taus 0,0.5,1,2,4 --n 500 \
        --out results/mnist_v1_pilot_tilt.csv --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tilt.load_ckpt import load_tilt_checkpoint, sample_fns
from tilt.mnist import CORE, D, N_CLASSES, decode_bits, reward_tilt


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = torch.flatten(x, 1)
        return self.fc2(F.relu(self.fc1(x)))


def get_scorer(path: Path, datadir: Path, device) -> tuple[SmallCNN, float]:
    net = SmallCNN().to(device)
    if path.exists():
        ck = torch.load(path, map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        return net, ck["test_acc"]
    print(f"[scorer] {path} not found -- training a fresh one", flush=True)
    xi = np.load(datadir / "images.npy").astype(np.float32) / 255.0
    yi = np.load(datadir / "labels.npy").astype(np.int64)
    xh = np.load(datadir / "holdout_images.npy").astype(np.float32) / 255.0
    yh = np.load(datadir / "holdout_labels.npy").astype(np.int64)
    x = torch.from_numpy(xi).view(-1, 1, 28, 28)
    y = torch.from_numpy(yi)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    net.train()
    for epoch in range(3):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = F.cross_entropy(net(x[idx].to(device)), y[idx].to(device))
            loss.backward()
            opt.step()
        print(f"[scorer] epoch {epoch + 1} done", flush=True)
    net.eval()
    with torch.no_grad():
        xt = torch.from_numpy(xh).view(-1, 1, 28, 28)
        preds = torch.cat([net(xt[i:i + 1024].to(device)).argmax(1).cpu()
                           for i in range(0, len(xt), 1024)])
    acc = float((preds == torch.from_numpy(yh)).float().mean())
    print(f"[scorer] test accuracy {acc:.4f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "test_acc": acc}, path)
    return net, acc


@torch.no_grad()
def classify(scorer: SmallCNN, imgs: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """(argmax preds, softmax probs).

    The probs exist because ARGMAX ACCURACY CEILINGS AT THE SCORER'S OWN TEST
    ACCURACY (0.9874 for the 3-epoch CNN). The pilot hit 0.985 at tau=4, i.e.
    indistinguishable from perfect, which makes cls_acc -- and any tau_eff
    built from it -- useless for comparing checkpoints or K values in that
    regime. mean log p(target) keeps resolving past the point where argmax
    saturates, so it is the column to read when cls_acc is pinned near 1.
    """
    x = torch.from_numpy(imgs.astype(np.float32) / 255.0).view(-1, 1, 28, 28)
    logits = torch.cat([scorer(x[i:i + 1024].to(device)).cpu()
                        for i in range(0, len(x), 1024)])
    return logits.argmax(1).numpy(), torch.softmax(logits, dim=1).numpy()


# ---------------------------------------------------------------------------
# net-on-device shim (verbatim pattern from run_sudoku9_reward_tilt_check.py)
# ---------------------------------------------------------------------------
class _OnDevice:
    def __init__(self, net, device):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()

    def _wrap(self, fn):
        def call(u, *args, **kw):
            u = u.to(self.device)
            args = tuple(v.to(self.device) if torch.is_tensor(v) else v for v in args)
            kw = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                  for k, v in kw.items()}
            return fn(u, *args, **kw).detach().cpu()
        return call

    @property
    def natural(self):
        return self._wrap(self.net.natural)

    @property
    def natural_self_cond(self):
        return self._wrap(self.net.natural_self_cond)


def predicted_class_prob(prior: float, tau: float, sep: float = 4.0) -> float:
    """EXACT class marginal of the tau-pattern-tilted target -- the analog of
    sudoku9.success_rate_exact, with no free parameters.

    The reward block is the one-hot class code, so for h = tau * r_c the inner
    product <h, x> takes exactly two values over the support of pi:

        10*tau   if class(x) == c        (<r_c, r_c>   = n_extra)
         6*tau   otherwise               (<r_c, r_c'>  = n_extra - 4)

    so pi_tau(c) = pi(c) e^{sep*tau} / [pi(c) e^{sep*tau} + (1 - pi(c))] with
    sep = 4 for one-hot +/-1 codes. Equivalently logit pi_tau(c) = logit pi(c)
    + sep*tau -- which is why tau_eff (the measured logit shift over sep) should
    equal tau exactly for a faithful sampler, and why a tau_eff that bends is
    the saturation signature.

    NOTE the single-axis tilt h = tau * e_c would give sep = 2 instead. The
    factor is a property of the tilt direction, not of the model.
    """
    odds = prior / (1.0 - prior) * math.exp(sep * tau)
    return odds / (1.0 + odds)


def _logit(p: float, n: int) -> float:
    """Empirical logit, clamped away from 0/1 by the rule of half a count."""
    lo, hi = 0.5 / n, 1.0 - 0.5 / n
    p = min(max(p, lo), hi)
    return float(np.log(p / (1.0 - p)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/mnist_v1")
    p.add_argument("--taus", default="0,0.5,1,2,4",
                   help="PATTERN-dial taus (tau_pattern = tau_axis / 2)")
    p.add_argument("--n", type=int, default=500, help="samples per (tau, class)")
    p.add_argument("--T", type=float, default=20.0)
    p.add_argument("--K", type=int, default=1000)
    p.add_argument("--schedule", default="sstar")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scorer", default="ckpt/mnist_cnn_scorer.pt")
    p.add_argument("--out", default="results/mnist_reward_tilt.csv")
    p.add_argument("--save-samples-dir", default=None,
                   help="if set, writes <dir>/tau{t}_c{c}.npy raw uint8 images "
                        "for the memorisation check and figures")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    device = torch.device(a.device) if a.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    taus = [float(t) for t in a.taus.split(",")]
    datadir = Path(a.data)

    meta = json.loads((datadir / "meta.json").read_text())
    counts = np.asarray(meta["class_counts"], dtype=np.float64)
    priors = counts / counts.sum()

    scorer, scorer_acc = get_scorer(Path(a.scorer), datadir, device)
    net, info = load_tilt_checkpoint(a.ckpt)
    if getattr(net.cfg, "d", None) != D:
        raise SystemExit(f"checkpoint d={getattr(net.cfg, 'd', None)} != {D}")
    shim = _OnDevice(net, device)
    sl, decode, needs_net = sample_fns(info)
    drift_arg = shim if needs_net else shim.natural

    rows = []
    base_logit = {}          # class -> logit p_c at tau = 0, filled on first tau
    base_acc = {}            # class -> raw p_c at tau = 0, for pred_base
    for tau in taus:
        for c in range(N_CLASSES):
            t0 = time.time()
            h0 = reward_tilt(tau, c)
            raw = sl(drift_arg, d=D, n=a.n, h=h0, T=a.T, K=a.K,
                     schedule=a.schedule, seed=a.seed + 1000 * c + int(tau * 16))
            hard = decode(raw, drift_arg, h0)
            imgs, xstar_cls = decode_bits(hard)
            preds, probs = classify(scorer, imgs.reshape(len(imgs), -1), device)

            acc = float((preds == c).mean())
            p_target = probs[:, c]
            # clamped like the empirical logit: one misclassified sample at
            # p~0 would otherwise dominate the mean of the log.
            logp_target = float(np.log(np.clip(p_target, 1e-6, 1.0)).mean())
            onehot_ok = float((hard[:, CORE:].sum(1) == -(N_CLASSES - 2))
                              .float().mean())
            xstar_agree = float((xstar_cls == preds).mean())
            # Theory prediction, two references:
            #   *_prior  -- true MNIST class frequency: the END-TO-END claim,
            #               base measure and tilt together.
            #   *_base   -- this checkpoint's OWN measured tau=0 rate: isolates
            #               TILT FIDELITY from base-measure error (the pilot's
            #               tau=0 came in at 0.094 for class 0 against a true
            #               prior of 0.099, and that offset would otherwise be
            #               charged to the tilt).
            pred_prior = predicted_class_prob(float(priors[c]), tau)
            base_rate = base_acc.get(c)
            pred_base = (predicted_class_prob(base_rate, tau)
                         if base_rate not in (None, 0.0, 1.0) else float("nan"))
            row = {
                "ckpt": Path(a.ckpt).name, "tau": tau, "cls": c, "n": a.n,
                "p_pred_prior": pred_prior, "p_pred_base": pred_base,
                "gap_prior": acc - pred_prior, "gap_base": acc - pred_base,
                "cls_acc": acc, "xstar_agree": xstar_agree,
                "p_target_mean": float(p_target.mean()),
                "logp_target_mean": logp_target,
                "onehot_ok": onehot_ok,
                "mean_intensity": float(imgs.mean()),
                "K": a.K, "schedule": a.schedule, "T": a.T,
                "scorer_acc": scorer_acc, "seconds": time.time() - t0,
            }
            if tau == 0.0:
                base_logit[c] = _logit(acc, a.n)
                base_acc[c] = acc
                row["tau_eff"] = 0.0
            elif c in base_logit:
                row["tau_eff"] = (_logit(acc, a.n) - base_logit[c]) / 4.0
            else:
                row["tau_eff"] = float("nan")   # no tau=0 run in this ladder
            rows.append(row)
            print(f"  tau {tau:>4}  cls {c}  acc {acc:.3f}  "
                  f"pred {pred_prior:.3f}  gap {acc - pred_prior:+.3f}  "
                  f"logp {logp_target:>7.3f}  "
                  f"xstar_agree {xstar_agree:.3f}  onehot {onehot_ok:.3f}  "
                  f"tau_eff {row['tau_eff']:.3f}  ({row['seconds']:.0f}s)",
                  flush=True)
            if a.save_samples_dir:
                sd = Path(a.save_samples_dir)
                sd.mkdir(parents=True, exist_ok=True)
                np.save(sd / f"tau{tau}_c{c}.npy", imgs)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    piv = df.pivot_table(index="tau", values=["cls_acc", "p_pred_prior",
                                              "gap_prior", "gap_base",
                                              "logp_target_mean", "tau_eff",
                                              "xstar_agree", "onehot_ok"],
                         aggfunc="mean")
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\ncls_acc vs p_pred_prior IS THE THEORY TEST: the tilted target's "
          "exact class marginal, no free parameters. tau_eff should equal tau "
          "for a faithful sampler. Read gap_base to separate tilt error from "
          "base-measure error.")
    if float(df.loc[df.tau > 0, "cls_acc"].max() or 0) > 0.95 * scorer_acc:
        print(f"\nNOTE cls_acc is within 5% of the scorer's OWN test accuracy "
              f"({scorer_acc:.4f}) -- that column is saturated and cannot "
              f"separate this checkpoint from a perfect one. Compare "
              f"logp_target_mean instead, and put the tau ladder where "
              f"cls_acc lands in [0.2, 0.9].")


if __name__ == "__main__":
    main()
