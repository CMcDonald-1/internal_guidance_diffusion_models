"""Class-conditional MNIST training: block reward tilt on a 10-bit one-hot.

The mnist_v1 counterpart of run_train_sudoku9_reward.py, with three defaults
that are NOT that script's values scaled -- each is a load-bearing decision
from earlier experiments, so they live here rather than in flags you have to
remember:

LAMBDA_MAX CEILING IS sqrt(d / n_extra), NOT sqrt(d). With a 10-wide reward
block the budget identity ||Lambda x||^2 = d costs 10 * lambda^2, so the core
scale hits zero at lambda = sqrt(6282/10) = 25.064 -- and that endpoint is
exactly the t=0 pattern-tilt query (all norm on the reward block). Default
24.0 is the same close-beneath-the-endpoint fraction the 6x6 -> 9x9 transfer
used. This script REFUSES lambda_max >= the ceiling and warns below 0.8 of it.

NO INFILL MODE (mode1_prob=0). This experiment never issues an infill query:
class-conditional sampling tilts the reward block only. Training capability
that is never queried is budget leak (cf. the sudoku6 rn0034 finding), and
with the tilted-draw budget already spread over 10 class patterns there is
less to spare. Class-conditional INPAINTING is the v2 run that turns it on.

SAMPLE-TIME TILT IS THE PATTERN, h = tau * r_c, NOT tau * e_c. Block-Lambda
training never manufactures asymmetric reward-block geometry, so a single-axis
tilt starts the trajectory outside the training measure. tilt.mnist.reward_tilt
builds the right one. (Dial conversion: pattern separation is e^{4 tau}.)

HOLDOUT IS ON BY DEFAULT and is the memorisation watch: a frozen batch from
the official 10k test split -- a source BatchSampler never draws from --
scored with the run's own augmentation recipe. 60k images resampled for
steps*batch draws is ~1k+ epochs; watch train-vs-holdout diverge, keep
checkpoints, retreat to the pre-divergence one.

Pilot (256x4, ~3.7M params, ~1k epochs):
    python scripts/run_train_mnist_reward.py --data data/mnist_v1 \
        --out ckpt/mnist_v1_pilot_l24_xf49_d256x4_s8264_30k.pt \
        --steps 30000 --batch-size 2048

Larger variant (512x8, ~27M params; not used in the paper):
    python scripts/run_train_mnist_reward.py --data data/mnist_v1 \
        --out ckpt/mnist_v1_l24_xf49_d512x8_sc_s8264_100k.pt \
        --selfcond --d-model 512 --n-layers 8 --steps 100000 --batch-size 4096
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tilt.amp import maybe_autocast
from tilt.data import build_frozen_holdout, load_dataset
from tilt.mnist import (D, LAMBDA_MAX_CEILING, LAMBDA_MAX_RECOMMENDED,
                        N_CLASSES, ImageBackedX)
from tilt.models_transformer import TransformerConfig, TransformerTilt
from tilt.train import train
from tilt.train_selfcond import train_selfcond

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/mnist_v1")
p.add_argument("--out", default="ckpt/mnist_v1_pilot.pt")

# --- featuriser: transformer only. A flat MLP posted 0.0000 at d=325; at
# --- d=6282 it is not an ablation worth a flag. ------------------------------
p.add_argument("--xf-seq", type=int, default=49,
               help="tokens; 49 = 7x7 grid of 4x4-pixel patches. 196 (2x2 "
                    "patches x 32 bits) is the open tokenisation ablation.")
p.add_argument("--xf-bits", type=int, default=128)
p.add_argument("--d-model", type=int, default=256)
p.add_argument("--n-layers", type=int, default=4)
p.add_argument("--n-heads", type=int, default=8)
p.add_argument("--d-ff", type=int, default=None)
p.add_argument("--selfcond", action="store_true")
p.add_argument("--self-cond-prob", type=float, default=0.5)

# --- optimisation ------------------------------------------------------------
p.add_argument("--steps", type=int, default=30000)
p.add_argument("--batch-size", type=int, default=2048)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dt-weight", action="store_true")
p.add_argument("--amp", default="bf16", choices=["off", "bf16", "fp16"],
               help="bf16 autocast via tilt.amp.AutocastNet. DEFAULTS ON here, "
                    "unlike run_train_transformer.py: mnist_v1 is a new "
                    "experiment with no fp32 baselines to protect, and the "
                    "first fp32 run measured ~31 TFLOPS effective on an H100 "
                    "(the card does ~990 at bf16). Checkpoints are ordinary "
                    "TransformerTilt files -- AMP leaves no trace.")
p.add_argument("--tf32", default="on", choices=["on", "off"],
               help="TF32 matmuls (torch.set_float32_matmul_precision). Free "
                    "on H100/A100 and orthogonal to --amp; covers whatever "
                    "stays in fp32. Same new-experiment reasoning as --amp.")

# --- tilting / noising -------------------------------------------------------
p.add_argument("--lambda-max", type=float, default=LAMBDA_MAX_RECOMMENDED,
               help=f"reward-BLOCK stretch; must be < sqrt(d/{N_CLASSES}) = "
                    f"{LAMBDA_MAX_CEILING:.4f} (NOT sqrt(d) -- 10-wide block). "
                    f"Default {LAMBDA_MAX_RECOMMENDED}. 1.0 disables tilting.")
p.add_argument("--mode1-prob", type=float, default=0.0,
               help="infill focusing. 0 here BY DESIGN (see module docstring); "
                    "the inpainting run is the place to raise it.")
p.add_argument("--lambda1-prob", type=float, default=0.25)
p.add_argument("--beta-focus-lo", type=float, default=0.5)
p.add_argument("--beta-focus-hi", type=float, default=2.5)
p.add_argument("--beta-focus-prob", type=float, default=0.0)

# --- holdout (memorisation watch) --------------------------------------------
p.add_argument("--holdout-n", type=int, default=4096)
p.add_argument("--holdout-every", type=int, default=1000)
p.add_argument("--holdout-seed", type=int, default=1234,
               help="fixed ACROSS runs so holdout curves are comparable "
                    "between checkpoints trained with the same augmentation.")
p.add_argument("--no-holdout", action="store_true")

p.add_argument("--ckpt-every", type=int, default=5000)
p.add_argument("--no-resume", action="store_true")
p.add_argument("--seed", type=int, default=8264)
p.add_argument("--device", default=None)
p.add_argument("--log-every", type=int, default=500)

# --- wandb -------------------------------------------------------------------
p.add_argument("--wandb", action="store_true")
p.add_argument("--wandb-project", default="tilted-diffusions")
p.add_argument("--wandb-entity", default=None)
p.add_argument("--wandb-name", default=None)
p.add_argument("--wandb-tags", default=None)
p.add_argument("--wandb-notes", default=None)

a = p.parse_args()

_, _, _, meta = load_dataset(a.data)
if meta["d"] != D:
    raise SystemExit(f"dataset d={meta['d']} but tilt.mnist says d={D}")
n_extra = int(meta.get("n_extra", N_CLASSES))

# --- the lambda coverage guard, block edition --------------------------------
ceiling = math.sqrt(meta["d"] / n_extra)
if a.lambda_max >= ceiling:
    raise SystemExit(
        f"--lambda-max={a.lambda_max} is not < sqrt(d/n_extra)="
        f"{ceiling:.4f}. With a {n_extra}-wide reward block the budget "
        f"identity is core*scale + {n_extra}*lambda^2 = d; sqrt(d)="
        f"{math.sqrt(meta['d']):.2f} is NOT the ceiling here.")
if 1.0 < a.lambda_max < 0.8 * ceiling:
    print(f"WARNING: --lambda-max={a.lambda_max} is {a.lambda_max/ceiling:.0%} "
          f"of the block ceiling {ceiling:.4f}. The t=0 pattern-tilt query "
          f"sits AT the ceiling, so the top of the range is drawn with "
          f"probability ZERO during training (sudoku6 finding #3).",
          file=sys.stderr, flush=True)

cfg = TransformerConfig(
    d=meta["d"], seq_len=a.xf_seq, token_bits=a.xf_bits,
    d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads,
    d_ff=a.d_ff, self_cond=bool(a.selfcond),
).validate()
if cfg.n_extra != n_extra:
    raise SystemExit(
        f"xf_seq*xf_bits = {a.xf_seq * a.xf_bits} leaves n_extra={cfg.n_extra}, "
        f"dataset wants {n_extra}. 49x128 + 10 = 6282 is the intended layout.")
if a.tf32 == "on":
    # Affects fp32 matmuls only; a no-op on CPU and on pre-Ampere cards.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

net = TransformerTilt(cfg)
# maybe_autocast returns `net` itself when amp is off -- no wrapper in that path.
net = maybe_autocast(net, a.amp)

# --- frozen holdout from the official test split -----------------------------
holdout = None
if not a.no_holdout:
    datadir = Path(a.data)
    hx = ImageBackedX(np.load(datadir / "holdout_images.npy"),
                      np.load(datadir / "holdout_labels.npy"))

    def _sampler(n: int, g: torch.Generator) -> torch.Tensor:
        if n > len(hx):
            raise ValueError(f"holdout_n={n} > {len(hx)} test rows")
        return hx[torch.randperm(len(hx), generator=g)[:n]]

    holdout = build_frozen_holdout(
        _sampler, a.holdout_n, beta_lo=meta["beta_lo"], beta_hi=meta["beta_hi"],
        lambda_max=a.lambda_max, mode1_prob=a.mode1_prob,
        lambda1_prob=a.lambda1_prob,
        beta_focus_range=(a.beta_focus_lo, a.beta_focus_hi),
        beta_focus_prob=a.beta_focus_prob, seed=a.holdout_seed,
        n_extra=n_extra)

epochs = a.steps * a.batch_size / meta["n"]
wandb_tags = [t.strip() for t in a.wandb_tags.split(",") if t.strip()] if a.wandb_tags else None
wandb_config = {
    "board": "mnist", "selfcond": bool(a.selfcond), "featuriser": "transformer",
    "d_model": a.d_model, "n_layers": a.n_layers, "n_heads": a.n_heads,
    "xf_seq": a.xf_seq, "xf_bits": a.xf_bits, "n_extra": n_extra,
    "self_cond_prob": a.self_cond_prob if a.selfcond else None,
    "lambda_max_frac_ceiling": a.lambda_max / ceiling,
    "lambda_ceiling": ceiling, "epochs_equivalent": epochs,
    "holdout_n": None if a.no_holdout else a.holdout_n,
    "encoding": meta.get("encoding"),
    "amp": a.amp, "tf32": a.tf32,
}

common = dict(
    steps=a.steps, batch_size=a.batch_size, lr=a.lr,
    dt_weight=a.dt_weight, lambda_max=a.lambda_max,
    mode1_prob=a.mode1_prob, lambda1_prob=a.lambda1_prob,
    beta_focus_range=(a.beta_focus_lo, a.beta_focus_hi),
    beta_focus_prob=a.beta_focus_prob,
    seed=a.seed, log_every=a.log_every,
    ckpt_every=(a.ckpt_every or None), resume=not a.no_resume,
    eval_fn=None,                    # no exact operator; see holdout + tilt check
    holdout=holdout, holdout_every=a.holdout_every,
    use_wandb=a.wandb, wandb_project=a.wandb_project,
    wandb_run_name=a.wandb_name, wandb_config=wandb_config,
    wandb_entity=a.wandb_entity, wandb_tags=wandb_tags,
    wandb_notes=a.wandb_notes, device=a.device,
)

print(f"### featuriser : transformer  (d={cfg.d}, {cfg.seq_len}x{cfg.token_bits} "
      f"tokens + {cfg.n_extra} class bits, d_model={cfg.d_model}, "
      f"layers={cfg.n_layers}, heads={cfg.n_heads}, {net.n_params():,} params)",
      flush=True)
print(f"### precision  : amp={a.amp}  tf32={a.tf32}   (fp32 baseline ran at "
      f"~1.05 s/step for 512x8 @ batch 4096)", flush=True)
print(f"### self-cond  : {a.selfcond}"
      + (f"  (p={a.self_cond_prob})" if a.selfcond else ""), flush=True)
print(f"### lambda_max : {a.lambda_max}  ({a.lambda_max/ceiling:.0%} of block "
      f"ceiling sqrt(d/{n_extra})={ceiling:.4f})   mode1_prob: {a.mode1_prob}",
      flush=True)
print(f"### data       : {a.data}  n={meta['n']:,}  ->  {epochs:,.0f} epochs "
      f"equivalent over {a.steps:,} steps x {a.batch_size}", flush=True)
print(f"### holdout    : "
      + ("OFF" if a.no_holdout else
         f"{a.holdout_n} rows from the 10k test split, every "
         f"{a.holdout_every} steps (the memorisation watch)"), flush=True)
print(f"### ckpt       : every {a.ckpt_every or 0} steps, "
      f"resume={not a.no_resume}", flush=True)

if a.selfcond:
    train_selfcond(a.data, a.out, cfg, net=net,
                   self_cond_prob=a.self_cond_prob, **common)
else:
    train(a.data, a.out, cfg, net=net, **common)
