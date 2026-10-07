"""Reward-tilted sudoku9 training.

The 9x9 counterpart of `run_train_sudoku6_reward.py`, same three featuriser
arms behind the same two boolean switches:

    --transformer   -> models_transformer.TransformerTilt   (9x36 + reward)
    --grouped       -> models.grouped_config                (9 groups of 36)
    (default)       -> models.MLPConfig                     (flat)
    --selfcond      -> tilt.train_selfcond.train_selfcond

Separate from the sudoku6 runner rather than a flag on it, because two defaults
here are not "the 6x6 value scaled" but load-bearing findings from those runs
applied to a different d, and burying them in a flag invites running the wrong
one:

LAMBDA_MAX. `lambda_matrix` admits lambda in [1, sqrt(d)), and the sudoku6
experiments established that the sampler queries lambda_eff = sqrt(d) EXACTLY at
t=0 -- so anything the training never draws is drawn with probability zero, not
merely under-weighted, and no reweighting scheme can repair it. At d=109 the
fix was 5.0 -> 10.0 (0.48 -> 0.96 of sqrt(d)), worth 0.605 -> 0.716 validity at
tau=8. Here sqrt(325) = 18.0278 and the default is 17.0, the same fraction.
This script REFUSES lambda_max >= sqrt(d) outright and warns below 0.8 sqrt(d).

NO IN-TRAINING EVAL. There is no exact operator to score against, so eval_fn is
None and the only number logged is the loss. On the 6x6 30k reference run the
loss moved -3.9% while the true operator error moved -50.8%. Treat the loss as
a liveness check, NOT as progress. Evaluate saved checkpoints with
`scripts/run_sudoku9_reward_tilt_check.py`, which scores against the exact
closed form `sudoku9.success_rate_exact`.

Usage:
    python scripts/run_train_sudoku9_reward.py --data data/sudoku9_v1 \
        --out ckpt/sudoku9_v1_l17_xf_sc_s8264_180k.pt \
        --transformer --selfcond --lambda-max 17.0 --steps 180000 \
        --d-model 384 --n-layers 6 --batch-size 8192 --wandb
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tilt.data import load_dataset
from tilt.models import MLPConfig, grouped_config
from tilt.models_transformer import TransformerTilt, transformer_config
from tilt.sudoku9 import LAMBDA_MAX_RECOMMENDED
from tilt.train import train
from tilt.train_selfcond import train_selfcond

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/sudoku9_v1")
p.add_argument("--out", default="ckpt/sudoku9_v1_reward.pt")

# --- featuriser -------------------------------------------------------------
p.add_argument("--selfcond", action="store_true")
p.add_argument("--self-cond-prob", type=float, default=0.5)
p.add_argument("--grouped", action="store_true",
               help="row-grouped (matrix) featuriser: 9 groups of 36 bits = "
                    "one 9x9 row of 9 cells x 4 bits, reward passes raw.")
p.add_argument("--n-groups", type=int, default=9)
p.add_argument("--group-bits", type=int, default=36)
p.add_argument("--h-group", type=int, default=32)
p.add_argument("--group-cond", default="film", choices=["none", "bias", "film"])
p.add_argument("--group-depth", type=int, default=1, choices=[1, 2])

p.add_argument("--transformer", action="store_true",
               help="TransformerTilt: --xf-seq tokens of --xf-bits bits plus "
                    "the reward slot. Default 9x36 = one token per grid row, "
                    "the arm that took 6x6 from 0.685 to 0.985 at tau=8. "
                    "81x4 (one token per CELL) also gives d=325 and is the "
                    "open tokenisation ablation.")
p.add_argument("--xf-seq", type=int, default=9)
p.add_argument("--xf-bits", type=int, default=36)
p.add_argument("--d-model", type=int, default=384)
p.add_argument("--n-layers", type=int, default=6)
p.add_argument("--n-heads", type=int, default=8)
p.add_argument("--d-ff", type=int, default=None)

# --- model / optimisation ---------------------------------------------------
p.add_argument("--width", type=int, default=1024)
p.add_argument("--blocks", type=int, default=8)
p.add_argument("--no-skip", action="store_true")
p.add_argument("--steps", type=int, default=180000)
p.add_argument("--batch-size", type=int, default=8192)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dt-weight", action="store_true")

# --- tilting / noising ------------------------------------------------------
p.add_argument("--lambda-max", type=float, default=LAMBDA_MAX_RECOMMENDED,
               help=f"reward-axis stretch; must be < sqrt(d) = 18.0278. "
                    f"Default {LAMBDA_MAX_RECOMMENDED} is the same fraction of "
                    f"sqrt(d) that 10.0 was at 6x6. 1.0 disables tilting.")
p.add_argument("--mode1-prob", type=float, default=0.25)
p.add_argument("--lambda1-prob", type=float, default=0.25)
p.add_argument("--beta-focus-lo", type=float, default=0.5)
p.add_argument("--beta-focus-hi", type=float, default=2.5)
p.add_argument("--beta-focus-prob", type=float, default=0.3)

p.add_argument("--ckpt-every", type=int, default=5000,
               help="write a resumable partial checkpoint every N steps. At "
                    "~0.23 s/step that is ~19 min, so a walltime kill costs at "
                    "most that much rather than the whole run. 0 disables.")
p.add_argument("--no-resume", action="store_true",
               help="ignore any existing <out>.partial.pt and start from step 1. "
                    "Default is to resume, which is a no-op on a first "
                    "submission, so the same command works either way.")

p.add_argument("--seed", type=int, default=8264)
p.add_argument("--device", default=None)
p.add_argument("--log-every", type=int, default=1000)

# --- wandb ------------------------------------------------------------------
p.add_argument("--wandb", action="store_true")
p.add_argument("--wandb-project", default="tilted-diffusions")
p.add_argument("--wandb-entity", default=None)
p.add_argument("--wandb-name", default=None)
p.add_argument("--wandb-tags", default=None)
p.add_argument("--wandb-notes", default=None)

a = p.parse_args()

_, _, _, meta = load_dataset(a.data)

if a.transformer and a.grouped:
    raise SystemExit("--transformer and --grouped are different featurisers; pick one")

# --- the lambda coverage guard (sudoku6 finding #3) --------------------------
sqrt_d = math.sqrt(meta["d"])
if a.lambda_max >= sqrt_d:
    raise SystemExit(
        f"--lambda-max={a.lambda_max} is not < sqrt(d)={sqrt_d:.4f}; "
        f"lambda_matrix is undefined there.")
if 1.0 < a.lambda_max < 0.8 * sqrt_d:
    print(f"WARNING: --lambda-max={a.lambda_max} is {a.lambda_max/sqrt_d:.0%} of "
          f"sqrt(d)={sqrt_d:.4f}. The sampler queries lambda_eff = sqrt(d) "
          f"exactly at t=0, so the top of that range is drawn with probability "
          f"ZERO during training. At 6x6 closing this gap (0.48 -> 0.96 of "
          f"sqrt(d)) moved validity 0.605 -> 0.716 at tau=8.",
          file=sys.stderr, flush=True)

net = None
if a.transformer:
    cfg = transformer_config(
        a.xf_seq, a.xf_bits, reward=True,
        d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads,
        d_ff=a.d_ff, skip=not a.no_skip, self_cond=bool(a.selfcond),
    )
    if cfg.d != meta["d"]:
        raise ValueError(
            f"transformer config d={cfg.d} (xf_seq={a.xf_seq} * xf_bits="
            f"{a.xf_bits} + 1 reward coord) does not match dataset d={meta['d']}")
    net = TransformerTilt(cfg)
elif a.grouped:
    cfg = grouped_config(
        a.n_groups, a.group_bits, reward=True,
        width=a.width, n_blocks=a.blocks, skip=not a.no_skip,
        h_group=a.h_group, group_cond=a.group_cond, group_depth=a.group_depth,
    )
    if cfg.d != meta["d"]:
        raise ValueError(
            f"grouped config d={cfg.d} does not match dataset d={meta['d']}")
else:
    cfg = MLPConfig(d=meta["d"], width=a.width, n_blocks=a.blocks,
                    skip=not a.no_skip)

success_rate = meta.get("success_rate")
if a.lambda_max > 1.0 and success_rate is not None and success_rate < 1e-6:
    print(f"WARNING: --lambda-max={a.lambda_max} but dataset {a.data} has "
          f"success_rate={success_rate:.4%} -- the reward coordinate is "
          f"constant, so there is nothing to tilt against.",
          file=sys.stderr, flush=True)

wandb_tags = [t.strip() for t in a.wandb_tags.split(",") if t.strip()] if a.wandb_tags else None
featuriser = "transformer" if a.transformer else ("matrix" if a.grouped else "flat")
wandb_config = {
    "board": "9x9",
    "selfcond": bool(a.selfcond),
    "featuriser": featuriser,
    "d_model": a.d_model if a.transformer else None,
    "n_layers": a.n_layers if a.transformer else None,
    "n_heads": a.n_heads if a.transformer else None,
    "xf_seq": a.xf_seq if a.transformer else None,
    "xf_bits": a.xf_bits if a.transformer else None,
    "self_cond_prob": a.self_cond_prob if a.selfcond else None,
    "n_groups": a.n_groups if a.grouped else None,
    "group_bits": a.group_bits if a.grouped else None,
    "lambda_max_frac_sqrt_d": a.lambda_max / sqrt_d,
    "success_rate": success_rate,
    "distinct_valid": meta.get("distinct_valid"),
    "seed_generator": meta.get("seed_generator"),
}

common = dict(
    steps=a.steps, batch_size=a.batch_size, lr=a.lr,
    dt_weight=a.dt_weight, lambda_max=a.lambda_max,
    mode1_prob=a.mode1_prob, lambda1_prob=a.lambda1_prob,
    beta_focus_range=(a.beta_focus_lo, a.beta_focus_hi),
    beta_focus_prob=a.beta_focus_prob,
    seed=a.seed, log_every=a.log_every,
    ckpt_every=(a.ckpt_every or None), resume=not a.no_resume,
    eval_fn=None,                      # see module docstring
    use_wandb=a.wandb, wandb_project=a.wandb_project,
    wandb_run_name=a.wandb_name, wandb_config=wandb_config,
    wandb_entity=a.wandb_entity, wandb_tags=wandb_tags,
    wandb_notes=a.wandb_notes, device=a.device,
)

if a.transformer:
    print(f"### featuriser : transformer  (d={cfg.d}, {cfg.seq_len}x{cfg.token_bits} "
          f"tokens, d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}, "
          f"{net.n_params():,} params)", flush=True)
else:
    print(f"### featuriser : {featuriser}  (d={cfg.d}, width={cfg.width}, "
          f"blocks={cfg.n_blocks})", flush=True)
print(f"### self-cond  : {a.selfcond}"
      + (f"  (p={a.self_cond_prob})" if a.selfcond else ""), flush=True)
print(f"### lambda_max : {a.lambda_max}  ({a.lambda_max/sqrt_d:.0%} of sqrt(d)="
      f"{sqrt_d:.4f})   mode1_prob: {a.mode1_prob}", flush=True)
print(f"### ckpt       : every {a.ckpt_every or 0} steps, "
      f"resume={not a.no_resume}", flush=True)
print(f"### dataset    : {a.data}  success_rate={success_rate}  "
      f"distinct_valid={meta.get('distinct_valid')}  "
      f"gen={meta.get('seed_generator')}", flush=True)

if a.selfcond:
    train_selfcond(a.data, a.out, cfg, net=net,
                   self_cond_prob=a.self_cond_prob, **common)
else:
    train(a.data, a.out, cfg, net=net, **common)
