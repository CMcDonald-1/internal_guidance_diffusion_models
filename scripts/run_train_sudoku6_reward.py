"""Reward-tilted sudoku6 training, covering the 2x2 of
{self-conditioned, not} x {flat featuriser, matrix/grouped featuriser}.

WHY THIS FILE EXISTS
--------------------
No existing runner can do both reward tilting and the matrix featuriser:

    run_train.py                          --lambda-max  but no --grouped
    run_train_selfcond.py                 --lambda-max  but no --grouped
    run_train_sudoku6_rows.py             --grouped     but lambda_max hardcoded 1.0
    run_train_sudoku6_rows_selfcond.py    neither

Rather than patch two of them (and risk the flat and matrix arms drifting
apart), this puts all four cells behind two boolean switches on ONE code path,
so the only differences between arms are the factors under study.

    --selfcond   -> tilt.train_selfcond.train_selfcond + SelfCondTiltMLP
    (default)    -> tilt.train.train                   + TiltMLP
    --grouped    -> models.grouped_config (K groups of M bits + reward coord)
    (default)    -> models.MLPConfig      (n_groups=None reproduces flat exactly)

NO IN-TRAINING EVAL
-------------------
sudoku6_v1 has no target.json -- 28.2M valid 6x6 solutions is not enumerable,
so there is no exact operator to score against and eval_fn is None. The only
number logged during training is the loss, and on the 30k reference run the
loss moved -3.9% while the true operator error moved -50.8%. Treat the loss
as a liveness check, NOT as progress. Evaluate the saved checkpoints locally.

(If you later want an in-training signal here, tilt/data.py's
build_frozen_holdout + train()'s holdout= argument is the ground-truth-free
route -- it needs an x_sampler drawn from a source DISJOINT from train.csv.)

Usage:
    python scripts/run_train_sudoku6_reward.py --data data/sudoku6_v1 \
        --out ckpt/sudoku6_v1_lambda5_flat_nosc_s0_180k.pt \
        --lambda-max 5.0 --steps 180000 --width 1024 --blocks 8 \
        --batch-size 8192 --wandb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tilt.data import load_dataset
from tilt.models import MLPConfig, grouped_config
from tilt.models_transformer import TransformerTilt, transformer_config
from tilt.train import train
from tilt.train_selfcond import train_selfcond

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/sudoku6_v1")
p.add_argument("--out", default="ckpt/sudoku6_v1_reward.pt")

# --- the two factors under study -------------------------------------------
p.add_argument("--selfcond", action="store_true",
               help="train a SelfCondTiltMLP via tilt.train_selfcond instead of "
                    "a plain TiltMLP via tilt.train.")
p.add_argument("--self-cond-prob", type=float, default=0.5,
               help="per-step probability of the two-pass self-conditioning "
                    "update (Bit Diffusion Algorithm 1); the rest of the time "
                    "trains the x_pred=0 fallback needed at the first sampling "
                    "step. Ignored unless --selfcond.")
p.add_argument("--grouped", action="store_true",
               help="use the row-grouped (matrix) featuriser: one shared encoder "
                    "applied to each of --n-groups contiguous --group-bits blocks "
                    "(default 6 groups of 18 bits = one 6x6 row of 6 cells x 3 "
                    "bits). The reward coordinate always passes through raw.")
p.add_argument("--n-groups", type=int, default=6)
p.add_argument("--group-bits", type=int, default=18)
p.add_argument("--h-group", type=int, default=32)
p.add_argument("--group-cond", default="film", choices=["none", "bias", "film"])
p.add_argument("--group-depth", type=int, default=1, choices=[1, 2])

# --- transformer featuriser (third arm, mutually exclusive with --grouped) --
p.add_argument("--transformer", action="store_true",
               help="use TransformerTilt instead of the MLP: --xf-seq tokens of "
                    "--xf-bits bits plus the reward slot, attention across "
                    "tokens. Default 6x18 = one token per grid row, which is "
                    "the sudoku6 case in models_transformer's own self-test. "
                    "Composes with --selfcond (sets TransformerConfig.self_cond).")
p.add_argument("--xf-seq", type=int, default=6)
p.add_argument("--xf-bits", type=int, default=18)
p.add_argument("--d-model", type=int, default=256)
p.add_argument("--n-layers", type=int, default=4)
p.add_argument("--n-heads", type=int, default=8)
p.add_argument("--d-ff", type=int, default=None)

# --- model / optimisation --------------------------------------------------
p.add_argument("--width", type=int, default=1024)
p.add_argument("--blocks", type=int, default=8)
p.add_argument("--no-skip", action="store_true")
p.add_argument("--steps", type=int, default=180000)
p.add_argument("--batch-size", type=int, default=8192)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--dt-weight", action="store_true")

# --- tilting / noising -----------------------------------------------------
p.add_argument("--lambda-max", type=float, default=5.0,
               help="reward-axis stretch. 1.0 disables tilting entirely.")
p.add_argument("--mode1-prob", type=float, default=0.25,
               help="probability per row of the infill-focusing profile "
                    "(sparse random-k-subset q) instead of the reward-focusing "
                    "profile (q=ones).")
p.add_argument("--lambda1-prob", type=float, default=0.25,
               help="probability per row of forcing lambda=1 (no stretch) "
                    "instead of drawing log-uniform on [1, lambda-max].")
p.add_argument("--beta-focus-lo", type=float, default=0.5)
p.add_argument("--beta-focus-hi", type=float, default=2.5)
p.add_argument("--beta-focus-prob", type=float, default=0.3)

p.add_argument("--seed", type=int, default=8264)
p.add_argument("--device", default=None)
p.add_argument("--log-every", type=int, default=1000)

# --- wandb -----------------------------------------------------------------
p.add_argument("--wandb", action="store_true")
p.add_argument("--wandb-project", default="tilted-diffusions")
p.add_argument("--wandb-entity", default=None)
p.add_argument("--wandb-name", default=None)
p.add_argument("--wandb-tags", default=None)
p.add_argument("--wandb-notes", default=None)

a = p.parse_args()

_, _, _, meta = load_dataset(a.data)

# ---------------------------------------------------------------------------
# Model config. MLPConfig(n_groups=None) reproduces the flat model bit for bit,
# so the two arms differ only in the featuriser.
# ---------------------------------------------------------------------------
if a.transformer and a.grouped:
    raise SystemExit("--transformer and --grouped are different featurisers; pick one")

net = None                      # train()/train_selfcond() build the MLP if this stays None
if a.transformer:
    cfg = transformer_config(
        a.xf_seq, a.xf_bits, reward=True,
        d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads,
        d_ff=a.d_ff, skip=not a.no_skip, self_cond=bool(a.selfcond),
    )
    if cfg.d != meta["d"]:
        raise ValueError(
            f"transformer config d={cfg.d} (xf_seq={a.xf_seq} * xf_bits="
            f"{a.xf_bits} + 1 reward coord) does not match dataset d={meta['d']}"
        )
    net = TransformerTilt(cfg)
elif a.grouped:
    cfg = grouped_config(
        a.n_groups, a.group_bits, reward=True,
        width=a.width, n_blocks=a.blocks, skip=not a.no_skip,
        h_group=a.h_group, group_cond=a.group_cond, group_depth=a.group_depth,
    )
    if cfg.d != meta["d"]:
        raise ValueError(
            f"grouped config d={cfg.d} (n_groups={a.n_groups} * group_bits="
            f"{a.group_bits} + 1 reward coord) does not match dataset d={meta['d']}"
        )
else:
    cfg = MLPConfig(d=meta["d"], width=a.width, n_blocks=a.blocks,
                    skip=not a.no_skip)

# Sanity: tilting against a constant reward coordinate is a no-op. sudoku6_v1
# should report ~5.01%; sudoku6_rn0034_v1 reports 0.000% and is the WRONG
# dataset for a tilted run. Warn rather than fail -- lambda_max=1.0 on an
# untilted set is a legitimate control arm.
success_rate = meta.get("success_rate")
if a.lambda_max > 1.0 and success_rate is not None and success_rate < 1e-6:
    print(f"WARNING: --lambda-max={a.lambda_max} but dataset {a.data} has "
          f"success_rate={success_rate:.4%} -- the reward coordinate is "
          f"constant, so there is nothing to tilt against. Did you mean "
          f"data/sudoku6_v1?", file=sys.stderr, flush=True)

wandb_tags = [t.strip() for t in a.wandb_tags.split(",") if t.strip()] if a.wandb_tags else None

# Log the 2x2 factors as first-class config fields so runs can be grouped and
# filtered in the wandb UI by factor, not by parsing run names.
featuriser = "transformer" if a.transformer else ("matrix" if a.grouped else "flat")
wandb_config = {
    "selfcond": bool(a.selfcond),
    "featuriser": featuriser,
    "d_model": a.d_model if a.transformer else None,
    "n_layers": a.n_layers if a.transformer else None,
    "n_heads": a.n_heads if a.transformer else None,
    "self_cond_prob": a.self_cond_prob if a.selfcond else None,
    "n_groups": a.n_groups if a.grouped else None,
    "group_bits": a.group_bits if a.grouped else None,
    "group_cond": a.group_cond if a.grouped else None,
    "success_rate": success_rate,
}

common = dict(
    steps=a.steps, batch_size=a.batch_size, lr=a.lr,
    dt_weight=a.dt_weight, lambda_max=a.lambda_max,
    mode1_prob=a.mode1_prob, lambda1_prob=a.lambda1_prob,
    beta_focus_range=(a.beta_focus_lo, a.beta_focus_hi),
    beta_focus_prob=a.beta_focus_prob,
    seed=a.seed, log_every=a.log_every,
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
    print(f"### featuriser : {'matrix' if a.grouped else 'flat'}  (d={cfg.d}, "
          f"width={cfg.width}, blocks={cfg.n_blocks})", flush=True)
print(f"### self-cond  : {a.selfcond}"
      + (f"  (p={a.self_cond_prob})" if a.selfcond else ""), flush=True)
print(f"### lambda_max : {a.lambda_max}   mode1_prob: {a.mode1_prob}", flush=True)
print(f"### dataset    : {a.data}  success_rate={success_rate}", flush=True)

if a.selfcond:
    train_selfcond(a.data, a.out, cfg, net=net,
                   self_cond_prob=a.self_cond_prob, **common)
else:
    train(a.data, a.out, cfg, net=net, **common)
