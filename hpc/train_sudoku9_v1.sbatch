#!/bin/bash
# ============================================================================
# Reward-tilted sudoku9 training.
#
#   sbatch --array=0-0 -t 20:00:00 hpc/train_sudoku9_v1.sbatch xf_sc
#
# 20 hours, not 9. See WALLTIME below. No smoke needed any more -- the pipeline
# is validated and a partial checkpoint now survives a kill.
#
# EVERYTHING inline-env must go through --export, e.g.
#   sbatch --export=ALL,LAMBDA_MAX=17.0,STEPS=400 --array=0-0 ...
# The bare `VAR=val sbatch ...` prefix form does NOT survive submission on some
# Slurm clusters -- it silently runs the defaults instead. (Same trap as the sudoku6
# scripts; it has cost a run before.)
#
# Cells:
#   flat_nosc / flat_sc        MLP, flat featuriser
#   matrix_nosc / matrix_sc    MLP, row-grouped featuriser (9 groups x 36 bits)
#   xf_nosc / xf_sc            TransformerTilt, 9 tokens x 36 bits + reward
#
# WHAT CARRIES OVER FROM THE 6x6 RUNS
# -----------------------------------
# LAMBDA_MAX=17.0, not 10.0. lambda_matrix admits lambda in [1, sqrt(d)) and
# the sampler queries lambda_eff = sqrt(d) EXACTLY at t=0. Here d=325 so
# sqrt(d)=18.0278 and 17.0 is 94% of it -- the same fraction 10.0 was of
# sqrt(109)=10.4403. Training below that draws the hard band with probability
# exactly ZERO, which no reweighting scheme repairs; closing this gap at 6x6
# was worth 0.605 -> 0.716 validity at tau=8. run_train_sudoku9_reward.py
# refuses lambda_max >= sqrt(d) and warns below 0.8*sqrt(d).
#
# The transformer arm is the default cell for a reason: on identical data,
# lambda, sampler and seed at 6x6 it beat the flat MLP 0.985 vs 0.685 at tau=8,
# and it was the only arm whose tau_eff still rose with tau. d_model=384 /
# 6 layers here (10.8M params) rather than 6x6's 256/4 (3.25M), for a state
# space 3x larger.
#
# DO NOT TRUST THE LOSS. There is no exact operator to score against, so
# eval_fn is None. On the 6x6 30k reference run the loss moved -3.9% while the
# true operator error moved -50.8%. The loss is a liveness check. Score the
# checkpoint with scripts/run_sudoku9_reward_tilt_check.py, at K=1000 and
# schedule=sstar -- K=200 was the DOMINANT error term at 6x6 and every
# conclusion drawn there at K=200 was later shown unsafe.
#
# WALLTIME. MEASURED, not estimated: 0.231 s/step for xf_sc at d_model=384 /
# 6 layers, batch 8192. 180,000 steps is therefore 11.6 HOURS. The first
# attempt was submitted at -t 09:00:00, died at 140k steps, and -- because
# net.save() only ran after the loop -- produced no checkpoint at all. 18 GPU
# hours and a day, gone. Use -t 20:00:00. The cap is 48h; there is no prize
# for cutting it fine.
#
# CHECKPOINTING now makes that unrecoverable case recoverable. A resumable
# partial is written every CKPT_EVERY steps (default 5000, ~19 min) and the
# resume restores weights + Adam moments + the cosine LR schedule + the
# BatchSampler RNG, which is bitwise identical to an uninterrupted run
# (asserted in tilt/checkpointing.py::_selftest). A walltime kill now costs
# <=19 minutes, not the run.
#
# THE SELF-HEALING PAIR. Submit the run AND a follow-up gated on afterany.
# If the first job finishes, the second sees the final checkpoint and exits 0
# immediately. If the first hits walltime, the second resumes from the partial:
#
#   J1=$(sbatch --parsable --array=0-0 -t 20:00:00 \
#            hpc/train_sudoku9_v1.sbatch xf_sc)
#   sbatch --dependency=afterany:$J1 --array=0-0 -t 20:00:00 \
#            hpc/train_sudoku9_v1.sbatch xf_sc
#
# afterany, not afterok -- the whole point is to run after a FAILED/TIMEOUT
# predecessor. No trap, no self-resubmission, no runaway risk.
# ============================================================================

#SBATCH --job-name=s9v1
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gpus=1                      # paper runs: one 80GB H100
#SBATCH --ntasks-per-node=12
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

CELL="${1:?usage: sbatch --array=0-0 train_sudoku9_v1.sbatch <flat_nosc|flat_sc|matrix_nosc|matrix_sc|xf_nosc|xf_sc>}"

AI_MODULE="${AI_MODULE:-}"          # optional: cluster PyTorch module to load
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repo root
VENV_PATH="${VENV_PATH:-$CODE_DIR/.venv}"
DATA_ROOT="${DATA_ROOT:-$CODE_DIR/data}"
CKPT_ROOT="${CKPT_ROOT:-$CODE_DIR/ckpt}"

DATASET_NAME="${DATASET_NAME:-sudoku9_v1}"
DATA_DIR="$DATA_ROOT/$DATASET_NAME"

STEPS="${STEPS:-180000}"
WIDTH=1024
BLOCKS=8
BATCH_SIZE="${BATCH_SIZE:-8192}"
LR=1e-3
LAMBDA_MAX="${LAMBDA_MAX:-17.0}"     # must stay < sqrt(325) = 18.0278
MODE1_PROB=0.25
LAMBDA1_PROB=0.25
BETA_FOCUS_LO=0.5
BETA_FOCUS_HI=2.5
BETA_FOCUS_PROB=0.3
SELF_COND_PROB=0.5
LOG_EVERY="${LOG_EVERY:-1000}"
CKPT_EVERY="${CKPT_EVERY:-5000}"     # ~19 min at 0.23 s/step
RESUME="${RESUME:-1}"

# transformer geometry (ignored by the MLP cells)
XF_SEQ="${XF_SEQ:-9}"                # 9 tokens of 36 bits = one per grid row
XF_BITS="${XF_BITS:-36}"             # 81 x 4 (one per cell) also gives d=325
D_MODEL="${D_MODEL:-384}"
N_LAYERS="${N_LAYERS:-6}"
N_HEADS="${N_HEADS:-8}"

SEEDS=(8264 1337 20260814)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#SEEDS[@]}" ]; then
    echo "ERROR: array index $IDX exceeds the ${#SEEDS[@]} seeds defined." >&2
    exit 1
fi
SEED="${SEEDS[$IDX]}"

XF_ARGS="--transformer --xf-seq $XF_SEQ --xf-bits $XF_BITS --d-model $D_MODEL --n-layers $N_LAYERS --n-heads $N_HEADS"
case "$CELL" in
    flat_nosc)   CELL_ARGS="" ;;
    flat_sc)     CELL_ARGS="--selfcond --self-cond-prob $SELF_COND_PROB" ;;
    matrix_nosc) CELL_ARGS="--grouped" ;;
    matrix_sc)   CELL_ARGS="--grouped --selfcond --self-cond-prob $SELF_COND_PROB" ;;
    xf_nosc)     CELL_ARGS="$XF_ARGS" ;;
    xf_sc)       CELL_ARGS="$XF_ARGS --selfcond --self-cond-prob $SELF_COND_PROB" ;;
    *) echo "ERROR: unknown cell '$CELL'" >&2; exit 1 ;;
esac

LTAG="l$(printf '%g' "$LAMBDA_MAX" | tr -d '.')"

# TAG distinguishes runs that differ in something the name does not otherwise
# encode -- d_model/n_layers, tokenisation, batch size. Without it two xf_sc
# arms produce the same RUN_NAME and the second dies on the overwrite guard.
#
# It goes BEFORE the cell, not at the end. tilt/load_ckpt.py matches
#     _(flat|matrix|xf)_(sc|nosc)_s(\w+?)_(\d+k)\.pt$
# so anything appended after the steps tag stops that regex matching, and an
# MLP arm would then silently load as non-self-conditioned -- the exact bug
# that module was written to kill. Placed here the regex still matches:
#     sudoku9_v1_l17_d256x4_xf_sc_s8264_180k.pt
TAG="${TAG:-}"
SUFFIX="${TAG:+_$TAG}"

if [ "$STEPS" -lt 180000 ]; then
    RUN_NAME="SMOKE_${DATASET_NAME}_${LTAG}${SUFFIX}_${CELL}_s${SEED}_${STEPS}"
else
    RUN_NAME="${DATASET_NAME}_${LTAG}${SUFFIX}_${CELL}_s${SEED}_180k"
fi

# With RESUME=1 an already-finished run is not an error: it is the expected
# outcome of the paired follow-up job described at the bottom of this file.
# Exit 0 so the chain reports CANCELLED/COMPLETED rather than a spurious FAILED.
if [ -f "$CKPT_ROOT/${RUN_NAME}.pt" ]; then
    if [ "$RESUME" = "1" ]; then
        echo "$CKPT_ROOT/${RUN_NAME}.pt already exists -- run already finished."
        echo "Nothing to do. (This is the normal exit for a follow-up job.)"
        exit 0
    fi
    echo "ERROR: $CKPT_ROOT/${RUN_NAME}.pt already exists -- refusing to overwrite." >&2
    exit 1
fi

echo "=========================================================="
echo "array job    : ${SLURM_ARRAY_JOB_ID:-n/a} task ${IDX}"
echo "job id       : $SLURM_JOB_ID"
echo "node         : $(hostname)"
echo "started      : $(date)"
echo "cell         : $CELL     (args: ${CELL_ARGS:-<none>})"
echo "dataset      : $DATASET_NAME"
echo "lambda_max   : $LAMBDA_MAX   (sqrt(d) = 18.0278)"
echo "batch size   : $BATCH_SIZE"
echo "seed         : $SEED"
echo "run name     : $RUN_NAME"
echo "ckpt every   : $CKPT_EVERY steps   resume: $RESUME"
if [ -f "$CKPT_ROOT/${RUN_NAME}.partial.pt" ]; then
    echo "partial      : FOUND -- this job will resume, not restart"
fi
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

if [ -n "$AI_MODULE" ]; then module load "$AI_MODULE"; fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

python -c "import torch; print('torch', torch.__version__, '| device', torch.cuda.get_device_name(0))"
echo

cd "$CODE_DIR"
mkdir -p "$CKPT_ROOT" logs

# --- PREREQUISITE: build the dataset first, once ----------------------------
#   sbatch hpc/make_data_sudoku9_v1.sbatch
# Never generated here: concurrent array tasks would race to write it.
if [ ! -f "$DATA_DIR/grids.npy" ] && [ ! -f "$DATA_DIR/train.csv" ]; then
    echo "ERROR: dataset not found at $DATA_DIR" >&2
    echo "       Run hpc/make_data_sudoku9_v1.sbatch first." >&2
    exit 1
fi

python - "$DATA_DIR" <<'PY'
import json, sys
meta = json.load(open(f"{sys.argv[1]}/meta.json"))
sr = meta.get("success_rate", 0.0)
dv = meta.get("distinct_valid")
print(f"### dataset d={meta['d']} n={meta['n']:,} seed={meta['seed']} "
      f"success_rate={sr:.4%} format={meta.get('format','csv')} "
      f"gen={meta.get('seed_generator','?')}")
if meta.get("d") != 325:
    sys.exit(f"ERROR: d={meta['d']}, expected 325. Is this a sudoku6 dataset?")
if dv:
    exp = 180_000 * 8192 * meta["p_inject"] / dv
    print(f"### distinct valid grids: {dv:,}  -> {exp:,.0f} exposures each "
          f"(sudoku6_v1 was 7,511 -> 9,816, and memorised 33.6%)")
if sr < 1e-6:
    sys.exit("ERROR: success_rate is 0 -- nothing to tilt against.")
PY
echo

SECONDS=0

# shellcheck disable=SC2086
python scripts/run_train_sudoku9_reward.py $CELL_ARGS \
    --data "$DATA_DIR" \
    --out "$CKPT_ROOT/${RUN_NAME}.pt" \
    --steps "$STEPS" \
    --width "$WIDTH" \
    --blocks "$BLOCKS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --lambda-max "$LAMBDA_MAX" \
    --mode1-prob "$MODE1_PROB" \
    --lambda1-prob "$LAMBDA1_PROB" \
    --beta-focus-lo "$BETA_FOCUS_LO" \
    --beta-focus-hi "$BETA_FOCUS_HI" \
    --beta-focus-prob "$BETA_FOCUS_PROB" \
    --seed "$SEED" \
    --log-every "$LOG_EVERY" \
    --ckpt-every "$CKPT_EVERY" \
    $([ "$RESUME" = "1" ] || echo --no-resume)

ELAPSED=$SECONDS
echo
echo "=========================================================="
echo "finished     : $(date)"
echo "elapsed      : $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s"
echo "sec/step     : $(python -c "print(f'{$ELAPSED/$STEPS:.5f}')")"
echo "checkpoint   : $CKPT_ROOT/${RUN_NAME}.pt"
echo "=========================================================="
ls -la "$CKPT_ROOT/${RUN_NAME}.pt" || echo "WARNING: checkpoint not found!"

# ---------------------------------------------------------------------------
# NEXT, once the checkpoint exists:
#
#   python scripts/run_sudoku9_reward_tilt_check.py \
#       --ckpt-dir ckpt \
#       --pattern 'sudoku9_v1_*.pt' --taus 0,3,5,8 --n 2000 \
#       --K 1000 --schedule sstar \
#       --out results/sudoku9_v1_reward_tilt.csv
#
# Read tau_eff across tau before reading the validity rate. Flat in tau means
# the tilt has saturated and the checkpoint is not delivering what was asked
# for, whatever the headline number says.
# ---------------------------------------------------------------------------
