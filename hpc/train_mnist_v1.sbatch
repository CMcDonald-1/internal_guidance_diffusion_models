#!/bin/bash
# ============================================================================
# Class-conditional MNIST training (10-bit one-hot reward block, pattern tilt).
#
#   PILOT (256x4, ~3.7M params, 30k steps, ~1k epochs):
#     sbatch --array=0-0 -t 04:00:00 hpc/train_mnist_v1.sbatch xf_nosc
#
#   MAIN (512x8, ~27M params, 100k steps):
#     sbatch --export=ALL,D_MODEL=512,N_LAYERS=8,STEPS=100000,BATCH_SIZE=4096 \
#         --array=0-0 -t 16:00:00 hpc/train_mnist_v1.sbatch xf_sc
#
# Inline env must go through --export (the bare VAR=val prefix silently runs
# defaults on some Slurm clusters -- same trap as every previous script).
#
# WHAT IS DIFFERENT FROM sudoku9, ON PURPOSE
# ------------------------------------------
# LAMBDA_MAX CEILING IS sqrt(d/10) = 25.064, NOT sqrt(d) = 79.3. The reward
# block is 10 coordinates stretched together, so the budget identity spends
# 10*lambda^2; the core scale hits zero at 25.064, and that endpoint IS the
# t=0 pattern-tilt query. Default 24.0 = 96% of the ceiling (the sudoku
# close-beneath-the-endpoint fraction against the CORRECT ceiling).
# run_train_mnist_reward.py refuses >= ceiling and warns below 0.8x.
#
# MODE1_PROB=0. This experiment never issues an infill query; the tilted-draw
# budget is already spread over 10 class patterns. Inpainting is the v2 run.
#
# HOLDOUT IS ON AND IS THE POINT. 60k images x STEPS*BATCH draws is ~1k-7k
# epochs -- the memorisation regime the sudoku runs never had. The runner
# freezes a 4096-row batch from the official 10k test split and scores it
# every 1000 steps; watch train-vs-holdout diverge in the
# training log (holdout_mean_sse) and keep every partial checkpoint. The loss is
# still only a liveness check; the holdout GAP is the signal here.
#
# DATASET IS NOT BUILT HERE and needs internet for the IDX download, which
# compute nodes do not have. Build it ONCE from a login node:
#     bash hpc/make_data_mnist_v1.sh
#
# Cells:
#   xf_nosc / xf_sc     TransformerTilt, 49 tokens x 128 bits (4x4-pixel
#                       patches x 8 Gray bits) + the 10-bit class block.
#                       No MLP cells: the flat arm posted 0.0000 at d=325.
# ============================================================================

#SBATCH --job-name=mn1
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gpus=1                      # paper runs: one 80GB H100
#SBATCH --ntasks-per-node=12
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

CELL="${1:?usage: sbatch --array=0-0 train_mnist_v1.sbatch <xf_nosc|xf_sc>}"

AI_MODULE="${AI_MODULE:-}"          # optional: cluster PyTorch module to load
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repo root
VENV_PATH="${VENV_PATH:-$CODE_DIR/.venv}"
DATA_ROOT="${DATA_ROOT:-$CODE_DIR/data}"
CKPT_ROOT="${CKPT_ROOT:-$CODE_DIR/ckpt}"

DATASET_NAME="${DATASET_NAME:-mnist_v1}"
DATA_DIR="$DATA_ROOT/$DATASET_NAME"

STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
LR=1e-3
LAMBDA_MAX="${LAMBDA_MAX:-24.0}"     # must stay < sqrt(6282/10) = 25.0639
MODE1_PROB="${MODE1_PROB:-0.0}"      # BY DESIGN -- see header
LAMBDA1_PROB=0.25
BETA_FOCUS_PROB="${BETA_FOCUS_PROB:-0.0}"
SELF_COND_PROB=0.5
HOLDOUT_N="${HOLDOUT_N:-4096}"
HOLDOUT_EVERY="${HOLDOUT_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-500}"
CKPT_EVERY="${CKPT_EVERY:-5000}"
RESUME="${RESUME:-1}"
AMP="${AMP:-bf16}"                   # fp32 measured ~1.05 s/step at 512x8/4096
TF32="${TF32:-on}"

# transformer geometry
XF_SEQ="${XF_SEQ:-49}"               # 7x7 grid of 4x4-pixel patches
XF_BITS="${XF_BITS:-128}"            # 196x32 (2x2 patches) is the open ablation
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-4}"
N_HEADS="${N_HEADS:-8}"

SEEDS=(8264 1337 20260828)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#SEEDS[@]}" ]; then
    echo "ERROR: array index $IDX exceeds the ${#SEEDS[@]} seeds defined." >&2
    exit 1
fi
SEED="${SEEDS[$IDX]}"

case "$CELL" in
    xf_nosc) CELL_ARGS="" ;;
    xf_sc)   CELL_ARGS="--selfcond --self-cond-prob $SELF_COND_PROB" ;;
    *) echo "ERROR: unknown cell '$CELL'" >&2; exit 1 ;;
esac

LTAG="l$(printf '%g' "$LAMBDA_MAX" | tr -d '.')"
STEPS_TAG="$(python3 -c "print(f'{$STEPS//1000}k' if $STEPS % 1000 == 0 else '$STEPS')")"
# TAG before the cell so load_ckpt's _(xf)_(sc|nosc)_s<seed>_<steps>k.pt$
# regex keeps matching (same reasoning as the sudoku9 script).
TAG="${TAG:-d${D_MODEL}x${N_LAYERS}}"
RUN_NAME="${DATASET_NAME}_${LTAG}_${TAG}_${CELL}_s${SEED}_${STEPS_TAG}"

if [ -f "$CKPT_ROOT/${RUN_NAME}.pt" ]; then
    if [ "$RESUME" = "1" ]; then
        echo "$CKPT_ROOT/${RUN_NAME}.pt already exists -- run already finished."
        exit 0
    fi
    echo "ERROR: $CKPT_ROOT/${RUN_NAME}.pt already exists -- refusing to overwrite." >&2
    exit 1
fi

echo "=========================================================="
echo "job id       : $SLURM_JOB_ID  (array ${SLURM_ARRAY_JOB_ID:-n/a} task ${IDX})"
echo "node         : $(hostname)"
echo "started      : $(date)"
echo "cell         : $CELL   (args: ${CELL_ARGS:-<none>})"
echo "geometry     : ${XF_SEQ}x${XF_BITS} tokens, d_model=$D_MODEL x $N_LAYERS layers"
echo "precision    : amp=$AMP  tf32=$TF32"
echo "lambda_max   : $LAMBDA_MAX   (ceiling sqrt(6282/10) = 25.0639)"
echo "steps/batch  : $STEPS / $BATCH_SIZE"
echo "seed         : $SEED"
echo "run name     : $RUN_NAME"
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

if [ ! -f "$DATA_DIR/images.npy" ]; then
    echo "ERROR: dataset not found at $DATA_DIR" >&2
    echo "       Build it ONCE from a LOGIN node (compute nodes have no" >&2
    echo "       internet): bash hpc/make_data_mnist_v1.sh" >&2
    exit 1
fi

SECONDS=0

# shellcheck disable=SC2086
python scripts/run_train_mnist_reward.py $CELL_ARGS \
    --data "$DATA_DIR" \
    --out "$CKPT_ROOT/${RUN_NAME}.pt" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --xf-seq "$XF_SEQ" \
    --xf-bits "$XF_BITS" \
    --d-model "$D_MODEL" \
    --n-layers "$N_LAYERS" \
    --n-heads "$N_HEADS" \
    --amp "$AMP" \
    --tf32 "$TF32" \
    --lambda-max "$LAMBDA_MAX" \
    --mode1-prob "$MODE1_PROB" \
    --lambda1-prob "$LAMBDA1_PROB" \
    --beta-focus-prob "$BETA_FOCUS_PROB" \
    --holdout-n "$HOLDOUT_N" \
    --holdout-every "$HOLDOUT_EVERY" \
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
