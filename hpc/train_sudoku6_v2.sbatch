#!/bin/bash
# ============================================================================
# Reward-tilted sudoku6 training on the HIGH-DIVERSITY dataset (sudoku6_v2).
#
#   sbatch --array=0-0 -t 02:30:00 hpc/train_sudoku6_v2.sbatch flat_sc
#   sbatch --array=0-0 -t 05:00:00 hpc/train_sudoku6_v2.sbatch xf_sc
#
# Separate from sweep_reward_2x2.sbatch on purpose: that script encodes a
# specific 2x2 factorial on sudoku6_v1 at lambda_max=5, and its checkpoint
# names say so. This one targets the rebuilt dataset, defaults
# to lambda_max=10 (measured: 0.605 -> 0.716 valid at tau=8, tau_eff 1.87 ->
# 2.04), and adds the transformer arm.
#
# Cells:
#   flat_nosc / flat_sc        MLP, flat featuriser
#   matrix_nosc / matrix_sc    MLP, row-grouped featuriser
#   xf_nosc / xf_sc            TransformerTilt, 6 tokens x 18 bits + reward
#
# EVERYTHING inline-env must go through --export, e.g.
#   sbatch --export=ALL,LAMBDA_MAX=10.0,STEPS=200 --array=0-0 ...
# The bare `VAR=val sbatch ...` prefix form does NOT survive submission on
# some Slurm clusters -- it silently runs the defaults instead.
# ============================================================================

#SBATCH --job-name=s6v2
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gpus=1                      # paper runs: one 80GB H100
#SBATCH --ntasks-per-node=12
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

CELL="${1:?usage: sbatch --array=0-0 train_sudoku6_v2.sbatch <flat_nosc|flat_sc|matrix_nosc|matrix_sc|xf_nosc|xf_sc>}"

AI_MODULE="${AI_MODULE:-}"          # optional: cluster PyTorch module to load
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repo root
VENV_PATH="${VENV_PATH:-$CODE_DIR/.venv}"
DATA_ROOT="${DATA_ROOT:-$CODE_DIR/data}"
CKPT_ROOT="${CKPT_ROOT:-$CODE_DIR/ckpt}"

DATASET_NAME="${DATASET_NAME:-sudoku6_v2}"
DATA_DIR="$DATA_ROOT/$DATASET_NAME"

STEPS="${STEPS:-180000}"
WIDTH=1024
BLOCKS=8
BATCH_SIZE=8192
LR=1e-3
LAMBDA_MAX="${LAMBDA_MAX:-10.0}"     # must stay < sqrt(109) = 10.4403
MODE1_PROB=0.25
LAMBDA1_PROB=0.25
BETA_FOCUS_LO=0.5
BETA_FOCUS_HI=2.5
BETA_FOCUS_PROB=0.3
SELF_COND_PROB=0.5
LOG_EVERY="${LOG_EVERY:-1000}"

# transformer geometry (ignored by the MLP cells)
XF_SEQ="${XF_SEQ:-6}"                # 6 tokens of 18 bits = one per grid row
XF_BITS="${XF_BITS:-18}"             # 36 x 3 (one per cell) also gives d=109
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-4}"
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
if [ "$STEPS" -lt 180000 ]; then
    RUN_NAME="SMOKE_${DATASET_NAME}_${LTAG}_${CELL}_s${SEED}_${STEPS}"
else
    RUN_NAME="${DATASET_NAME}_${LTAG}_${CELL}_s${SEED}_180k"
fi

if [ -f "$CKPT_ROOT/${RUN_NAME}.pt" ]; then
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
echo "lambda_max   : $LAMBDA_MAX"
echo "seed         : $SEED"
echo "run name     : $RUN_NAME"
echo "=========================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

if [ -n "$AI_MODULE" ]; then module load "$AI_MODULE"; fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

python -c "import torch; print('torch', torch.__version__, '| device', torch.cuda.get_device_name(0))"
echo

cd "$CODE_DIR"
mkdir -p "$CKPT_ROOT"

# --- PREREQUISITE: build the dataset first, once ----------------------------
#   sbatch hpc/make_data_sudoku6_v2.sbatch
# Never generated here: concurrent array tasks would race to write it.
if [ ! -f "$DATA_DIR/grids.npy" ] && [ ! -f "$DATA_DIR/train.csv" ]; then
    echo "ERROR: dataset not found at $DATA_DIR" >&2
    echo "       Run hpc/make_data_sudoku6_v2.sbatch first." >&2
    exit 1
fi

python - "$DATA_DIR" <<'PY'
import json, sys
meta = json.load(open(f"{sys.argv[1]}/meta.json"))
sr = meta.get("success_rate", 0.0)
dv = meta.get("distinct_valid")
print(f"### dataset d={meta['d']} n={meta['n']:,} seed={meta['seed']} "
      f"success_rate={sr:.4%} format={meta.get('format','csv')}")
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
python scripts/run_train_sudoku6_reward.py $CELL_ARGS \
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
    --log-every "$LOG_EVERY"

ELAPSED=$SECONDS
echo
echo "=========================================================="
echo "finished     : $(date)"
echo "elapsed      : $((ELAPSED/3600))h $((ELAPSED%3600/60))m $((ELAPSED%60))s"
echo "sec/step     : $(python -c "print(f'{$ELAPSED/$STEPS:.5f}')")"
echo "checkpoint   : $CKPT_ROOT/${RUN_NAME}.pt"
echo "=========================================================="
ls -la "$CKPT_ROOT/${RUN_NAME}.pt" || echo "WARNING: checkpoint not found!"
