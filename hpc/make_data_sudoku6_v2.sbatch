#!/bin/bash
# ============================================================================
# Build the high-diversity sudoku6 dataset.
#
#   sbatch hpc/make_data_sudoku6_v2.sbatch
#   N=20000000 OUTDIR=data/sudoku6_v3 sbatch hpc/make_data_sudoku6_v2.sbatch
#
# This is a CPU-bound job -- it never touches the GPU. A CPU partition is the
# natural home; it is ~2 minutes, so requesting the smallest GPU partition
# instead costs very little (~0.03 GPU-hours).
#
# It is also short enough to run directly on a LOGIN NODE for free, no job at
# all -- see the block at the bottom of this file for that command.
#
# The old generator was ~240 ms per injected valid solution (150k rows took 30
# minutes, essentially all of it in 7,512 calls to sample_valid_solutions).
# sudoku6_fast does the same exact-uniform draw in ~3 us via the (39,168
# reduced grids) x (720 relabellings) bijection, so 5M rows is a couple of
# minutes of one core.
#
# WHY 5M: memorisation is driven by how many times each DISTINCT valid grid is
# seen, not by the reward rate. At 150k rows that was 7,511 grids seen ~9,800
# times each over a 180k-step run, and the trained model reproduced 33.6% of
# its valid outputs verbatim (2,522x chance). 5M rows gives ~250k distinct
# grids at ~295 exposures -- a 33x reduction. p_inject stays at 0.05: the
# reward must remain rare, that is the whole point of tilting.
# ============================================================================

#SBATCH --job-name=s6data
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

AI_MODULE="${AI_MODULE:-}"          # optional: cluster PyTorch module to load
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repo root
VENV_PATH="${VENV_PATH:-$CODE_DIR/.venv}"

N="${N:-5000000}"
OUTDIR="${OUTDIR:-data/sudoku6_v2}"
SEED="${SEED:-5209}"
ROW_NOISE="${ROW_NOISE:-0.02}"
P_INJECT="${P_INJECT:-0.05}"
FORMAT="${FORMAT:-npy}"

echo "=========================================================="
echo "job id   : $SLURM_JOB_ID"
echo "node     : $(hostname)"
echo "started  : $(date)"
echo "n        : $N"
echo "outdir   : $OUTDIR"
echo "p_inject : $P_INJECT   (unchanged -- the reward stays rare)"
echo "format   : $FORMAT"
echo "=========================================================="

if [ -n "$AI_MODULE" ]; then module load "$AI_MODULE"; fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

cd "$CODE_DIR"
mkdir -p "$(dirname "$OUTDIR")"

if [ -e "$OUTDIR/grids.npy" ] || [ -e "$OUTDIR/train.csv" ]; then
    echo "ERROR: $OUTDIR already holds a dataset -- refusing to overwrite." >&2
    echo "       Move it aside or set OUTDIR to a new path." >&2
    exit 1
fi

SECONDS=0

# Warm the reduced-grid cache first so the timing below is generation, not
# the one-off 14 s enumeration. Cached next to tilt/sudoku6_fast.py.
python - <<'PY'
from tilt.sudoku6_fast import reduced_grids
g = reduced_grids()
print(f"### reduced-grid cache ready: {len(g):,} grids "
      f"(expect 39,168 = 28,200,960 / 720)", flush=True)
PY

python scripts/make_sudoku6_data.py \
    --n "$N" \
    --outdir "$OUTDIR" \
    --seed "$SEED" \
    --row-noise "$ROW_NOISE" \
    --p-inject "$P_INJECT" \
    --format "$FORMAT"

echo
echo "=== verifying ==="
python - "$OUTDIR" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
meta = json.loads((d / "meta.json").read_text())
print(f"  n              : {meta['n']:,}")
print(f"  d              : {meta['d']}")
print(f"  success_rate   : {meta['success_rate']:.4%}   (want ~5%)")
print(f"  distinct_valid : {meta.get('distinct_valid', 0):,}")
exp = 180_000 * 8192 * meta["p_inject"] / max(meta.get("distinct_valid", 1), 1)
print(f"  exposures/grid : {exp:,.0f}   (was ~9,816 on sudoku6_v1)")
if meta["success_rate"] < 1e-6:
    sys.exit("ERROR: success_rate is 0 -- nothing to tilt against.")
if meta.get("distinct_valid", 0) < 0.5 * meta["n"] * meta["p_inject"]:
    sys.exit("ERROR: distinct_valid is far below n*p_inject -- the injected "
             "grids are repeating, which defeats the point of this rebuild.")
print("  OK")
PY

echo
echo "elapsed  : $SECONDS s"
ls -lh "$OUTDIR"

# ---------------------------------------------------------------------------
# FREE ALTERNATIVE: run it on a login node, no job and no allocation spent.
# Two minutes of one core is well inside what a login node tolerates.
#
#   source .venv/bin/activate
#   cd <repo root>
#   python scripts/make_sudoku6_data.py \
#       --n 5000000 \
#       --outdir data/sudoku6_v2 \
#       --seed 5209 --row-noise 0.02 --p-inject 0.05 --format npy
#
# Confirm it prints success_rate ~5% and distinct_valid ~250,000.
# ---------------------------------------------------------------------------
