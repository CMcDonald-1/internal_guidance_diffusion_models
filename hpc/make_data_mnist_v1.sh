#!/bin/bash
# Build data/mnist_v1 ONCE, from a LOGIN node (needs internet for the
# 11 MB IDX download; compute nodes have none). Not an sbatch on purpose:
# it is a 30-second CPU job and the download is the only real work.
#
#   bash hpc/make_data_mnist_v1.sh
set -euo pipefail

AI_MODULE="${AI_MODULE:-}"          # optional: cluster PyTorch module to load
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repo root
VENV_PATH="${VENV_PATH:-$CODE_DIR/.venv}"
DATA_ROOT="${DATA_ROOT:-$CODE_DIR/data}"

if [ -n "$AI_MODULE" ]; then module load "$AI_MODULE"; fi
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
cd "$CODE_DIR"

python scripts/make_mnist_data.py \
    --out "$DATA_ROOT/mnist_v1" \
    --cache "$DATA_ROOT/mnist_raw"

python - "$DATA_ROOT/mnist_v1" <<'PY'
import json, sys
meta = json.load(open(f"{sys.argv[1]}/meta.json"))
assert meta["d"] == 6282, meta["d"]
print(f"OK: n={meta['n']:,} holdout={meta['holdout_n']:,} d={meta['d']} "
      f"encoding={meta['encoding']}")
PY
