#!/bin/bash
# ============================================================================
# One-time environment setup (login node, or any machine; no GPU needed).
#
#   bash hpc/setup_env.sh [MODULE] [VENV_PATH]
#
# MODULE    optional cluster module providing a CUDA-matched PyTorch build
#           (e.g. the output of `module avail pytorch`). Leave empty to skip.
# VENV_PATH where to create the virtual environment (default: ./.venv).
#
# If MODULE is given the venv is created with --system-site-packages so it
# inherits the module's torch/CUDA build instead of downloading a new one.
# The remaining dependencies come from requirements.txt at the repo root.
# ============================================================================

set -euo pipefail

MODULE="${1:-}"
ENV_PATH="${2:-$PWD/.venv}"

echo "=== environment setup ==="
echo "module   : ${MODULE:-<none>}"
echo "venv     : $ENV_PATH"
echo

if [ -n "$MODULE" ]; then
    module load "$MODULE"
    VENV_FLAGS="--system-site-packages"
else
    VENV_FLAGS=""
fi

if [ -d "$ENV_PATH" ]; then
    echo "venv already exists at $ENV_PATH -- skipping creation."
else
    mkdir -p "$(dirname "$ENV_PATH")"
    # shellcheck disable=SC2086
    python3 -m venv $VENV_FLAGS "$ENV_PATH"
    echo "created venv at $ENV_PATH"
fi

# shellcheck disable=SC1091
source "$ENV_PATH/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo
echo "--- verification ---"
python - <<'PY'
import torch, numpy, pandas, scipy, matplotlib
print("torch      :", torch.__version__, "| cuda:", torch.version.cuda,
      "| gpu available:", torch.cuda.is_available())
print("numpy      :", numpy.__version__)
print("pandas     :", pandas.__version__)
print("scipy      :", scipy.__version__)
print("matplotlib :", matplotlib.__version__)
PY

echo
echo "=== done ==="
echo "Jobs activate this env with:"
[ -n "$MODULE" ] && echo "    module load $MODULE"
echo "    source $ENV_PATH/bin/activate"
