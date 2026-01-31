#!/bin/bash
# Training script for TinyRecursiveModels
# Usage: ./run_train.sh [data_path] [arch] [additional_args...]
#
# Available architectures:
#   trm       - Full model (512 hidden, ~7M params)
#   trm_small - Small model (256 hidden, ~2M params)
#   trm_tiny  - Tiny model (128 hidden, ~500K params) - fastest
#
# Examples:
#   ./run_train.sh                                           # Use defaults (tiny model)
#   ./run_train.sh data/sudoku-test-100-aug-10              # Specify dataset
#   ./run_train.sh data/sudoku-test-100-aug-10 trm_small    # Use small model
#   ./run_train.sh data/sudoku-test-100-aug-10 trm epochs=1000
#
# For multi-GPU training, use torchrun directly:
#   torchrun --nproc-per-node 4 pretrain.py arch=trm data_paths="[data/your-dataset]"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Resolve python from local venv (avoid hardcoded activate paths)
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
else
    echo "Error: Virtual environment not found or missing python. Please run: uv venv .venv && uv pip install -r requirements.txt"
    exit 1
fi

# Default dataset path and architecture
DATA_PATH="${1:-data/sudoku-test-50-aug-5}"
ARCH="${2:-trm_tiny}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true

# Validate architecture - if it looks like a hydra override, treat as extra arg
if [[ "$ARCH" == *"="* ]]; then
    set -- "$ARCH" "$@"
    ARCH="trm_tiny"
fi

# Check if dataset exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH"
    echo "Run ./run_prepare_data.sh first to create a test dataset"
    exit 1
fi

echo "=== TinyRecursiveModels Training ==="
echo "Dataset: $DATA_PATH"
echo "Architecture: $ARCH"
echo "Additional args: $@"
echo ""

# Disable wandb by default (set WANDB_MODE=online to enable)
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Disable torch.compile by default (requires python3-dev for Triton JIT)
# Set DISABLE_COMPILE="" to enable compilation
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

# Default training parameters for smoke test (quick run)
# Override by passing arguments like: epochs=1000 eval_interval=500
"$PYTHON_BIN" pretrain.py \
    arch="$ARCH" \
    "data_paths=[${DATA_PATH}]" \
    global_batch_size=32 \
    epochs=100 \
    eval_interval=50 \
    "$@"
