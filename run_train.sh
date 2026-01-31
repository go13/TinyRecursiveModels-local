#!/bin/bash
# Training script for TinyRecursiveModels
# Usage: ./run_train.sh [data_path] [additional_args...]
#
# Examples:
#   ./run_train.sh                                    # Use default test dataset
#   ./run_train.sh data/sudoku-test-100-aug-10       # Specify dataset
#   ./run_train.sh data/sudoku-test-100-aug-10 epochs=1000 eval_interval=500
#
# For multi-GPU training, use torchrun directly:
#   torchrun --nproc-per-node 4 pretrain.py arch=trm data_paths="[data/your-dataset]"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run: uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt"
    exit 1
fi

# Default dataset path
DATA_PATH="${1:-data/sudoku-test-100-aug-10}"
shift 2>/dev/null || true

# Check if dataset exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH"
    echo "Run ./run_prepare_data.sh first to create a test dataset"
    exit 1
fi

echo "=== TinyRecursiveModels Training ==="
echo "Dataset: $DATA_PATH"
echo "Additional args: $@"
echo ""

# Disable wandb by default (set WANDB_MODE=online to enable)
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Disable torch.compile by default (requires python3-dev for Triton JIT)
# Set DISABLE_COMPILE="" to enable compilation
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

# Default training parameters for smoke test (quick run)
# Override by passing arguments like: epochs=1000 eval_interval=500
python pretrain.py \
    arch=trm \
    "data_paths=[${DATA_PATH}]" \
    global_batch_size=32 \
    epochs=100 \
    eval_interval=50 \
    arch.mlp_t=True \
    arch.L_layers=2 \
    arch.H_cycles=3 \
    arch.L_cycles=6 \
    "$@"
