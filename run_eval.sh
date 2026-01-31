#!/bin/bash
# Evaluation script for TinyRecursiveModels
# Usage: ./run_eval.sh <checkpoint_path> [data_path] [additional_args...]
#
# Examples:
#   ./run_eval.sh checkpoints/MyProject/my-run/step_100
#   ./run_eval.sh checkpoints/MyProject/my-run/step_100 data/sudoku-test-100-aug-10

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found."
    exit 1
fi

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: ./run_eval.sh <checkpoint_path> [data_path] [additional_args...]"
    echo ""
    echo "Example: ./run_eval.sh checkpoints/MyProject/my-run/step_100"
    exit 1
fi

# Disable wandb by default
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Disable torch.compile by default
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

CHECKPOINT_PATH="$1"
DATA_PATH="${2:-data/sudoku-test-100-aug-10}"
shift 2 2>/dev/null || shift 1

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Dataset not found at $DATA_PATH"
    exit 1
fi

echo "=== TinyRecursiveModels Evaluation ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset: $DATA_PATH"
echo ""

# Run evaluation only (epochs=0 means only eval, no training)
python pretrain.py \
    arch=trm \
    "data_paths=[${DATA_PATH}]" \
    load_checkpoint="$CHECKPOINT_PATH" \
    global_batch_size=32 \
    epochs=1 \
    eval_interval=1 \
    min_eval_interval=0 \
    arch.mlp_t=True \
    arch.L_layers=2 \
    arch.H_cycles=3 \
    arch.L_cycles=6 \
    "$@"
