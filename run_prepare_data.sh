#!/bin/bash
# Prepare a small Sudoku dataset for testing
# Usage: ./run_prepare_data.sh [subsample_size] [num_aug]

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

# Default values for a quick smoke test
SUBSAMPLE_SIZE="${1:-100}"
NUM_AUG="${2:-10}"
OUTPUT_DIR="data/sudoku-test-${SUBSAMPLE_SIZE}-aug-${NUM_AUG}"

echo "=== Preparing Sudoku Dataset ==="
echo "Subsample size: $SUBSAMPLE_SIZE"
echo "Augmentations: $NUM_AUG"
echo "Output directory: $OUTPUT_DIR"
echo ""

cd dataset
python build_sudoku_dataset.py \
    --output-dir "../$OUTPUT_DIR" \
    --subsample-size "$SUBSAMPLE_SIZE" \
    --num-aug "$NUM_AUG"

echo ""
echo "=== Dataset prepared successfully ==="
echo "Train data: $OUTPUT_DIR/train/"
echo "Test data: $OUTPUT_DIR/test/"
