#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${1:-data/sudoku-test-50-aug-5}
EPOCHS=${2:-1000}
EVAL_INTERVAL=${3:-200}
shift $(( $# > 3 ? 3 : $# )) || true
EXTRA_ARGS=("$@")

OUT_DIR="outputs/compare_logs"
mkdir -p "$OUT_DIR"
RESULTS_FILE="$(mktemp)"
trap 'rm -f "$RESULTS_FILE"' EXIT

get_metric() {
  local log_file="$1"
  local key="$2"
  awk -F': ' -v key="$key" '
    { gsub(/\r/, "", $0) }
    /^[[:space:]]*([A-Za-z_]+)[[:space:]]*:/ {
      k=$1; gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
      if (k==key) {val=$2}
    }
    END { if (val != "") print val }
  ' "$log_file"
}

run_model() {
  local label="$1"
  local arch="$2"
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local log_file="$OUT_DIR/${arch}_${stamp}.log"

  {
    echo "============================================================"
    echo "[${label}] arch=${arch}"
    echo "Dataset: ${DATA_PATH}"
    echo "epochs=${EPOCHS} eval_interval=${EVAL_INTERVAL}"
    if ((${#EXTRA_ARGS[@]})); then
      echo "extra: ${EXTRA_ARGS[*]}"
    fi
    echo "------------------------------------------------------------"
    echo "Training + evaluation in progress..."
  } >&2

  bash run_train.sh "$DATA_PATH" "$arch" epochs="$EPOCHS" eval_interval="$EVAL_INTERVAL" "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$log_file" >&2

  local acc exact lm qhalt steps
  acc=$(get_metric "$log_file" "accuracy")
  exact=$(get_metric "$log_file" "exact_accuracy")
  lm=$(get_metric "$log_file" "lm_loss")
  qhalt=$(get_metric "$log_file" "q_halt_accuracy")
  steps=$(get_metric "$log_file" "steps")

  printf "%s|%s|%s|%s|%s|%s\n" "$arch" "$acc" "$exact" "$lm" "$qhalt" "$steps" >> "$RESULTS_FILE"
}

run_model "1/3" "trm_overnight"
run_model "2/3" "iect_overnight"
run_model "3/3" "lbvs_overnight"

printf "\n%-16s | %-12s | %-14s | %-10s | %-16s | %-8s\n" "model" "accuracy" "exact_accuracy" "lm_loss" "q_halt_accuracy" "steps"
printf "%s\n" "------------------------------------------------------------------------------------------------"
while IFS='|' read -r model acc exact lm qhalt steps; do
  printf "%-16s | %-12s | %-14s | %-10s | %-16s | %-8s\n" "$model" "${acc:-NA}" "${exact:-NA}" "${lm:-NA}" "${qhalt:-NA}" "${steps:-NA}"
done < "$RESULTS_FILE"
