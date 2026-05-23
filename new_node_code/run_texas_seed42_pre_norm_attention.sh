#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_results/texas_seed42_pre_norm_attention_best_models}"
SEED="${SEED:-42}"
SPLIT_INDEX="${SPLIT_INDEX:-0}"
EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-200}"

PRE_NORM_DIR="${OUTPUT_ROOT}/pre_norm_attention"
mkdir -p "$PRE_NORM_DIR"

run_symmetric() {
  "$PYTHON_BIN" NodeClassification_CPU.py \
    --dataset Texas \
    --split-index "$SPLIT_INDEX" \
    --num-layers 3 \
    --hop 3 \
    --wl 3 \
    --dim_hidden 32 \
    --lr 0.01 \
    --dropout 0.2 \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --numheads 1 \
    --GL_k 5 \
    --batch_size 32 \
    --kernels WL \
    --isgnn True \
    --cluster-wl-features \
    --n-clusters 64 \
    --seed "$SEED" \
    --outdir "${OUTPUT_ROOT}/symmetric_seed_${SEED}_run" \
    --params-str "True_1_False_WL_32_3_5_3l_3h_clustered_C64_0.2_0.01_32" \
    --save-pre-norm-attention \
    --pre-norm-attention-output-dir "$PRE_NORM_DIR" \
    --skip-curves
}

run_asymmetric() {
  "$PYTHON_BIN" NodeClassification_CPU.py \
    --dataset Texas \
    --split-index "$SPLIT_INDEX" \
    --num-layers 1 \
    --hop 3 \
    --wl 5 \
    --dim_hidden 128 \
    --lr 0.01 \
    --dropout 0.0 \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --numheads 1 \
    --GL_k 5 \
    --batch_size 32 \
    --kernels WL \
    --isgnn True \
    --cluster-wl-features \
    --n-clusters 16 \
    --seed "$SEED" \
    --outdir "${OUTPUT_ROOT}/asymmetric_seed_${SEED}_run" \
    --params-str "True_1_False_WL_128_5_5_1l_3h_clustered_C16_0.0_0.01_32" \
    --asymmetric-gate \
    --save-pre-norm-attention \
    --pre-norm-attention-output-dir "$PRE_NORM_DIR" \
    --skip-curves
}

echo "Running Texas symmetric seed=${SEED} pre-norm attention export..."
run_symmetric

echo "Running Texas asymmetric seed=${SEED} pre-norm attention export..."
run_asymmetric

echo "Done. Pre-norm attention files are in: ${PRE_NORM_DIR}"
