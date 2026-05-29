#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=eval_temporal_sliding
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/eval/temporal_sliding/%j.out

# --- Common setup ---
source setup.sh

mkdir -p slurm_logs/eval/temporal_sliding results/temporal

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate attentive probe with stride-1 frame-level classification
# (TempTAC-comparable protocol: sliding windows across full test clips)
#
# Usage:
#   sbatch run_eval_temporal_sliding.sh dinov3 kassab_attn_v1
#   sbatch run_eval_temporal_sliding.sh vjepa2 kassab_attn_v1
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE_TYPE=${1:-vjepa2}
MODEL_SUFFIX=${2:-kassab_attn_v1}
BACKBONE_SIZE="large"
WINDOW_SIZE=10
FPS=5.0
SEED=42

BACKBONE_ID="${BACKBONE_TYPE}_${BACKBONE_SIZE:0:1}"
EVAL_OUT="results/temporal/${BACKBONE_ID}_${MODEL_SUFFIX}_sliding.json"

echo "=========================================="
echo "Stride-1 frame-level evaluation (sliding)"
echo "=========================================="
echo "Backbone:      ${BACKBONE_TYPE} (${BACKBONE_SIZE})"
echo "Model suffix:  ${MODEL_SUFFIX}"
echo "Window / FPS:  W=${WINDOW_SIZE} / ${FPS}"
echo "Eval JSON:     ${EVAL_OUT}"
echo "=========================================="

uv run python -u src/eval_temporal.py \
    --backbone-type ${BACKBONE_TYPE} \
    --backbone-size ${BACKBONE_SIZE} \
    --model-suffix ${MODEL_SUFFIX} \
    --window-size ${WINDOW_SIZE} \
    --fps ${FPS} \
    --metric sliding \
    --seed ${SEED} \
    --feature-cache 16 \
    --save-json ${EVAL_OUT}

echo
echo "=========================================="
echo "Sliding window evaluation complete."
echo "  Results saved to: ${EVAL_OUT}"
echo "=========================================="
