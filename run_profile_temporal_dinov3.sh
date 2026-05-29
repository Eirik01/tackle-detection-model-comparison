#!/bin/bash
# --- Head-only inference-cost profiler: DINOv3 attentive probe ---
# Re-runs the existing centred-eval for the trained attentive probe and then
# appends one row to results/head_efficiency.csv. Pinned to RTX 3090 to match
# the spatial and V-JEPA 2 profiler runs.
#SBATCH --account=ec12
#SBATCH --job-name=profile_temporal_dinov3
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/profile/temporal_dinov3_%j.out

source setup.sh

mkdir -p slurm_logs/profile results

# Usage:
#   sbatch run_profile_temporal_dinov3.sh                    # defaults to centered_v1
#   sbatch run_profile_temporal_dinov3.sh <model_suffix>
BACKBONE_TYPE=dinov3
BACKBONE_SIZE=large
MODEL_SUFFIX=${1:-centered_v1}
WINDOW_SIZE=10
FPS=5.0
SEED=42
PROTOCOL=${PROTOCOL:-centered}
METRIC=${METRIC:-balanced}
FEATURE_CACHE=${FEATURE_CACHE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
PADDING_MODE=${PADDING_MODE:-reflect}

BACKBONE_ID="${BACKBONE_TYPE}_${BACKBONE_SIZE:0:1}"

echo "=========================================="
echo "DINOv3 attentive probe -- head efficiency"
echo "=========================================="
echo "Backbone:      ${BACKBONE_TYPE} (${BACKBONE_SIZE})"
echo "Model suffix:  ${MODEL_SUFFIX}"
echo "Window / FPS:  W=${WINDOW_SIZE} / ${FPS}"
echo "Protocol:      ${PROTOCOL}"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

uv run python -u src/eval_temporal.py \
    --backbone-type ${BACKBONE_TYPE} \
    --backbone-size ${BACKBONE_SIZE} \
    --model-suffix ${MODEL_SUFFIX} \
    --window-size ${WINDOW_SIZE} \
    --fps ${FPS} \
    --protocol ${PROTOCOL} \
    --metric ${METRIC} \
    --seed ${SEED} \
    --feature-cache ${FEATURE_CACHE} \
    --num-workers ${NUM_WORKERS} \
    --padding-mode ${PADDING_MODE} \
    --profile-efficiency

echo "=========================================="
echo "Profiling complete. Row appended to results/head_efficiency.csv"
echo "=========================================="
