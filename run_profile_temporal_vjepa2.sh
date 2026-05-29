#!/bin/bash
# --- Head-only inference-cost profiler: V-JEPA 2 attentive probe ---
# Re-runs the existing centred-eval for the trained attentive probe and then
# appends one row to results/head_efficiency.csv. Pinned to RTX 3090 to match
# the spatial and DINOv3 profiler runs.
#
# Memory budget matches run_train_eval_temporal_vjepa2.sh (6w / cache=6 -> ~20 GiB).
#SBATCH --account=ec12
#SBATCH --job-name=profile_temporal_vjepa2
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/profile/temporal_vjepa2_%j.out

source setup.sh

mkdir -p slurm_logs/profile results

# Usage:
#   sbatch run_profile_temporal_vjepa2.sh                    # defaults to centered_v1
#   sbatch run_profile_temporal_vjepa2.sh <model_suffix>
BACKBONE_TYPE=vjepa2
BACKBONE_SIZE=large
MODEL_SUFFIX=${1:-centered_v1}
WINDOW_SIZE=10
FPS=5.0
SEED=42
PROTOCOL=${PROTOCOL:-centered}
METRIC=${METRIC:-balanced}
FEATURE_CACHE=${FEATURE_CACHE:-6}
NUM_WORKERS=${NUM_WORKERS:-6}
PADDING_MODE=${PADDING_MODE:-reflect}

BACKBONE_ID="${BACKBONE_TYPE}_${BACKBONE_SIZE:0:1}"

echo "=========================================="
echo "V-JEPA 2 attentive probe -- head efficiency"
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
