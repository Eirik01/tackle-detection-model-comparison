#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=sweep_temporal_dinov3
#SBATCH --partition=accel
#SBATCH --gres=gpu:a100_80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-17%6
#SBATCH --output=slurm_logs/sweep/temporal/dinov3_%A_%a.out

# --gres=gpu:a100_80:1 pins each task to an 80 GB A100 (nodes gpu-7/8/9).
# Plain --gres=gpu:1 ended up sharing physical GPUs on the 40 GB a100 nodes
# (gpu-1/2/13) -- one prior run OOM'd with 128 MiB free out of 40 GiB on
# gpu-13. a100_80 nodes enforce isolation correctly. %6 throttle stays as a
# polite cap on the shared filesystem.

# LR/WD grid sweep for the DINOv3 attentive probe (temporal).
#
# Grid (DINOv3 paper, video attentive-probe protocol): 6 LR x 3 WD = 18 combos.
# Each array task trains ONE combo on the fixed train/val split. The checkpoint
# goes to TACDEC_MODELS keyed by --model-suffix; the train JSON (incl.
# best_val_macro_f1) goes to the per-combo sweep dir. No test eval here.
#
# Usage:
#   sbatch sweep_temporal_dinov3.sh             # defaults (seed 42, 30 epochs)
#   sbatch sweep_temporal_dinov3.sh 42 30       # seed epochs
#
# After all 18 tasks finish:
#   uv run python -m src.select_hparams \
#       --sweep-dir sweeps/dinov3_l/seed42 --pipeline temporal

# --- Common setup ---
source setup.sh

mkdir -p slurm_logs/sweep/temporal

SEED=${1:-42}
NUM_EPOCHS=${2:-30}
BATCH_SIZE=${3:-64}

BACKBONE_TYPE=dinov3
BACKBONE_SIZE="large"
WINDOW_SIZE=10
FPS=5.0
SOURCE_FPS=25.0
PROTOCOL=${PROTOCOL:-centred}
FEATURE_CACHE=${FEATURE_CACHE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
# Extraction flavour to load: reflect-padded dense files by default. Set
# PADDING_MODE=center_crop to load the centre-crop files instead.
PADDING_MODE=${PADDING_MODE:-reflect}

# DINOv3 video attentive-probe grid (AdamW + cosine).
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3)
WDS=(1e-3 1e-2 1e-1)
N_WD=${#WDS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
LR=${LRS[$((IDX / N_WD))]}
WD=${WDS[$((IDX % N_WD))]}

BACKBONE_ID="${BACKBONE_TYPE}_${BACKBONE_SIZE:0:1}"
MODEL_SUFFIX="sweep_seed${SEED}_lr${LR}_wd${WD}"
RUN_DIR="sweeps/${BACKBONE_ID}/seed${SEED}/lr${LR}_wd${WD}"
TRAIN_INFO="${RUN_DIR}/train.json"
mkdir -p "${RUN_DIR}"

echo "=========================================="
echo "DINOv3 attentive probe -- sweep task ${IDX}"
echo "=========================================="
echo "LR / WD:       ${LR} / ${WD}"
echo "Seed:          ${SEED}"
echo "Epochs:        ${NUM_EPOCHS}  batch=${BATCH_SIZE}"
echo "Model suffix:  ${MODEL_SUFFIX}"
echo "Train info:    ${TRAIN_INFO}"
echo "=========================================="

uv run python -u src/train_temporal.py \
    --backbone-type ${BACKBONE_TYPE} \
    --backbone-size ${BACKBONE_SIZE} \
    --window-size ${WINDOW_SIZE} \
    --fps ${FPS} \
    --source-fps ${SOURCE_FPS} \
    --protocol ${PROTOCOL} \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LR} \
    --weight-decay ${WD} \
    --patience 5 \
    --model-suffix ${MODEL_SUFFIX} \
    --seed ${SEED} \
    --feature-cache ${FEATURE_CACHE} \
    --num-workers ${NUM_WORKERS} \
    --padding-mode ${PADDING_MODE} \
    --save-info ${TRAIN_INFO}

echo ""
echo "Sweep task ${IDX} complete (lr=${LR} wd=${WD}). Train info: ${TRAIN_INFO}"
