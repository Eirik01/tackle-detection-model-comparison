#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=sweep_temporal_vjepa2
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --array=0-19
#SBATCH --output=slurm_logs/sweep/temporal/vjepa2_%A_%a.out

# LR/WD grid sweep for the V-JEPA 2 attentive probe (temporal).
#
# Grid (V-JEPA 2 K400/SSv2 multihead_kwargs): 5 LR x 4 WD = 20 combos. The
# official multihead wrapper is not vendored, so each combo is a separate run
# (per-combo runs, per the agreed plan). Each array task trains ONE combo on
# the fixed train/val split. No test eval here.
#
# V-JEPA 2 dense features are ~290 MB compressed .npz, decompressed on cache
# miss -- I/O-bound. 20 array tasks reading the same feature dir in parallel
# will stress the shared FS; consider throttling concurrency with
# `--array=0-19%6` if the cluster FS struggles.
#
# Usage:
#   sbatch sweep_temporal_vjepa2.sh             # defaults (seed 42, 30 epochs)
#   sbatch sweep_temporal_vjepa2.sh 42 30       # seed epochs
#
# After all 20 tasks finish:
#   uv run python -m src.select_hparams \
#       --sweep-dir sweeps/vjepa2_l/seed42 --pipeline temporal

# --- Common setup ---
source setup.sh

mkdir -p slurm_logs/sweep/temporal

SEED=${1:-42}
NUM_EPOCHS=${2:-30}
BATCH_SIZE=${3:-64}

BACKBONE_TYPE=vjepa2
BACKBONE_SIZE="large"
WINDOW_SIZE=10
FPS=5.0
SOURCE_FPS=5.0
PROTOCOL=${PROTOCOL:-centred}
FEATURE_CACHE=${FEATURE_CACHE:-6}
NUM_WORKERS=${NUM_WORKERS:-6}
# Extraction flavour to load: reflect-padded dense files by default. Set
# PADDING_MODE=center_crop to load the centre-crop files instead.
PADDING_MODE=${PADDING_MODE:-reflect}

# V-JEPA 2 attentive-probe grid (AdamW + cosine).
LRS=(1e-4 3e-4 1e-3 3e-3 5e-3)
WDS=(1e-2 1e-1 0.4 0.8)
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
echo "V-JEPA 2 attentive probe -- sweep task ${IDX}"
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
