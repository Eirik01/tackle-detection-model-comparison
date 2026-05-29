#!/bin/bash
# --- Slurm job parameters (V-JEPA 2 sizing) ---
# V-JEPA 2 dense features are 290 MB / video, compressed .npz that must be
# fully decompressed on cache miss. Past run with 2 workers: ~4.5 h wall, mostly
# I/O-bound. With 6 workers + persistent caches we expect ~1-1.5 h.
#
# Memory budget (calibrated from 2w/cache16 = 12.9 GiB):
#   mem ~= 1 + N_workers * (1.3 + cache * 0.29) GiB
#   6w/cache6  -> ~20 GiB peak  (fits 24 G with margin)  <-- current
#   6w/cache12 -> ~30 GiB peak  (OOM under 24 G — DO NOT)
#SBATCH --account=ec12
#SBATCH --job-name=train_eval_temporal_vjepa2
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/train_eval/temporal/vjepa2_%j.out

# --- Common setup ---
source setup.sh

mkdir -p slurm_logs/train_eval/temporal results/temporal

# Usage:
#   sbatch run_train_eval_temporal_vjepa2.sh                          # defaults (centered)
#   sbatch run_train_eval_temporal_vjepa2.sh centered_v2 50 64        # custom suffix / epochs / batch
#
# V-JEPA 2 only supports --protocol centered: its dense features are pre-extracted
# per clip, so the kassab_concat cross-clip parity protocol is DINOv3-only.

BACKBONE_TYPE=vjepa2
MODEL_SUFFIX=${1:-centered_v1}
NUM_EPOCHS=${2:-30}
BATCH_SIZE=${3:-64}
# Selected hyperparameters from sweeps/vjepa2_l/seed42/selection.json under
# the decoupled stop/save protocol (checkpoint on max val macro-F1, early-stop
# on val_loss with patience 5). Cell (3e-4, 1e-2) is the chosen pick with the highest plateau F1 among cells with a healthy
# train/val loss gap (gap = 0.115, val_loss = 0.448, plateau F1 = 0.855).
# The peak-F1 winner (5e-3, 0.4) was rejected as a single-epoch noise spike
# (plateau F1 0.819 vs peak 0.882, gap 0.79).
LEARNING_RATE=${4:-3e-4}
SEED=${5:-42}
WEIGHT_DECAY=${6:-1e-2}
PROTOCOL=${PROTOCOL:-centered}
# Caps are no-ops for V-JEPA 2 (centered ignores them); kept only so the shared
# --replay-cap/--bg-count flags below have values. Kassab Table 7.6 defaults.
REPLAY_CAP=${REPLAY_CAP:-280}
BG_COUNT=${BG_COUNT:-500}
FEATURE_CACHE=${FEATURE_CACHE:-6}
NUM_WORKERS=${NUM_WORKERS:-6}
# Padding mode used at extraction time. Default keeps the centre-crop files;
# set PADDING_MODE=center_crop to explicitly load the centre-crop files
PADDING_MODE=${PADDING_MODE:-reflect}
# Kassab parity knobs (no-ops at their defaults):
#   CE_WEIGHT_STYLE=balanced   uses sklearn 'balanced' inverse-frequency CE weights.
#   SPLIT_FILE=<path.json>     overrides the seeded game-disjoint split with a
#                              fixed clip-ID partition (see dump_kassab_split.py).
CE_WEIGHT_STYLE=${CE_WEIGHT_STYLE:-min1}
SPLIT_FILE=${SPLIT_FILE:-}

WINDOW_SIZE=10                # 5 FPS * 2 s (even -> tubelet=2 OK)
FPS=5.0
SOURCE_FPS=5.0                # V-JEPA 2 dense files are on-disk at 5 FPS
BACKBONE_SIZE="large"

BACKBONE_ID="${BACKBONE_TYPE}_${BACKBONE_SIZE:0:1}"
TRAIN_INFO="results/temporal/${BACKBONE_ID}_${MODEL_SUFFIX}_train.json"
EVAL_OUT="results/temporal/${BACKBONE_ID}_${MODEL_SUFFIX}_test.json"

echo "=========================================="
echo "Attentive probe: train + evaluate"
echo "=========================================="
echo "Backbone:      ${BACKBONE_TYPE} (${BACKBONE_SIZE})"
echo "Probe:         AttentiveClassifier (V-JEPA2 paper)"
echo "Window / FPS:  W=${WINDOW_SIZE} / target=${FPS} / source=${SOURCE_FPS}"
echo "Protocol:      ${PROTOCOL}"
echo "Padding mode:  ${PADDING_MODE}"
echo "Workers:       num_workers=${NUM_WORKERS}  feature_cache=${FEATURE_CACHE}"
echo "Epochs:        ${NUM_EPOCHS}  batch=${BATCH_SIZE}  lr=${LEARNING_RATE}  wd=${WEIGHT_DECAY}  seed=${SEED}"
echo "Suffix:        ${MODEL_SUFFIX}"
echo "CE weight:     ${CE_WEIGHT_STYLE}"
echo "Split file:    ${SPLIT_FILE:-<seeded game-disjoint split>}"
echo "Train info:    ${TRAIN_INFO}"
echo "Eval JSON:     ${EVAL_OUT}"
echo "=========================================="

# Build optional flag arrays. Empty SPLIT_FILE -> no --split-file flag emitted.
SPLIT_FLAGS=()
if [ -n "${SPLIT_FILE}" ]; then
    SPLIT_FLAGS=(--split-file "${SPLIT_FILE}")
fi

# --- 1. Train ---
echo
echo ">>> [1/2] Training ..."
uv run python -u src/train_temporal.py \
    --backbone-type ${BACKBONE_TYPE} \
    --backbone-size ${BACKBONE_SIZE} \
    --window-size ${WINDOW_SIZE} \
    --fps ${FPS} \
    --source-fps ${SOURCE_FPS} \
    --protocol ${PROTOCOL} \
    --replay-cap ${REPLAY_CAP} \
    --bg-count ${BG_COUNT} \
    --num-epochs ${NUM_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --patience 5 \
    --model-suffix ${MODEL_SUFFIX} \
    --seed ${SEED} \
    --feature-cache ${FEATURE_CACHE} \
    --num-workers ${NUM_WORKERS} \
    --padding-mode ${PADDING_MODE} \
    --ce-weight-style ${CE_WEIGHT_STYLE} \
    "${SPLIT_FLAGS[@]}" \
    --save-info ${TRAIN_INFO}

TRAIN_EXIT=$?
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed (exit ${TRAIN_EXIT}). Skipping eval."
    exit $TRAIN_EXIT
fi

# --- 2. Evaluate ---
echo
echo ">>> [2/2] Evaluating ..."
uv run python -u src/eval_temporal.py \
    --backbone-type ${BACKBONE_TYPE} \
    --backbone-size ${BACKBONE_SIZE} \
    --model-suffix ${MODEL_SUFFIX} \
    --window-size ${WINDOW_SIZE} \
    --fps ${FPS} \
    --protocol ${PROTOCOL} \
    --replay-cap ${REPLAY_CAP} \
    --bg-count ${BG_COUNT} \
    --metric all \
    --seed ${SEED} \
    --feature-cache ${FEATURE_CACHE} \
    --num-workers ${NUM_WORKERS} \
    --padding-mode ${PADDING_MODE} \
    "${SPLIT_FLAGS[@]}" \
    --save-json ${EVAL_OUT} \
    --profile-efficiency

echo
echo "=========================================="
echo "Run complete (protocol=${PROTOCOL})."
echo "  Checkpoint: best_attn_${BACKBONE_ID}_${MODEL_SUFFIX}.pth"
echo "  Train log:  ${TRAIN_INFO}"
echo "  Eval JSON:  ${EVAL_OUT}"
echo "=========================================="
