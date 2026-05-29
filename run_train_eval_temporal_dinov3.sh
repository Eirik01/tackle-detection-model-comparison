#!/bin/bash
# --- Slurm job parameters (DINOv3 sizing) ---
# DINOv3 dense features are mmap'd .npy, so I/O is effectively free; the run is
# compute-bound on the ~55 M-param probe over [W*256, 1024] = [2560, 1024]
# tokens. Past runs: ~20 min wall, ~10 GiB peak, ~1.4 CPU cores effective.
#SBATCH --account=ec12
#SBATCH --job-name=train_eval_temporal_dinov3
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/train_eval/temporal/dinov3_%j.out

# --- Common setup ---
source setup.sh

mkdir -p slurm_logs/train_eval/temporal results/temporal

# Usage:
#   sbatch run_train_eval_temporal_dinov3.sh                          # defaults (centered)
#   sbatch run_train_eval_temporal_dinov3.sh centered_v2 50 64        # custom suffix / epochs / batch
#
# Kassab TempTAC parity, STRICT concat-and-slide (cross-clip windows, DINOv3-only):
#   PROTOCOL=kassab_concat \
#   CE_WEIGHT_STYLE=balanced \
#   SPLIT_FILE=data/kassab_split.json \
#   sbatch run_train_eval_temporal_dinov3.sh kassab_concat_v1
#
# Strict TempTAC eval-pool parity (replicates Kassab's extract_data bug so the
# test pool positionally matches his reported classification report). Default
# split_mode is kassab_bug, so the line below is equivalent to omitting it:
#   PROTOCOL=kassab_concat \
#   CE_WEIGHT_STYLE=balanced \
#   SPLIT_FILE=data/kassab_split.json \
#   SPLIT_MODE=kassab_bug \
#   sbatch run_train_eval_temporal_dinov3.sh kassab_concat_bug_v1

BACKBONE_TYPE=dinov3
MODEL_SUFFIX=${1:-centered_v1}
NUM_EPOCHS=${2:-30}
BATCH_SIZE=${3:-64}
# Selected hyperparameters from sweeps/dinov3_l/seed42/selection.json under
# the decoupled stop/save protocol (checkpoint on max val macro-F1, early-stop
# on val_loss with patience 5). Cell (1e-4, 1e-2) wins on peak macro-F1
# (0.8812), plateau macro-F1 (0.868), and has a healthy train/val loss gap.
LEARNING_RATE=${4:-1e-4}
SEED=${5:-42}
WEIGHT_DECAY=${6:-1e-2}
PROTOCOL=${PROTOCOL:-centered}
# 'kassab_concat' protocol caps (ignored by 'centered'). Kassab Table 7.6 defaults.
REPLAY_CAP=${REPLAY_CAP:-280}
BG_COUNT=${BG_COUNT:-500}
FEATURE_CACHE=${FEATURE_CACHE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
# Padding mode used at extraction time. Default keeps the centre-crop files;
# set PADDING_MODE=reflect to load the *_reflect_dense_* files instead.
# set PADDING_MODE=center_crop to explicitly load the centre-crop files
PADDING_MODE=${PADDING_MODE:-reflect}
# Kassab parity knobs (no-ops at their defaults):
#   CE_WEIGHT_STYLE=balanced   uses sklearn 'balanced' inverse-frequency CE weights.
#   SPLIT_FILE=<path.json>     overrides the seeded game-disjoint split with a
#                              fixed clip-ID partition (see dump_kassab_split.py).
CE_WEIGHT_STYLE=${CE_WEIGHT_STYLE:-min1}
SPLIT_FILE=${SPLIT_FILE:-}
# kassab_concat protocol only:
#   SPLIT_MODE=kassab_bug   (default) -- replicates Kassab's extract_data bug so
#                                         the train/val/test pools positionally
#                                         match his reported eval pool.
#   SPLIT_MODE=correct      -- real game-disjoint partition.
SPLIT_MODE=${SPLIT_MODE:-kassab_bug}

WINDOW_SIZE=10                # 5 FPS * 2 s
FPS=5.0
SOURCE_FPS=25.0               # DINOv3 dense files are on-disk at 25 FPS
BACKBONE_SIZE="large"

BACKBONE_ID="${BACKBONE_TYPE}_${BACKBONE_SIZE:0:1}"
TRAIN_INFO="results/temporal/${BACKBONE_ID}_${MODEL_SUFFIX}_train.json"
EVAL_OUT="results/temporal/${BACKBONE_ID}_${MODEL_SUFFIX}_test.json"

echo "=========================================="
echo "Attentive probe: train + evaluate"
echo "=========================================="
echo "Backbone:      ${BACKBONE_TYPE} (${BACKBONE_SIZE})"
echo "Probe:         DINOv3AttentiveProbe (paper RoPE)"
echo "Window / FPS:  W=${WINDOW_SIZE} / target=${FPS} / source=${SOURCE_FPS}"
echo "Protocol:      ${PROTOCOL}"
echo "Padding mode:  ${PADDING_MODE}"
echo "Workers:       num_workers=${NUM_WORKERS}  feature_cache=${FEATURE_CACHE}"
echo "Epochs:        ${NUM_EPOCHS}  batch=${BATCH_SIZE}  lr=${LEARNING_RATE}  wd=${WEIGHT_DECAY}  seed=${SEED}"
echo "Suffix:        ${MODEL_SUFFIX}"
echo "CE weight:     ${CE_WEIGHT_STYLE}"
echo "Split file:    ${SPLIT_FILE:-<seeded game-disjoint split>}"
echo "Split mode:    ${SPLIT_MODE}  (only relevant for protocol=kassab_concat)"
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
    --split-mode ${SPLIT_MODE} \
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
    --split-mode ${SPLIT_MODE} \
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
