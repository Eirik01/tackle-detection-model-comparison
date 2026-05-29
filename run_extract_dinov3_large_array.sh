#!/bin/bash

# --- Slurm job parameters ---
#SBATCH --account=ec12
#SBATCH --job-name=extract_dinov3_large
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=0:40:00
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/extract/dinov3_large/extraction_%A_%a.out
#SBATCH --array=0-9  # 5 jobs for 425 videos (100 per job)

# --- Common setup ---
source setup.sh

# --- Set backbone configuration ---
export BACKBONE_TYPE="dinov3"
export BACKBONE_SIZE="large"

# --- Calculate video range for this task (100 videos per job) ---
VIDEOS_PER_JOB=50
START_IDX=$((SLURM_ARRAY_TASK_ID * VIDEOS_PER_JOB))
END_IDX=$((START_IDX + VIDEOS_PER_JOB))

# Extract FPS - can specify multiple fps values for multi-resolution experiments
EXTRACT_FPS=${EXTRACT_FPS:-25.0}

# Optional override (set OVERRIDE=1 to force re-extraction)
OVERRIDE=${OVERRIDE:-0}
OVERRIDE_ARG=""
if [ "$OVERRIDE" = "1" ]; then
    OVERRIDE_ARG="--override"
fi

# Padding mode: center_crop (default, current behaviour) or reflect (border-
# reflected padding to square then resize to 256x256). Reflect runs land in
# files tagged "_reflect" so they don't collide with centre-crop runs.
PADDING_MODE=${PADDING_MODE:-center_crop}
PAD_ARG=""
if [ "$PADDING_MODE" = "reflect" ]; then
    PAD_ARG="--padding-mode reflect"
fi

# --- Run feature extraction ---
echo "=========================================="
echo "Running DINOv3 Large Feature Extraction"
echo "=========================================="
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing videos: [$START_IDX:$END_IDX]"
echo "Backbone: ${BACKBONE_TYPE}"
echo "Size: ${BACKBONE_SIZE}"
echo "Extract FPS: $EXTRACT_FPS"
echo "Padding mode: $PADDING_MODE"
if [ "$OVERRIDE" = "1" ]; then
    echo "Override: enabled (existing features will be overwritten)"
fi
echo "========================================="

echo ""
echo "Extracting at $EXTRACT_FPS FPS..."
uv run python extract_features.py \
    --model dinov3 \
    --size large \
    --output /cluster/work/projects/ec12/ec-eirikto/TACDEC/features/${BACKBONE_TYPE}_${BACKBONE_SIZE} \
    --fps $EXTRACT_FPS \
    --batch-size 16 \
    --device cuda \
    --save-dense \
    --start-idx $START_IDX \
    --end-idx $END_IDX \
    ${PAD_ARG} \
    ${OVERRIDE_ARG}

echo "Feature extraction completed for videos [$START_IDX:$END_IDX]!"
echo "Results saved to: /cluster/work/projects/ec12/ec-eirikto/TACDEC/features/${BACKBONE_TYPE}_${BACKBONE_SIZE}/"
