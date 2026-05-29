#!/bin/bash
# Run the best DINOv3 attentive probe over the extracted SoccerNet half and dump
# fired tackle events with timestamps for manual verification.
#
# Submit AFTER run_extract.sh:
#   MODEL_SUFFIX=centred_v1 sbatch soccernet_experiment/run_predict.sh
#
# MODEL_SUFFIX selects the checkpoint best_attn_dinov3_l_<MODEL_SUFFIX>.pth from
# TACDEC_MODELS/dinov3_l/ -- set it to YOUR best run's suffix.
#
# --- Slurm job parameters ---
#SBATCH --account=ec12
#SBATCH --job-name=sn_predict_dinov3
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/soccernet/predict_%j.out

source setup.sh
mkdir -p slurm_logs/soccernet

# Mirrors SOCCERNET_EXPERIMENT_DIR in src/config.py — keep in sync.
EXP_DIR=${EXP_DIR:-/cluster/work/projects/ec12/ec-eirikto/soccernet_thesis_experiment}
MODEL_SUFFIX=${MODEL_SUFFIX:-centred_v1}
MIN_CONFIDENCE=${MIN_CONFIDENCE:-0.5}
SIGMA=${SIGMA:-1.0}
MIN_DISTANCE_SEC=${MIN_DISTANCE_SEC:-0.5}
# Must match extraction: source FPS on disk, target/window of the probe.
# We now extract directly at 5 FPS, so source == target and src_stride = 1.
SOURCE_FPS=${SOURCE_FPS:-5.0}
FPS=${FPS:-5.0}
WINDOW_SIZE=${WINDOW_SIZE:-10}
# Set MAX_DURATION_SEC to score only the first N seconds (e.g. 600 for a
# quick 10-min qualitative pass). Unset = score the whole clip.
MAX_DURATION_SEC=${MAX_DURATION_SEC:-}

FEATURES_DIR="${EXP_DIR}/features"
OUT_DIR="${EXP_DIR}/predictions"

MAX_DURATION_ARG=()
if [ -n "${MAX_DURATION_SEC}" ]; then
    MAX_DURATION_ARG=(--max-duration-sec "${MAX_DURATION_SEC}")
fi

echo "=========================================="
echo "SoccerNet event prediction (DINOv3 attentive)"
echo "  features : ${FEATURES_DIR}  (video_id auto-detected)"
echo "  suffix   : ${MODEL_SUFFIX}   min_conf: ${MIN_CONFIDENCE}"
echo "  out      : ${OUT_DIR}"
echo "=========================================="

# --video-id is omitted on purpose: predict_soccernet.py picks up the single
# dense .npy in --features-dir and derives the id from its filename.
uv run python -u src/predict_soccernet.py \
    --features-dir "${FEATURES_DIR}" \
    --model-suffix "${MODEL_SUFFIX}" \
    --backbone-size large \
    --window-size "${WINDOW_SIZE}" \
    --fps "${FPS}" \
    --source-fps "${SOURCE_FPS}" \
    --padding-mode reflect \
    --min-confidence "${MIN_CONFIDENCE}" \
    --sigma "${SIGMA}" \
    --min-distance-sec "${MIN_DISTANCE_SEC}" \
    "${MAX_DURATION_ARG[@]}" \
    --out-dir "${OUT_DIR}"

echo "Done. Look in ${OUT_DIR}/ for *_dinov3_temporal_events.txt"
