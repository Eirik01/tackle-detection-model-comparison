#!/bin/bash
# Run the best V-JEPA 2 attentive probe over the extracted SoccerNet half and
# dump fired tackle events with timestamps for manual verification.
#
# Submit AFTER MODE=vjepa2 sbatch soccernet_experiment/run_extract.sh:
#   sbatch soccernet_experiment/run_predict_vjepa2.sh
#   MODEL_SUFFIX=other_run sbatch soccernet_experiment/run_predict_vjepa2.sh
#
# MODEL_SUFFIX selects the checkpoint best_attn_vjepa2_l_<MODEL_SUFFIX>.pth from
# TACDEC_MODELS/vjepa2_l/. Defaults to centred_v1 (the live single-fold suffix
# used by run_train_eval_temporal_vjepa2.sh and run_predict.sh).
#
# --- Slurm job parameters ---
#SBATCH --account=ec12
#SBATCH --job-name=sn_predict_vjepa2
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/soccernet/predict_vjepa2_%j.out

source setup.sh
mkdir -p slurm_logs/soccernet

# Mirrors SOCCERNET_EXPERIMENT_DIR in src/config.py — keep in sync.
EXP_DIR=${EXP_DIR:-/cluster/work/projects/ec12/ec-eirikto/soccernet_thesis_experiment}
MODEL_SUFFIX=${MODEL_SUFFIX:-centred_v1}
MIN_CONFIDENCE=${MIN_CONFIDENCE:-0.5}
SIGMA=${SIGMA:-1.0}
MIN_DISTANCE_SEC=${MIN_DISTANCE_SEC:-0.5}
# Must match extraction. V-JEPA 2 attentive probe protocol: W=10 raw frames
# @ 5 FPS = 2 s window (Kassab-style).
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
echo "SoccerNet event prediction (V-JEPA 2 attentive)"
echo "  features : ${FEATURES_DIR}  (video_id auto-detected)"
echo "  suffix   : ${MODEL_SUFFIX}   min_conf: ${MIN_CONFIDENCE}"
echo "  W / FPS  : ${WINDOW_SIZE} / ${FPS}"
echo "  out      : ${OUT_DIR}"
echo "=========================================="

uv run python -u src/predict_soccernet_vjepa2.py \
    --features-dir "${FEATURES_DIR}" \
    --model-suffix "${MODEL_SUFFIX}" \
    --backbone-size large \
    --window-size "${WINDOW_SIZE}" \
    --fps "${FPS}" \
    --padding-mode reflect \
    --min-confidence "${MIN_CONFIDENCE}" \
    --sigma "${SIGMA}" \
    --min-distance-sec "${MIN_DISTANCE_SEC}" \
    "${MAX_DURATION_ARG[@]}" \
    --out-dir "${OUT_DIR}"

echo "Done. Look in ${OUT_DIR}/ for *_vjepa2_temporal_events.txt"
