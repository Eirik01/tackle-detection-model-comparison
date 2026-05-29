#!/bin/bash
# Run the trained DINOv3 linear spatial probe over the extracted SoccerNet half
# and dump fired tackle events with timestamps for manual verification.
#
# Submit AFTER run_extract.sh (the CLS-feature extractor):
#   RUN_DIR=/cluster/work/projects/ec12/ec-eirikto/TACDEC/results/dinov3_linear_spatial/<RUN_NAME> \
#       sbatch soccernet_experiment/run_predict_spatial.sh
#
# RUN_DIR is the spatial training run directory and must contain
# model.pt + config.json (the standard eval_spatial.py inputs).
#
# --- Slurm job parameters ---
#SBATCH --account=ec12
#SBATCH --job-name=sn_predict_spatial
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/soccernet/predict_spatial_%j.out

source setup.sh
mkdir -p slurm_logs/soccernet

# Mirrors SOCCERNET_EXPERIMENT_DIR in src/config.py — keep in sync.
EXP_DIR=${EXP_DIR:-/cluster/work/projects/ec12/ec-eirikto/soccernet_thesis_experiment}
RUN_DIR=${RUN_DIR:?Set RUN_DIR=/path/to/results/dinov3_linear_spatial/<RUN_NAME>}
MIN_CONFIDENCE=${MIN_CONFIDENCE:-0.5}
MIN_SEGMENT_FRAMES=${MIN_SEGMENT_FRAMES:-2}

FEATURES_DIR="${EXP_DIR}/features"
OUT_DIR="${EXP_DIR}/predictions"

echo "=========================================="
echo "SoccerNet event prediction (DINOv3 linear spatial)"
echo "  run-dir  : ${RUN_DIR}"
echo "  features : ${FEATURES_DIR}  (video_id auto-detected)"
echo "  min_conf : ${MIN_CONFIDENCE}   min_seg: ${MIN_SEGMENT_FRAMES}"
echo "  out      : ${OUT_DIR}"
echo "=========================================="

uv run python -u src/predict_soccernet_spatial.py \
    --run-dir "${RUN_DIR}" \
    --features-dir "${FEATURES_DIR}" \
    --min-confidence "${MIN_CONFIDENCE}" \
    --min-segment-frames "${MIN_SEGMENT_FRAMES}" \
    --out-dir "${OUT_DIR}"

echo "Done. Look in ${OUT_DIR}/ for *_dinov3_spatial_events.txt"
