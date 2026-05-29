#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=eval_spatial
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/eval/spatial/%j.out

# --- Common setup ---
source setup.sh

export BACKBONE_TYPE="dinov3"
export BACKBONE_SIZE="large"

# --- Required: run directory ---
# Usage:
#   sbatch run_eval_spatial.sh results/dinov3_linear_spatial/run_<timestamp>
#   sbatch run_eval_spatial.sh results/dinov3_linear_spatial/run_<timestamp> tight
RUN_DIR=${1:?usage: sbatch run_eval_spatial.sh <run-dir> [metric]}
METRIC=${2:-tight}

mkdir -p slurm_logs/eval/spatial

echo "=========================================="
echo "DINOv3 linear probe -- evaluation"
echo "=========================================="
echo "Run dir: ${RUN_DIR}"
echo "Metric:  ${METRIC}"
echo "=========================================="

# Eval always profiles the head (params / latency / peak VRAM ->
# results/head_efficiency.csv) when CUDA is present. GPU is pinned to rtx30
# above for comparability.
uv run python src/eval_spatial.py \
    --run-dir "${RUN_DIR}" \
    --metric  "${METRIC}"

echo "=========================================="
echo "Evaluation complete."
echo "=========================================="
