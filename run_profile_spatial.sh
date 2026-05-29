#!/bin/bash
# --- Head-only inference-cost profiler: DINOv3 linear probe ---
# Runs the existing spatial eval (cheap; head consumes cached CLS features) and
# then appends one row to results/head_efficiency.csv with trainable params,
# mean head latency (batch=16), and peak head VRAM. Pinned to RTX 3090 so the
# three pipelines in the head-efficiency table are measured on identical hardware.
#SBATCH --account=ec12
#SBATCH --job-name=profile_spatial
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/profile/spatial_%j.out

source setup.sh

mkdir -p slurm_logs/profile results

# Usage:
#   sbatch run_profile_spatial.sh results/dinov3_linear_spatial/run_<timestamp>
#   sbatch run_profile_spatial.sh results/dinov3_linear_spatial/run_<timestamp> tight
RUN_DIR=${1:?usage: sbatch run_profile_spatial.sh <run-dir> [metric]}
METRIC=${2:-tight}

echo "=========================================="
echo "DINOv3 linear probe -- head efficiency"
echo "=========================================="
echo "Run dir: ${RUN_DIR}"
echo "Metric:  ${METRIC}"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

uv run python -u src/eval_spatial.py \
    --run-dir "${RUN_DIR}" \
    --metric  "${METRIC}" \
    --profile-efficiency

echo "=========================================="
echo "Profiling complete. Row appended to results/head_efficiency.csv"
echo "=========================================="
