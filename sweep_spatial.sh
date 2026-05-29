#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=sweep_spatial
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --array=0-29
#SBATCH --output=slurm_logs/sweep/spatial/%A_%a.out
# LR/WD grid sweep for the DINOv3 linear spatial probe.
#
# Grid (DINOv3 paper, linear-probe protocol): 15 LR x 2 WD = 30 combos.
# Each array task trains ONE combo on the fixed train/val split and writes its
# run dir (config.json + metrics.json, incl. best_val_macro_f1). No test eval
# here -- selection happens on val, test is touched once afterwards.
#
# Usage:
#   sbatch sweep_spatial.sh                # defaults (seed 42, 50 epochs)
#   sbatch sweep_spatial.sh 42 50          # seed epochs
#
# After all 30 tasks finish:
#   uv run python -m src.select_hparams \
#       --sweep-dir <SWEEP_ROOT>/dinov3_linear_spatial/seed42 --pipeline spatial

# --- Common setup ---
source setup.sh

export BACKBONE_TYPE="dinov3"
export BACKBONE_SIZE="large"

SEED=${1:-42}
EPOCHS=${2:-50}

# DINOv3 linear-probe grid (SGD, momentum 0.9).
LRS=(1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1 2e-1 5e-1 1e0 2e0 5e0)
WDS=(0 1e-5)
N_WD=${#WDS[@]}

IDX=${SLURM_ARRAY_TASK_ID}
LR=${LRS[$((IDX / N_WD))]}
WD=${WDS[$((IDX % N_WD))]}

SWEEP_ROOT="/cluster/work/projects/ec12/ec-eirikto/TACDEC/results/sweeps"
RUN_DIR="${SWEEP_ROOT}/dinov3_linear_spatial/seed${SEED}/lr${LR}_wd${WD}"

mkdir -p slurm_logs/sweep/spatial

echo "=========================================="
echo "DINOv3 linear probe -- sweep task ${IDX}"
echo "=========================================="
echo "LR / WD:     ${LR} / ${WD}"
echo "Seed:        ${SEED}  (split / balance / train)"
echo "Epochs:      ${EPOCHS}"
echo "Run dir:     ${RUN_DIR}"
echo "=========================================="

uv run python src/train_spatial.py \
    --seed-split   ${SEED} \
    --seed-balance ${SEED} \
    --seed-train   ${SEED} \
    --epochs       ${EPOCHS} \
    --lr           ${LR} \
    --weight-decay ${WD} \
    --patience     5 \
    --batch-size   256 \
    --output-dir   "${RUN_DIR}"

echo ""
echo "Sweep task ${IDX} complete (lr=${LR} wd=${WD}). Run dir: ${RUN_DIR}"
