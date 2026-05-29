#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=train_eval_spatial
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/train_eval/spatial/%j.out

# Train the DINOv3 linear spatial probe, then immediately evaluate it.
# Uses the SLURM job ID as the run name so the slurm log and the run directory
# share the same identifier.
#
# Usage:
#   sbatch run_train_eval_spatial.sh                          # defaults
#   sbatch run_train_eval_spatial.sh 42 100 1e-3 tight 0.0    # seed epochs lr metric wd

# --- Common setup ---
source setup.sh

export BACKBONE_TYPE="dinov3"
export BACKBONE_SIZE="large"

# --- Optional CLI overrides ---
SEED=${1:-42}
EPOCHS=${2:-50}
# Selected from the spatial LR/WD sweep (selection.json under
# TACDEC/results/sweeps/dinov3_linear_spatial/seed42/) under the decoupled
# stop/save protocol (checkpoint on max val macro-F1, early-stop on val_loss
# with patience 5). Cell (2e-4, 0) has the highest plateau macro-F1 (0.821)
# and a healthy train/val loss gap (0.120); peak val macro-F1 = 0.822.
# At this grid weight decay had no measurable effect, so wd=0 was kept.
LR=${3:-2e-4}
METRIC=${4:-tight}
WEIGHT_DECAY=${5:-0}

# Single source of truth for the run directory (mirrors config.TACDEC_RESULTS).
RUN_NAME="run_${SLURM_JOB_ID}"
RUN_DIR="/cluster/work/projects/ec12/ec-eirikto/TACDEC/results/dinov3_linear_spatial/${RUN_NAME}"

mkdir -p slurm_logs/train_eval/spatial

echo "=========================================="
echo "DINOv3 linear probe -- train + eval"
echo "=========================================="
echo "Run name:    ${RUN_NAME}"
echo "Run dir:     ${RUN_DIR}"
echo "Seeds:       ${SEED}  (split / balance / train)"
echo "Epochs:      ${EPOCHS}"
echo "LR:          ${LR}"
echo "Weight decay:${WEIGHT_DECAY}"
echo "Event metric:${METRIC}"
echo "=========================================="

echo ""
echo "[stage 1/2] Training"
echo "------------------------------------------"
uv run python src/train_spatial.py \
    --seed-split  ${SEED} \
    --seed-balance ${SEED} \
    --seed-train  ${SEED} \
    --epochs      ${EPOCHS} \
    --lr          ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --patience    5 \
    --batch-size  256 \
    --output-dir  "${RUN_DIR}"

echo ""
echo "[stage 2/2] Evaluation"
echo "------------------------------------------"
# Eval always profiles the head (params / latency / peak VRAM ->
# results/head_efficiency.csv) when CUDA is present. GPU is pinned to rtx30
# above so the numbers are comparable across pipelines.
uv run python src/eval_spatial.py \
    --run-dir "${RUN_DIR}" \
    --metric  "${METRIC}"

echo ""
echo "=========================================="
echo "Pipeline complete."
echo "Run artifacts: ${RUN_DIR}"
echo "=========================================="
