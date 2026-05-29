#!/bin/bash
#SBATCH --account=ec12
#SBATCH --job-name=train_eval_spatial_kfold
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/train_eval/spatial_kfold/%j.out

# Train + evaluate the DINOv3 linear spatial probe under k-fold cross-validation.
# Loops over fold_idx in {0..N_FOLDS-1}, writing each fold to <BASE>/fold_<i>/,
# then runs the aggregator to produce <BASE>/aggregate.json.
#
# Usage:
#   sbatch run_train_eval_spatial_kfold.sh                              # defaults
#   sbatch run_train_eval_spatial_kfold.sh 42 100 0.005 tight 5 0.0     # seed epochs lr metric n_folds wd

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
N_FOLDS=${5:-5}
WEIGHT_DECAY=${6:-0}

BASE_NAME="kfold_${SLURM_JOB_ID}"
BASE_DIR="${FOX_DATADIR_PATH}/TACDEC/results/dinov3_linear_spatial/${BASE_NAME}"

mkdir -p slurm_logs/train_eval/spatial_kfold
mkdir -p "${BASE_DIR}"

echo "=========================================="
echo "DINOv3 linear probe -- k-fold train + eval"
echo "=========================================="
echo "Base name:   ${BASE_NAME}"
echo "Base dir:    ${BASE_DIR}"
echo "Seeds:       ${SEED}  (split / balance / train, held constant across folds)"
echo "Epochs:      ${EPOCHS}"
echo "LR:          ${LR}"
echo "Weight decay:${WEIGHT_DECAY}"
echo "Event metric:${METRIC}"
echo "n_folds:     ${N_FOLDS}"
echo "=========================================="

for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    FOLD_DIR="${BASE_DIR}/fold_${FOLD}"
    echo ""
    echo "=========================================="
    echo "Fold $((FOLD + 1))/${N_FOLDS}   ->   ${FOLD_DIR}"
    echo "=========================================="

    echo ""
    echo "[stage 1/2] Training (fold ${FOLD})"
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
        --n-folds     ${N_FOLDS} \
        --fold-idx    ${FOLD} \
        --output-dir  "${FOLD_DIR}"

    echo ""
    echo "[stage 2/2] Evaluation (fold ${FOLD})"
    echo "------------------------------------------"
    uv run python src/eval_spatial.py \
        --run-dir "${FOLD_DIR}" \
        --metric  "${METRIC}"
done

echo ""
echo "=========================================="
echo "Aggregating ${N_FOLDS} folds"
echo "=========================================="
uv run python src/aggregate_kfold_spatial.py --base-dir "${BASE_DIR}"

echo ""
echo "=========================================="
echo "K-fold pipeline complete."
echo "Base dir: ${BASE_DIR}"
echo "Aggregate: ${BASE_DIR}/aggregate.json"
echo "=========================================="
