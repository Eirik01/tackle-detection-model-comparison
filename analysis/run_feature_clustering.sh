#!/bin/bash

# --- Slurm job parameters ---
#SBATCH --account=ec12
#SBATCH --job-name=feat_clustering
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/clustering/%j.out

# --- Common setup ---
source setup.sh
mkdir -p slurm_logs/clustering

# ─────────────────────────────────────────────────────────────────────────────
# Backbone feature clustering on the TACDEC test split.
#
#   (1) DINOv3 CLS @ 25 FPS  -- the linear probe's input
#   (2) DINOv3 + V-JEPA 2 dense (mean-pooled per W=10 window) -- attentive
#       probe inputs, apples-to-apples
#
# Outputs:
#   figures/feature_clustering_dinov3_cls.pdf
#   figures/feature_clustering_dense.pdf
#   figures/feature_clustering_dense_metrics.json
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "Feature clustering on test split"
echo "Start: $(date)"
echo "=========================================="

echo "----- (1) DINOv3 CLS -----"
uv run python visualization/feature_clustering_figure.py

echo "----- (2) Dense (attentive-probe inputs) -----"
uv run python visualization/feature_clustering_dense.py

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
