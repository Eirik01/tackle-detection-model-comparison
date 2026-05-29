#!/bin/bash

# --- Slurm job parameters ---
#SBATCH --account=ec12
#SBATCH --job-name=dump_dinov3_kassab
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/kassab/dump_dinov3_%j.out

# --- Common setup ---
source setup.sh

mkdir -p slurm_logs/kassab

# ─────────────────────────────────────────────────────────────────────────────
# Build a Kassab-format stacked feature tensor from DINOv3 CLS features.
#
# Output (Fox work area, subject to cleanup policy):
#   /cluster/work/projects/ec12/ec-eirikto/TACDEC/dinov3_kassab_format/
#     ├── dinov3_sorted_cls_tokens_features.pt   (~1.1 GB float32 [N, 1024])
#     ├── dinov3_frame_counts.npy                (425 ints)
#     └── dinov3_video_ids.txt                   (425 lines)
#
# After this finishes, scp the .pt down to your local machine:
#   scp ec-eirikto@fox.educloud.no:/cluster/work/projects/ec12/ec-eirikto/TACDEC/dinov3_kassab_format/dinov3_sorted_cls_tokens_features.pt \
#       tacdec-kassab-implementation/
#
# Then in spatial-approach.ipynb cell 3, change:
#   X = torch.load('./sorted_cls_tokens_features.pt', map_location=device)
# to:
#   X = torch.load('./dinov3_sorted_cls_tokens_features.pt', map_location=device)
#
# Re-run cells 1..43 (training) + cell 45 (meta-test eval). The 600-sample
# evaluation is now byte-identical to Kassab's published spatial baseline,
# only the backbone differs — a clean A/B comparison.
#
# Usage:
#   sbatch run_dump_dinov3_kassab_format.sh                      # center_crop, 25.0 fps
#   sbatch run_dump_dinov3_kassab_format.sh 25.0                 # explicit fps
#   sbatch run_dump_dinov3_kassab_format.sh 25.0 reflect         # reflect variant
#
# To produce BOTH variants, sbatch the script twice:
#   sbatch run_dump_dinov3_kassab_format.sh 25.0 center_crop
#   sbatch run_dump_dinov3_kassab_format.sh 25.0 reflect
#
# Outputs (named by padding mode, no collision):
#   dinov3_sorted_cls_tokens_features.pt           (center_crop, default)
#   dinov3_reflect_sorted_cls_tokens_features.pt   (reflect)
# ─────────────────────────────────────────────────────────────────────────────

FPS=${1:-25.0}
PADDING_MODE=${2:-center_crop}

echo "=========================================="
echo "Dump DINOv3 features (Kassab format)"
echo "=========================================="
echo "FPS:           ${FPS}"
echo "Padding mode:  ${PADDING_MODE}"
echo "=========================================="

uv run python dump_dinov3_kassab_format.py --fps ${FPS} --padding-mode ${PADDING_MODE}

echo "=========================================="
echo "Done."
echo "=========================================="
