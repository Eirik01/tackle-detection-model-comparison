#!/bin/bash
# Generate attention-heatmap contact sheets + mp4s for the qualitative figure
# block in Chapter 7 (Section 7.6: "Attention in temporal approaches").
#
# For each anchor we run analysis/attn_window.py twice (DINOv3 and V-JEPA 2)
# so the two attentive probes can be compared on the same window. Outputs land
# in results/attn_windows/{dinov3_l,vjepa2_l}/.
#
# The script only runs the probe head on pre-extracted dense features (no
# backbone forward), so the resource footprint is small.
#SBATCH --account=ec12
#SBATCH --job-name=attn_windows
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/attn_windows_%j.out

source setup.sh

mkdir -p slurm_logs results/attn_windows

MODEL_SUFFIX=${MODEL_SUFFIX:-centered_v1}

# (clip_id, anchor_idx) pairs. anchor_idx is on the 5 FPS grid (matches the
# f<NN> suffix in figures/qualitative/ and the CSV frame_idx column).
ANCHORS=(
  "3266_abvbketxnf6d8 47"    # Mode 1: shared FP, 1.7 s past real tackle
  "3357_apoq8yrw4fe25 106"   # Mode 4: replay->live, attentive-only failure
  "3198_amju0ph4fpfqa 110"   # V-JEPA 2-only replay->live
  "3248_arl905135myb5 122"   # V-JEPA 2-only replay->live
  "3271_ahjx9rzchewnj 122"   # V-JEPA 2-only replay->live
  "3386_a47m0uqqg8f5e 115"   # V-JEPA 2-only replay->live
  "3393_ab6hyp2jdvmjj 85"    # POS anchor: clean tackle-live, all 3 correct
  "3389_ammj6l4t44a04 88"    # POS anchor: clean tackle-replay, all 3 correct
)

for entry in "${ANCHORS[@]}"; do
  for BACKBONE in dinov3 vjepa2; do
    echo
    echo "=========================================="
    echo "anchor: ${entry}   backbone: ${BACKBONE}"
    echo "=========================================="
    uv run python -u -m analysis.attn_window ${entry} ${MODEL_SUFFIX} \
        --backbone-type ${BACKBONE} \
        --backbone-size large \
        --padding-mode reflect \
        --view reflect \
        --no-open
  done
done

echo
echo "=========================================="
echo "Done. Outputs under results/attn_windows/{dinov3_l,vjepa2_l}/"
echo "=========================================="
