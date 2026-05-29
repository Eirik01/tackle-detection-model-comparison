#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DEPRECATED. Superseded by per-backbone wrappers:
#
#   sbatch run_train_eval_temporal_dinov3.sh   # 45 min / 16 G / 4 CPUs
#   sbatch run_train_eval_temporal_vjepa2.sh   # 2 h    / 24 G / 8 CPUs
#
# Both source _train_eval_temporal_body.sh, which carries the shared train+eval
# pipeline. The single-script version oversized DINOv3 (asked 6 h / 32 G when
# 20 min / 10 G was used), hurting queue position. Kept only as a pointer for
# old aliases / scripts that still call this filename.
# ─────────────────────────────────────────────────────────────────────────────

echo "run_train_eval_temporal.sh is deprecated."
echo "Use one of:"
echo "  sbatch run_train_eval_temporal_dinov3.sh"
echo "  sbatch run_train_eval_temporal_vjepa2.sh"
exit 2
