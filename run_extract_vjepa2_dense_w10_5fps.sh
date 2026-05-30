#!/bin/bash

# --- Slurm job parameters ---
#SBATCH --account=ec12
#SBATCH --job-name=extract_vjepa2_dense
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=00:30:00       # generous; per-task ~5 min at 4 FPS / W=8
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/extract/vjepa2_dense/%A_%a.out
#SBATCH --array=0-16          # 17 jobs × ~25 videos = 425 covered

# --- Common setup ---
source setup.sh

mkdir -p slurm_logs/extract/vjepa2_dense

# ─────────────────────────────────────────────────────────────────────────────
# V-JEPA2-Large dense (spatio-temporal token grid) extraction.
#
# Defaults: 4 FPS / window_size=8 (= 2-second context, matches the Kassab
# 2-second window plan but at coarser temporal density). Auto-stride at fps
# (no --stride passed) → 1 forward per output frame.
#
# Preprocessing matches DINOv3 exactly: shortest_edge=256 → center-crop
# 256×256 (configured in src/feature_extractors/vjepa2_extractor.py at the
# processor call).
#
# Token math at default (W=8):
#   tubelet 2×16×16  →  (8/2) * (16*16)  =  4 * 256  =  1,024 tokens / window
#   fp16 storage     →  1024 * 1024 * 2 bytes  =  ~2 MB / window
#   30-s video @ 4 FPS = 120 windows → ~240 MB / video
#   Full TACDEC (~425 videos) → ~100 GB total.
#
# Output filename includes window/stride tags so different runs don't collide:
#   {video_id}_vjepa2_l_4.0fps_dense_w8.npz   (auto-stride; no _s tag)
#   {video_id}_vjepa2_l_25.0fps_dense_w50_s1.npz   (if you re-run at W=50)
#
# Override examples (keep the hot path defaults light):
#   sbatch run_extract_vjepa2_dense_w50_25fps.sh                                  # 4 FPS / W=8 default
#   EXTRACT_FPS=25.0 WINDOW_SIZE=50 STRIDE=1 sbatch run_extract_vjepa2_dense_w50_25fps.sh
#       → fall back to the original W=50 stride-1 plan (~3 TB; only if needed).
#   OVERRIDE=1 sbatch run_extract_vjepa2_dense_w50_25fps.sh                       # re-extract
# ─────────────────────────────────────────────────────────────────────────────

export BACKBONE_TYPE="vjepa2"
export BACKBONE_SIZE="large"

VIDEOS_PER_JOB=${VIDEOS_PER_JOB:-25}
START_IDX=$((SLURM_ARRAY_TASK_ID * VIDEOS_PER_JOB))
END_IDX=$((START_IDX + VIDEOS_PER_JOB))

EXTRACT_FPS=${EXTRACT_FPS:-5.0}        # 25/5 = 5 exact integer stride (no drift)
WINDOW_SIZE=${WINDOW_SIZE:-10}         # 5 FPS * 2 s, even (tubelet=2 OK)
STRIDE=${STRIDE:-}                     # empty = auto (one window per output frame)
INTRA_WINDOW_STRIDE=${INTRA_WINDOW_STRIDE:-}   # empty = match anchor stride
                                       # (Claude Desktop spec: same source frames as DINOv3)

OVERRIDE=${OVERRIDE:-0}
OVERRIDE_ARG=""
if [ "$OVERRIDE" = "1" ]; then
    OVERRIDE_ARG="--override"
fi

# Padding mode: reflect (default for V-JEPA2 / the temporal pipeline -- border-
# reflected padding to square then resize to 256x256, no pixels cropped away) or
# center_crop. Reflect runs land in files tagged "_reflect" so they don't
# collide with centre-crop runs. Passed explicitly so it never falls back to the
# extract_features.py CLI default of center_crop.
PADDING_MODE=${PADDING_MODE:-reflect}
PAD_ARG="--padding-mode $PADDING_MODE"

STRIDE_ARG=""
if [ -n "$STRIDE" ]; then
    STRIDE_ARG="--stride $STRIDE"
fi

INTRA_ARG=""
if [ -n "$INTRA_WINDOW_STRIDE" ]; then
    INTRA_ARG="--intra-window-stride $INTRA_WINDOW_STRIDE"
fi

# Optional efficiency profiling (set PROFILE=1 to log GPU compute time /
# throughput / peak VRAM to extraction_throughput.csv). Off by default: the
# bulk run just caches features. NOTE: profile from a SINGLE, non-array task on
# a pinned GPU (e.g. --gpus=rtx30:1) — concurrent array tasks all append to the
# same CSV and would interleave rows.
PROFILE=${PROFILE:-0}
PROFILE_ARG=""
if [ "$PROFILE" = "1" ]; then
    PROFILE_ARG="--profile-efficiency"
fi

echo "=========================================="
echo "V-JEPA2-Large dense extraction (W=${WINDOW_SIZE}, fps=${EXTRACT_FPS})"
echo "=========================================="
echo "Job array task:  $SLURM_ARRAY_TASK_ID"
echo "Video range:     [$START_IDX:$END_IDX]"
echo "Backbone:        ${BACKBONE_TYPE} (${BACKBONE_SIZE})"
echo "FPS:             $EXTRACT_FPS"
echo "Window size:     $WINDOW_SIZE  (frames per V-JEPA2 forward)"
echo "Anchor stride:   ${STRIDE:-auto (= $(python -c "print(int(25/$EXTRACT_FPS))") frames @ 25 FPS source → 1 window per output frame)}"
echo "Intra-window:    ${INTRA_WINDOW_STRIDE:-default (= anchor stride)}  -> source span $(python -c "print(($WINDOW_SIZE - 1) * ${INTRA_WINDOW_STRIDE:-$(python -c "print(int(25/$EXTRACT_FPS))")})") frames"
if [ "$PADDING_MODE" = "reflect" ]; then
    echo "Preprocessing:   reflect-pad to square → resize 256x256 (matches DINOv3 reflect)"
else
    echo "Preprocessing:   shortest_edge=256 → center-crop 256x256 (matches DINOv3)"
fi
echo "Padding mode:    $PADDING_MODE"
echo "Feature type:    dense (fp16 spatio-temporal token grid)"
if [ "$OVERRIDE" = "1" ]; then
    echo "Override:        enabled"
fi
if [ "$PROFILE" = "1" ]; then
    echo "Profiling:       enabled (-> extraction_throughput.csv)"
fi
echo "=========================================="

uv run python extract_features.py \
    --model ${BACKBONE_TYPE} \
    --size ${BACKBONE_SIZE} \
    --output ${FOX_DATADIR_PATH}/TACDEC/features/${BACKBONE_TYPE}_${BACKBONE_SIZE} \
    --fps ${EXTRACT_FPS} \
    --batch-size 16 \
    --device cuda \
    --start-idx ${START_IDX} \
    --end-idx ${END_IDX} \
    --window-size ${WINDOW_SIZE} \
    ${STRIDE_ARG} \
    ${INTRA_ARG} \
    ${PAD_ARG} \
    ${PROFILE_ARG} \
    ${OVERRIDE_ARG}

echo "Done. Output dir:"
echo "  ${FOX_DATADIR_PATH}/TACDEC/features/${BACKBONE_TYPE}_${BACKBONE_SIZE}/*_dense_w${WINDOW_SIZE}*.npz"
