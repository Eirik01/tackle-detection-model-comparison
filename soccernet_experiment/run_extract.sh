#!/bin/bash
# Extract features for the downloaded SoccerNet half. Three modes,
# selected by the MODE env var:
#
#   MODE=dense (default)
#     - DINOv3-L, reflect padding -> 256x256, fp16, dense patch tokens only.
#     - Matches the DINOv3 attentive probe's TACDEC training preprocessing.
#     - Writes: <stem>_dinov3_l_<fps>fps_reflect_dense_features.npy
#
#   MODE=cls
#     - DINOv3-L, centre-crop padding -> 256x256, CLS only.
#     - Matches the linear (spatial) probe's TACDEC training preprocessing.
#     - Writes: <stem>_dinov3_l_<fps>fps_features.npz       (key='cls')
#
#   MODE=vjepa2
#     - V-JEPA 2-L, reflect padding -> 256x256, fp16, spatio-temporal token
#       grid per W=10 window.
#     - Matches the V-JEPA 2 attentive probe's TACDEC training preprocessing
#       (paper-faithful protocol: W=10 raw frames @ 5 FPS = 2 s).
#     - Writes: <stem>_vjepa2_l_<fps>fps_reflect_dense_w10.npz
#
# All three modes extract at 5 FPS directly (== probe target FPS).
#
# Submit AFTER the download step has populated $EXP_DIR:
#   sbatch soccernet_experiment/run_extract.sh                  # MODE=dense
#   MODE=cls sbatch soccernet_experiment/run_extract.sh         # MODE=cls
#   MODE=vjepa2 sbatch soccernet_experiment/run_extract.sh      # MODE=vjepa2
#
# --- Slurm job parameters ---
#SBATCH --account=ec12
#SBATCH --job-name=sn_extract_dinov3
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/soccernet/extract_%j.out

source setup.sh
mkdir -p slurm_logs/soccernet

# Mirrors SOCCERNET_EXPERIMENT_DIR in src/config.py — keep in sync.
EXP_DIR=${EXP_DIR:-/cluster/work/projects/ec12/ec-eirikto/soccernet_thesis_experiment}
EXTRACT_FPS=${EXTRACT_FPS:-5.0}       # on-disk source FPS == probe target FPS
BATCH_SIZE=${BATCH_SIZE:-16}
MODE=${MODE:-dense}
# V-JEPA 2 only: must match the attentive probe's training W (default 10 raw
# frames @ 5 FPS = 2 s). Ignored by the DINOv3 modes.
WINDOW_SIZE=${WINDOW_SIZE:-10}

INPUT_DIR="${EXP_DIR}"
OUTPUT_DIR="${EXP_DIR}/features"
mkdir -p "${OUTPUT_DIR}"

MODEL=dinov3
MODEL_SIZE=large
case "${MODE}" in
    dense)
        PADDING_MODE=reflect
        EXTRA_FLAGS=(--save-dense --skip-cls)
        OUT_DESCR="DINOv3-L dense (reflect)"
        OUT_FILE="<stem>_dinov3_l_${EXTRACT_FPS}fps_reflect_dense_features.npy"
        ;;
    cls)
        PADDING_MODE=center_crop
        EXTRA_FLAGS=()
        OUT_DESCR="DINOv3-L CLS (centre-crop)"
        OUT_FILE="<stem>_dinov3_l_${EXTRACT_FPS}fps_features.npz"
        ;;
    vjepa2)
        MODEL=vjepa2
        PADDING_MODE=reflect
        # V-JEPA 2 has its own arg vocabulary: --feature-type + --window-size,
        # no CLS/dense flags. Stride is auto (== source_fps/target_fps).
        EXTRA_FLAGS=(--feature-type dense --window-size "${WINDOW_SIZE}")
        OUT_DESCR="V-JEPA 2-L dense (reflect, W=${WINDOW_SIZE})"
        OUT_FILE="<stem>_vjepa2_l_${EXTRACT_FPS}fps_reflect_dense_w${WINDOW_SIZE}.npz"
        ;;
    *)
        echo "ERROR: MODE='${MODE}' is not recognised. Use MODE=dense, MODE=cls, or MODE=vjepa2."
        exit 1
        ;;
esac

echo "=========================================="
echo "SoccerNet ${OUT_DESCR} extraction @ ${EXTRACT_FPS} FPS"
echo "  input  : ${INPUT_DIR}  (top-level *.mp4 / *.mkv)"
echo "  output : ${OUTPUT_DIR}/${OUT_FILE}"
echo "=========================================="

# Sanity: the download step should have dropped a video file somewhere under
# the experiment root (SoccerNet nests it as <league>/<season>/<match>/2_720p.mkv).
if [ -z "$(find "${INPUT_DIR}" -type f \( -name '*.mp4' -o -name '*.mkv' \) -print -quit)" ]; then
    echo "ERROR: no .mp4/.mkv anywhere under ${INPUT_DIR}. Run download_half.py first."
    exit 1
fi

uv run python -u extract_features.py \
    --model "${MODEL}" \
    --size "${MODEL_SIZE}" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --fps "${EXTRACT_FPS}" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda \
    --padding-mode "${PADDING_MODE}" \
    "${EXTRA_FLAGS[@]}"

echo "Extraction done (MODE=${MODE}). Output in ${OUTPUT_DIR}/"
case "${MODE}" in
    dense)  echo "Next: MODEL_SUFFIX=<your_best> sbatch soccernet_experiment/run_predict.sh" ;;
    vjepa2) echo "Next: MODEL_SUFFIX=<your_best> sbatch soccernet_experiment/run_predict_vjepa2.sh" ;;
esac
