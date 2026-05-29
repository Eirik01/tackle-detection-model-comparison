"""
Verify that `tacdec-kassab-implementation/dinov3_sorted_cls_tokens_features.pt`
was produced with the documented preprocessing (shortest_edge=256, center-crop
256x256, ImageNet normalise) by re-extracting one frame through DINOv3Extractor
and comparing the CLS vector against the corresponding row of the stacked tensor.

Run on the cluster (or wherever you have TACDEC videos + HF token + a GPU
ideally). Picks the first video alphabetically and an interior frame.

Invoke as a module from the repo root (thesis_code/):
    uv run python -m src.verify_dinov3_kassab_features
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import TACDEC_VIDEOS
from .feature_extractors.dinov3_extractor import DINOv3Extractor

REPO_ROOT = Path(__file__).resolve().parent.parent
KASSAB_DIR = REPO_ROOT / "tacdec-kassab-implementation"
STACKED_PT = KASSAB_DIR / "dinov3_sorted_cls_tokens_features.pt"
FRAME_COUNTS = KASSAB_DIR / "frame_counts.npy"

CLIP_IDX = 0          # which alphabetically-sorted video to check
FRAME_WITHIN_CLIP = 100

videos = sorted(p for p in Path(TACDEC_VIDEOS).iterdir() if p.is_file())
clip = videos[CLIP_IDX]
print(f"Clip: {clip.name}  (alphabetical index {CLIP_IDX})")
print(f"Frame within clip: {FRAME_WITHIN_CLIP}")

frame_counts = np.load(FRAME_COUNTS)
assert FRAME_WITHIN_CLIP < frame_counts[CLIP_IDX], (
    f"FRAME_WITHIN_CLIP={FRAME_WITHIN_CLIP} >= clip length {frame_counts[CLIP_IDX]}"
)
global_row = int(frame_counts[:CLIP_IDX].sum() + FRAME_WITHIN_CLIP)
print(f"Global row in stacked tensor: {global_row}")

saved = torch.load(STACKED_PT, map_location="cpu")
print(f"Stacked tensor: shape={tuple(saved.shape)}, dtype={saved.dtype}")
saved_vec = saved[global_row].numpy().astype(np.float32)

cap = cv2.VideoCapture(str(clip))
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_WITHIN_CLIP)
ok, bgr = cap.read()
cap.release()
assert ok, f"Failed to read frame {FRAME_WITHIN_CLIP} from {clip}"
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nLoading DINOv3-Large on {device} ...")
extractor = DINOv3Extractor(
    input_dir=TACDEC_VIDEOS,
    output_dir="/tmp/dinov3_verify",
    model_size="large",
    device=device,
    padding_mode="center_crop",
)
fresh_vec = extractor.extract_frame_features([rgb])[0].astype(np.float32)

print(f"\nSaved vec  shape={saved_vec.shape}  mean={saved_vec.mean():+.4f}  std={saved_vec.std():.4f}")
print(f"Fresh vec  shape={fresh_vec.shape}  mean={fresh_vec.mean():+.4f}  std={fresh_vec.std():.4f}")

diff = saved_vec - fresh_vec
cos = float(
    (saved_vec @ fresh_vec) / (np.linalg.norm(saved_vec) * np.linalg.norm(fresh_vec))
)
print(f"\nCosine similarity: {cos:.6f}")
print(f"Max abs diff:      {np.abs(diff).max():.4e}")
print(f"Mean abs diff:     {np.abs(diff).mean():.4e}")

if cos > 0.9999:
    print("\nMATCH -- saved features match the documented center-crop pipeline.")
elif cos > 0.99:
    print("\nNEAR MATCH -- likely same pipeline, residual numeric noise (fp16/fp32, CUDA non-determinism).")
else:
    print("\nMISMATCH -- preprocessing differs, or wrong clip/frame, or the .pt was made with `reflect`.")
