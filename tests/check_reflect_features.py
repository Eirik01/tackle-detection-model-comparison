"""
Verify the reflect-padded dense feature files exist at the paths the temporal
loaders expect. Exercises the same DINOv3DenseLoader / VJEPA2DenseLoader code
that train_temporal.py and eval_temporal.py use, so a green run here means
``PADDING_MODE=reflect sbatch run_train_eval_temporal_*.sh`` will resolve every
clip's feature path.

Usage on Fox:
    cd /path/to/thesis_code
    uv run python tests/check_reflect_features.py
    uv run python tests/check_reflect_features.py --backbone vjepa2
    uv run python tests/check_reflect_features.py --backbone both --limit 5

What it does:
    1. Counts how many *_reflect_dense_* files live in the configured features dir.
    2. Picks a few clip IDs from the labels dir and instantiates the matching
       loader with ``dense_tag="reflect"``.
    3. Calls ``get_feature(video_id, anchor=valid_lo)`` to confirm the file
       actually opens and returns a tensor of the right shape.

Exit code 0 on success, 1 if any required file is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from src.config import (
    TACDEC_FEATURES_DINOV3,
    TACDEC_FEATURES_VJEPA2,
    TACDEC_LABELS,
)
from src.data.temporal_loaders import (
    DINOv3DenseLoader,
    VJEPA2DenseLoader,
)


# Defaults match the reflect runs the temporal wrappers will trigger.
DINOV3_SOURCE_FPS = 25.0
VJEPA2_FPS = 5.0
WINDOW_SIZE = 10


def list_clip_ids(labels_dir: Path) -> list[str]:
    return sorted(p.stem for p in labels_dir.glob("*.json"))


def check_dinov3(limit: int) -> int:
    feats = Path(TACDEC_FEATURES_DINOV3)
    pattern = f"*_dinov3_l_{DINOV3_SOURCE_FPS}fps_reflect_dense_features.npy"
    matches = sorted(feats.glob(pattern))
    print(f"[DINOv3] features dir : {feats}")
    print(f"[DINOv3] glob pattern : {pattern}")
    print(f"[DINOv3] files found  : {len(matches)}")
    if not matches:
        print("[DINOv3] FAIL: no reflect dense files present.")
        return 1

    sample_ids = [p.name.split("_dinov3_l_")[0] for p in matches[:limit]]
    loader = DINOv3DenseLoader(
        features_dir=feats,
        fps=VJEPA2_FPS,                       # target FPS used at probe time
        window_size=WINDOW_SIZE,
        source_fps=DINOV3_SOURCE_FPS,         # source FPS of the on-disk file
        max_cached=1,
        dense_tag="reflect",
    )
    ok = 0
    for vid in sample_ids:
        try:
            feat = loader.get_feature(vid, anchor=0)
        except Exception as e:
            print(f"  [{vid}] LOAD FAIL: {type(e).__name__}: {e}")
            continue
        print(f"  [{vid}] OK  shape={tuple(feat.shape)}  dtype={feat.dtype}")
        ok += 1
    if ok < len(sample_ids):
        print(f"[DINOv3] FAIL: {len(sample_ids) - ok}/{len(sample_ids)} samples failed to load.")
        return 1
    print(f"[DINOv3] PASS ({ok}/{ok} samples loaded)")
    return 0


def check_vjepa2(limit: int) -> int:
    feats = Path(TACDEC_FEATURES_VJEPA2)
    pattern = f"*_vjepa2_l_{VJEPA2_FPS}fps_reflect_dense_w{WINDOW_SIZE}*.npz"
    matches = sorted(feats.glob(pattern))
    print(f"[V-JEPA 2] features dir : {feats}")
    print(f"[V-JEPA 2] glob pattern : {pattern}")
    print(f"[V-JEPA 2] files found  : {len(matches)}")
    if not matches:
        print("[V-JEPA 2] FAIL: no reflect dense files present.")
        return 1

    sample_ids = [p.name.split("_vjepa2_l_")[0] for p in matches[:limit]]
    loader = VJEPA2DenseLoader(
        features_dir=feats,
        fps=VJEPA2_FPS,
        window_size=WINDOW_SIZE,
        max_cached=1,
        dense_tag="reflect",
    )
    ok = 0
    for vid in sample_ids:
        try:
            # Peek at valid_lo so we ask for a definitely-valid anchor.
            arr = loader._open_video(vid)
            anchor = int(arr["valid_lo"])
            feat = loader.get_feature(vid, anchor=anchor)
        except Exception as e:
            print(f"  [{vid}] LOAD FAIL: {type(e).__name__}: {e}")
            continue
        print(f"  [{vid}] OK  shape={tuple(feat.shape)}  dtype={feat.dtype}  "
              f"valid_lo={anchor}")
        ok += 1
    if ok < len(sample_ids):
        print(f"[V-JEPA 2] FAIL: {len(sample_ids) - ok}/{len(sample_ids)} samples failed to load.")
        return 1
    print(f"[V-JEPA 2] PASS ({ok}/{ok} samples loaded)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["dinov3", "vjepa2", "both"],
                    default="both")
    ap.add_argument("--limit", type=int, default=3,
                    help="Number of sample videos to fully open per backbone.")
    args = ap.parse_args()

    print("=" * 60)
    print("Reflect-padded dense feature presence check")
    print("=" * 60)
    print(f"Labels dir : {TACDEC_LABELS}")
    print(f"# clip IDs : {len(list_clip_ids(Path(TACDEC_LABELS)))}")
    print("=" * 60)

    rc = 0
    if args.backbone in ("dinov3", "both"):
        rc |= check_dinov3(args.limit)
        print("-" * 60)
    if args.backbone in ("vjepa2", "both"):
        rc |= check_vjepa2(args.limit)
        print("-" * 60)

    print("RESULT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())
