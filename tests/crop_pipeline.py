"""
Local end-to-end test of the YOLO crop + feature extraction pipeline.
Runs on a single video using local paths (no Fox required).

Usage:
    python tests/crop_pipeline.py
    python tests/crop_pipeline.py --backbone vjepa2
    python tests/crop_pipeline.py --no-crops   # test full-frame extraction only
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_extractors.yolo_crop_extractor import YOLOCropExtractor, extract_crops_for_video
from src.feature_extractors.dinov3_extractor import DINOv3Extractor
from src.feature_extractors.vjepa2_extractor import VJEPA2Extractor

LOCAL_VIDEOS = ROOT / "data" / "TACDEC" / "videos"
LOCAL_CROPS  = ROOT / "data" / "TACDEC" / "crops_test"
LOCAL_FEATS  = ROOT / "data" / "TACDEC" / "features_test"
YOLO_WEIGHTS = ROOT / "yolov8m_forzasys_soccer.pt"

FPS = 2.0
BATCH_SIZE = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["dinov3", "vjepa2"], default="dinov3")
    parser.add_argument("--no-crops", action="store_true", help="Skip crops, use full frame")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    video_path = next(iter(sorted(LOCAL_VIDEOS.glob("*.mp4"))), None)
    if video_path is None:
        print(f"ERROR: no .mp4 found in {LOCAL_VIDEOS}")
        sys.exit(1)

    print(f"Test video: {video_path.name}")
    print(f"Backbone:   {args.backbone}")
    print(f"Crops:      {'no (full frame)' if args.no_crops else 'yes (YOLO)'}")
    print(f"Device:     {args.device}")
    print("=" * 60)

    LOCAL_CROPS.mkdir(parents=True, exist_ok=True)
    LOCAL_FEATS.mkdir(parents=True, exist_ok=True)

    crop_dir = None

    # --- Step 1: YOLO crop extraction ---
    if not args.no_crops:
        print("\n[1/2] Extracting YOLO crops...")
        extractor = YOLOCropExtractor(
            input_dir=LOCAL_VIDEOS,
            output_dir=LOCAL_CROPS,
            yolo_weights=YOLO_WEIGHTS,
            device=args.device,
        )
        crop_path = YOLOCropExtractor.get_crop_path(LOCAL_CROPS, video_path.stem, FPS)
        if crop_path.exists():
            print(f"  Crop file already exists: {crop_path.name} — skipping")
        else:
            extract_crops_for_video(video_path, extractor.yolo, FPS, crop_path)
            print(f"  Saved: {crop_path}")
        crop_dir = LOCAL_CROPS
    else:
        print("\n[1/2] Skipping YOLO crops (--no-crops)")

    # --- Step 2: Feature extraction ---
    print(f"\n[2/2] Extracting {args.backbone} features...")
    output_dir = LOCAL_FEATS / f"{args.backbone}_local_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backbone == "dinov3":
        feat_extractor = DINOv3Extractor(
            input_dir=LOCAL_VIDEOS,
            output_dir=output_dir,
            model_size="base",
            device=args.device,
            crop_dir=crop_dir,
        )
    else:
        feat_extractor = VJEPA2Extractor(
            input_dir=LOCAL_VIDEOS,
            output_dir=output_dir,
            model_size="large",
            device=args.device,
            crop_dir=crop_dir,
        )

    # Only process the one test video
    feat_extractor.extract_features(
        fps=FPS,
        batch_size=BATCH_SIZE,
        start_idx=0,
        end_idx=1,
        override=True,
    )

    # --- Verify output ---
    import numpy as np
    output_files = sorted(output_dir.glob("*.npz"))
    if output_files:
        data = np.load(output_files[-1])
        print(f"\n✅ Feature file: {output_files[-1].name}")
        print(f"   Shape: {data['cls'].shape}  (frames × feature_dim)")
    else:
        print("\n❌ No output .npz found — something went wrong")


if __name__ == "__main__":
    main()
