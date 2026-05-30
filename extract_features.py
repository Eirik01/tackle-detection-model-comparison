#!/usr/bin/env python3
"""
Feature Extraction Entry Point
Master's Thesis Project - Action Spotting with Vision Transformers

Usage:
    python extract_features.py --model dinov3 --size base
    python extract_features.py --model vjepa2
"""

import argparse
from src.feature_extractors import DINOv3Extractor, VJEPA2Extractor
from src.config import TACDEC_VIDEOS, TACDEC_FEATURES


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from videos using vision transformers"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["dinov3", "vjepa2"],
        default="dinov3",
        help="Feature extractor model to use"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["base", "large"],
        default="base",
        help="Model size (for DINOv3)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=TACDEC_VIDEOS,
        help="Input directory containing videos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=TACDEC_FEATURES,
        help="Output directory for extracted features"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25,
        help="Target FPS for frame sampling (default: use original FPS)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing frames"
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-extract and overwrite existing feature files"
    )
    parser.add_argument(
        "--profile-efficiency",
        action="store_true",
        help="Profile computational efficiency (time, memory) and output to CSV"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index for batch processing (inclusive, 0-based). E.g., --start-idx 0 --end-idx 50 processes videos 0-49"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index for batch processing (exclusive, 0-based). E.g., --start-idx 50 --end-idx 100 processes videos 50-99"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--save-dense",
        action="store_true",
        help="DINOv3 only: also save dense patch features alongside CLS as "
             "{video_id}_..._dense_features.npy (fp16, shape (T, num_patches, D))."
    )
    parser.add_argument(
        "--skip-cls",
        action="store_true",
        help="DINOv3 only: do not write the CLS .npz output. Useful when the "
             "padding mode matches what's needed for dense (e.g. reflect for "
             "the attentive probe) but doesn't match the linear probe's CLS "
             "preprocessing, so the CLS file would be dead weight."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="V-JEPA2 only: number of raw frames per forward (default 16). "
             "Set to W (e.g. 50) for the patch-token attentive probe protocol."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="V-JEPA2 only: anchor stride in source frames between successive "
             "windows. If unset, auto-computes from --fps. Set to 1 for stride-1 "
             "sliding feature extraction (one forward per source frame)."
    )
    parser.add_argument(
        "--intra-window-stride",
        type=int,
        default=None,
        help="V-JEPA2 only: stride between adjacent frames *inside* one V-JEPA2 "
             "input clip. Default = anchor stride (=> shared 5 FPS protocol where "
             "DINOv3 and V-JEPA2 see the same source frames). Set to 1 for the "
             "legacy 'consecutive raw frames at 25 FPS' behaviour."
    )
    parser.add_argument(
        "--padding-mode",
        type=str,
        choices=["center_crop", "reflect"],
        default="center_crop",
        help="How to square the input frame before the backbone. 'center_crop' "
             "keeps the existing shortest_edge=256 + centre-crop pipeline. "
             "'reflect' mirrors the frame's borders (BORDER_REFLECT_101) to "
             "pad the shorter side until the frame is 1:1, then resizes to "
             "256x256 -- no pixels are cropped away. Output files get a "
             "'_reflect' tag to keep both runs side by side on disk."
    )

    args = parser.parse_args()
    
    # Initialize extractor
    if args.model == "dinov3":
        extractor = DINOv3Extractor(
            input_dir=args.input,
            output_dir=args.output,
            model_size=args.size,
            device=args.device,
            padding_mode=args.padding_mode,
        )
    elif args.model == "vjepa2":
        extractor = VJEPA2Extractor(
            input_dir=args.input,
            output_dir=args.output,
            model_size=args.size,
            device=args.device,
            padding_mode=args.padding_mode,
        )
    else:
        raise NotImplementedError(f"Model '{args.model}' not yet implemented")

    # Note: both DINOv3Extractor and VJEPA2Extractor call self.load_model() in
    # __init__, so the backbone is already loaded by the time we get here.
    # Don't re-load -- it costs 5-15s and momentary GPU memory thrash.

    # Extract features
    print("\n🚀 Starting feature extraction...")
    extract_kwargs = dict(
        fps=args.fps,
        batch_size=args.batch_size,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        override=args.override,
        profile_efficiency=args.profile_efficiency,
    )
    if args.model == "dinov3":
        extract_kwargs["save_dense"] = args.save_dense
        extract_kwargs["skip_cls"] = args.skip_cls
        if args.window_size != 16 or args.stride is not None:
            print("⚠️  --window-size / --stride are V-JEPA2-only and will be ignored "
                  "for DINOv3 (per-frame extractor).")
    elif args.model == "vjepa2":
        extract_kwargs["window_size"] = args.window_size
        extract_kwargs["stride"] = args.stride
        extract_kwargs["intra_window_stride"] = args.intra_window_stride
        if args.save_dense:
            print("⚠️  --save-dense is DINOv3-only; V-JEPA2 always writes dense features.")
        if args.skip_cls:
            print("⚠️  --skip-cls is DINOv3-only and will be ignored for V-JEPA2.")
    extractor.extract_features(**extract_kwargs)
    
    print("\n✅ Feature extraction complete!")


if __name__ == "__main__":
    main()
