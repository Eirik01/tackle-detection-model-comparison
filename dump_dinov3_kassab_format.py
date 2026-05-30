#!/usr/bin/env python3
"""
Dump DINOv3 CLS features in Kassab's stacked-tensor format.

Produces one big `[N_total_frames, 1024]` float32 torch tensor containing every
DINOv3-Large CLS feature from every TACDEC video, concatenated in alphabetical
video order — i.e., the exact format Kassab's `sorted_cls_tokens_features.pt`
uses (just with DINOv3 instead of DINOv2 inside).

Drop the resulting .pt into `tacdec-kassab-implementation/` and point cell 3 of
`spatial-approach.ipynb` at it. All downstream steps (split, undersample,
meta-test extraction) then run identically to the DINOv2 baseline — only the
backbone changes. That makes the comparison an A/B test, not a protocol-equivalent
approximation.

Usage (on Fox):
    sbatch analysis/run_dump_dinov3_kassab_format.sh

Outputs:
    {output_dir}/dinov3_sorted_cls_tokens_features.pt    (float32 [N, 1024] tensor)
    {output_dir}/dinov3_frame_counts.npy                  (int64 [425] frame counts)
    {output_dir}/dinov3_video_ids.txt                     (425 lines, one video_id per line)
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import TACDEC_FEATURES, TACDEC_KASSAB_DUMP_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=25.0,
                        help="FPS suffix to match in feature filenames (default 25.0).")
    parser.add_argument("--padding-mode", choices=("center_crop", "reflect"),
                        default="center_crop",
                        help="Which preprocessing variant to stack. center_crop "
                             "reads *_<fps>fps_features.npz; reflect reads "
                             "*_<fps>fps_reflect_features.npz.")
    parser.add_argument("--features-dir", type=str, default=None,
                        help=f"Override features dir. Default: {TACDEC_FEATURES}/dinov3_large")
    parser.add_argument("--output-dir", type=str,
                        default=str(TACDEC_KASSAB_DUMP_DIR),
                        help=f"Where to write the stacked tensor and metadata. "
                             f"Default: {TACDEC_KASSAB_DUMP_DIR}")
    args = parser.parse_args()

    features_dir = Path(args.features_dir) if args.features_dir else (TACDEC_FEATURES / "dinov3_large")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pad_tag = "_reflect" if args.padding_mode == "reflect" else ""

    print("=" * 70)
    print("Dump DINOv3 features in Kassab stacked-tensor format")
    print("=" * 70)
    print(f"  Features dir: {features_dir}")
    print(f"  FPS:          {args.fps}")
    print(f"  Padding mode: {args.padding_mode}")
    print(f"  Output dir:   {output_dir}")
    print("=" * 70)

    pattern = re.compile(
        rf"(.+?)_dinov3_[bl]_\d+(?:\.\d+)?fps{pad_tag}_features$"
    )
    fps_str = f"{args.fps}fps"
    glob_pat = f"*_{fps_str}{pad_tag}_features.npz"

    print(f"\nScanning {features_dir} for {glob_pat} ...")
    feat_paths = sorted(features_dir.glob(glob_pat))
    print(f"  found {len(feat_paths)} files")
    if not feat_paths:
        raise SystemExit("No matching feature files found.")

    video_ids = []
    frame_counts = []
    cls_arrays = []

    print("\nLoading and stacking ...")
    for i, feat_path in enumerate(feat_paths):
        m = pattern.match(feat_path.stem)
        if not m:
            print(f"  [skip] couldn't parse video_id from {feat_path.name}")
            continue
        video_id = m.group(1)
        cls = np.load(feat_path)["cls"].astype(np.float32)   # [N_i, 1024]
        video_ids.append(video_id)
        frame_counts.append(int(cls.shape[0]))
        cls_arrays.append(cls)
        if i < 10 or i == len(feat_paths) - 1:
            print(f"  [{i:3d}] {video_id}  cls.shape={cls.shape}")
        elif i == 10:
            print("  ...")

    print("\nConcatenating ...")
    big_cls = np.concatenate(cls_arrays, axis=0)           # [N_total, 1024]
    print(f"  stacked tensor shape: {big_cls.shape}  dtype={big_cls.dtype}")
    print(f"  size on disk: {big_cls.nbytes / 1e9:.2f} GB")

    expected_total = sum(frame_counts)
    assert big_cls.shape[0] == expected_total, \
        f"Total mismatch: {big_cls.shape[0]} vs sum(frame_counts)={expected_total}"

    name_tag = "_reflect" if args.padding_mode == "reflect" else ""

    print("\nSaving ...")
    pt_path = output_dir / f"dinov3{name_tag}_sorted_cls_tokens_features.pt"
    torch.save(torch.from_numpy(big_cls), pt_path)
    print(f"  → {pt_path}")

    fc_path = output_dir / f"dinov3{name_tag}_frame_counts.npy"
    np.save(fc_path, np.array(frame_counts, dtype=np.int64))
    print(f"  → {fc_path}")

    ids_path = output_dir / f"dinov3{name_tag}_video_ids.txt"
    with open(ids_path, "w") as f:
        for vid in video_ids:
            f.write(vid + "\n")
    print(f"  → {ids_path}")

    print("\nSanity checks:")
    print(f"  videos:                   {len(video_ids)}")
    print(f"  total frames:             {expected_total:,}")
    print(f"  first 10 frame counts:    {frame_counts[:10]}")
    print(f"  first 10 video IDs:       {video_ids[:10]}")
    print()
    print("Expected (verified earlier): first 10 frame counts =")
    print("  [600, 500, 650, 350, 350, 450, 550, 650, 650, 650]")

    print("\nDone. Copy the .pt to tacdec-kassab-implementation/ and point cell 3 at it:")
    print(f"  scp <fox>:{pt_path} tacdec-kassab-implementation/")
    print(f"  # then in cell 3:  X = torch.load('./dinov3_sorted_cls_tokens_features.pt', ...)")


if __name__ == "__main__":
    main()
