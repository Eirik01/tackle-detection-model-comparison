"""
Visualize YOLO crops on a video.

Reads a crop .npz file and the corresponding video, and produces a
side-by-side annotated MP4 showing:
  - Original frame with the crop bounding box drawn on it
  - The cropped + zero-padded region as the feature extractor would see it

Usage:
    python visualization/crops.py \
        --crops data/TACDEC/crops_test/1738_avxeiaxxw6ocr_yolo_5.0fps_crops.npz \
        --video data/TACDEC/videos/1738_avxeiaxxw6ocr.mp4
"""
import argparse
import cv2
import numpy as np
from pathlib import Path


def apply_crop(frame, x1, y1, x2, y2, target_size=256):
    """Crop and zero-pad to square (mirrors dinov3_extractor._apply_crop)."""
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    dy, dx = (size - h) // 2, (size - w) // 2
    padded[dy:dy + h, dx:dx + w] = crop
    return cv2.resize(padded, (target_size, target_size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', required=True, type=Path)
    ap.add_argument('--video', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=None,
                    help='Output MP4 (default: <crops_stem>_viz.mp4 next to crops file)')
    args = ap.parse_args()

    crops = np.load(args.crops)['crops']
    n_total = len(crops)
    n_valid = int((crops[:, 0] >= 0).sum())
    print(f'Loaded {n_total} crops, {n_valid} valid ({n_valid/n_total*100:.1f}%)')

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {args.video}')

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames')

    # Infer skip_interval from how many sampled frames the crop file contains
    skip_interval = max(1, round(total_frames / n_total))
    print(f'Inferred skip_interval: {skip_interval} (target ≈ {fps/skip_interval:.1f} fps)')

    out_path = args.out or args.crops.with_name(args.crops.stem + '_viz.mp4')
    crop_panel = 256
    out_w = width + crop_panel
    out_h = max(height, crop_panel)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        max(2.0, fps / skip_interval),
        (out_w, out_h),
    )

    sampled_idx = 0
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f_idx % skip_interval == 0 and sampled_idx < n_total:
            x1, y1, x2, y2 = [int(v) for v in crops[sampled_idx]]
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            display = frame.copy()

            if x1 >= 0:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                crop_img = apply_crop(frame, x1, y1, x2, y2, target_size=crop_panel)
                label = f'sample {sampled_idx}/{n_total}  box=({x1},{y1})-({x2},{y2})'
            else:
                crop_img = np.zeros((crop_panel, crop_panel, 3), dtype=np.uint8)
                cv2.putText(display, 'FALLBACK', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                label = f'sample {sampled_idx}/{n_total}  FALLBACK (no valid crop)'

            canvas[:height, :width] = display
            if crop_img is not None:
                canvas[:crop_panel, width:width + crop_panel] = crop_img

            cv2.putText(canvas, label, (10, out_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(canvas)
            sampled_idx += 1
        f_idx += 1

    cap.release()
    writer.release()
    print(f'Saved annotated video: {out_path}')


if __name__ == '__main__':
    main()
