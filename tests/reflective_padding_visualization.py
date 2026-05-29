"""
Visualise the reflective-padding preprocessing path against the default
centre-crop path. Reads one frame (DINOv3) and one W=10 window (V-JEPA 2)
from the first TACDEC video and writes side-by-side PNGs to
``figures/reflective_padding/``.

The backbones are NOT loaded here -- only the geometric preprocessing steps
that live inside the extractors. That keeps the script runnable without HF
tokens, CUDA, or model downloads.

Usage:
    python tests/reflective_padding_visualization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.feature_extractors.base_extractor import BaseFeatureExtractor

VIDEO_DIR = ROOT / "data" / "TACDEC" / "videos"
OUT_DIR = ROOT / "figures" / "reflective_padding"

# DINOv3 attentive probe uses W=10 @ 5 FPS = 2 s windows (see memory).
WINDOW_SIZE = 10
SOURCE_FPS = 25.0
TARGET_FPS = 5.0


def center_crop_to_square(frame: np.ndarray) -> np.ndarray:
    """Match the HF processor centre-crop path: shortest_edge=256 then crop."""
    h, w = frame.shape[:2]
    short = min(h, w)
    scale = 256.0 / short
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (new_h - 256) // 2
    x0 = (new_w - 256) // 2
    return resized[y0:y0 + 256, x0:x0 + 256]


def reflect_pad_to_256(frame: np.ndarray) -> np.ndarray:
    """Match the new reflect path: square via BORDER_REFLECT_101 then resize."""
    squared = BaseFeatureExtractor._square_with_reflect(frame)
    return cv2.resize(squared, (256, 256), interpolation=cv2.INTER_AREA)


def first_video() -> Path:
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        sys.exit(f"ERROR: no .mp4 found in {VIDEO_DIR}")
    return videos[0]


def read_frames(video_path: Path, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open {video_path}")
    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    i = 0
    while wanted - out.keys():
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    missing = wanted - out.keys()
    if missing:
        sys.exit(f"ERROR: video too short, missing frame indices {sorted(missing)}")
    return [out[i] for i in indices]


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def label_strip(width: int, text: str, height: int = 28) -> np.ndarray:
    strip = np.full((height, width, 3), 240, dtype=np.uint8)
    cv2.putText(strip, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return strip


def stack_with_labels(panels: list[tuple[str, np.ndarray]], pad: int = 12) -> np.ndarray:
    """Horizontally stitch labelled panels. Panels can have different sizes;
    each one is placed on its own white canvas of the row's max height."""
    max_h = max(p.shape[0] for _, p in panels)
    cols = []
    for title, p in panels:
        canvas = np.full((max_h, p.shape[1], 3), 255, dtype=np.uint8)
        y0 = (max_h - p.shape[0]) // 2
        canvas[y0:y0 + p.shape[0]] = p
        labelled = np.vstack([label_strip(p.shape[1], title), canvas])
        cols.append(labelled)
    spacer = np.full((cols[0].shape[0], pad, 3), 255, dtype=np.uint8)
    out = cols[0]
    for c in cols[1:]:
        out = np.hstack([out, spacer, c])
    return out


def grid_window(frames: list[np.ndarray], ncols: int = 5, pad: int = 6) -> np.ndarray:
    h, w = frames[0].shape[:2]
    nrows = (len(frames) + ncols - 1) // ncols
    canvas = np.full((nrows * h + (nrows - 1) * pad,
                       ncols * w + (ncols - 1) * pad, 3), 255, dtype=np.uint8)
    for k, f in enumerate(frames):
        r, c = divmod(k, ncols)
        y = r * (h + pad)
        x = c * (w + pad)
        canvas[y:y + h, x:x + w] = f
    return canvas


def dinov3_example(video_path: Path) -> None:
    print(f"[DINOv3] sampling frame 0 from {video_path.name}")
    frame = read_frames(video_path, [0])[0]
    h, w = frame.shape[:2]

    reflect_square = BaseFeatureExtractor._square_with_reflect(frame)
    reflect_256 = cv2.resize(reflect_square, (256, 256), interpolation=cv2.INTER_AREA)
    center_256 = center_crop_to_square(frame)

    save_rgb(OUT_DIR / "dinov3_original.png", frame)
    save_rgb(OUT_DIR / "dinov3_reflect_square.png", reflect_square)
    save_rgb(OUT_DIR / "dinov3_reflect_256.png", reflect_256)
    save_rgb(OUT_DIR / "dinov3_center_crop_256.png", center_256)

    comparison = stack_with_labels([
        (f"original {w}x{h}", frame),
        (f"reflect square {reflect_square.shape[1]}x{reflect_square.shape[0]}",
         reflect_square),
        ("reflect -> 256x256", reflect_256),
        ("center-crop 256x256", center_256),
    ])
    save_rgb(OUT_DIR / "dinov3_comparison.png", comparison)
    print(f"  wrote {OUT_DIR / 'dinov3_comparison.png'}")


def vjepa2_example(video_path: Path) -> None:
    stride = max(1, int(round(SOURCE_FPS / TARGET_FPS)))
    indices = [i * stride for i in range(WINDOW_SIZE)]
    print(f"[V-JEPA 2] sampling W={WINDOW_SIZE} frames at stride={stride} "
          f"-> source indices {indices}")
    frames = read_frames(video_path, indices)

    reflect_frames = [
        cv2.resize(BaseFeatureExtractor._square_with_reflect(f), (256, 256),
                   interpolation=cv2.INTER_AREA)
        for f in frames
    ]
    center_frames = [center_crop_to_square(f) for f in frames]

    reflect_grid = grid_window(reflect_frames, ncols=5)
    center_grid = grid_window(center_frames, ncols=5)

    save_rgb(OUT_DIR / "vjepa2_reflect_window.png", reflect_grid)
    save_rgb(OUT_DIR / "vjepa2_center_crop_window.png", center_grid)

    comparison = stack_with_labels([
        (f"reflect-pad window (W={WINDOW_SIZE}, 256x256 each)", reflect_grid),
        (f"center-crop window (W={WINDOW_SIZE}, 256x256 each)", center_grid),
    ])
    save_rgb(OUT_DIR / "vjepa2_comparison.png", comparison)
    print(f"  wrote {OUT_DIR / 'vjepa2_comparison.png'}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = first_video()
    dinov3_example(video_path)
    vjepa2_example(video_path)
    print(f"\nAll outputs under: {OUT_DIR}")


if __name__ == "__main__":
    main()
