"""Open a single frame from a TACDEC clip.

Usage:
    python -m analysis.show_frame <clip> <frame_idx> [--view MODE] [--out PATH] [--no-open]

`clip` can be either:
  - a clip id (e.g. 3284_aqvzrpx4l4ug8), resolved against data/TACDEC/videos/
  - an explicit path to an .mp4 file

The `--view` flag controls what is shown:
  - raw           : the original video frame (no preprocessing)
  - center_crop   : shortest_edge=256 -> centre-crop 256x256, matches the
                    default HF processor pipeline used at feature extraction.
  - reflect       : BORDER_REFLECT_101 pad to square -> resize to 256x256,
                    matches the alternative `--padding-mode reflect` pipeline.

For the current k-fold runs the CSV frame index lives in the source video's
frame grid (extract fps == src fps == 25). If a future run extracts at a
different fps, pass --src-fps and --extract-fps so the index is rescaled.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = REPO_ROOT / "data" / "TACDEC" / "videos"
DEFAULT_LABELS_DIR = REPO_ROOT / "data" / "TACDEC" / "labels"

# Per-tile label badge text + BGR colours (shared with show_window).
LABEL_TAG = {"tackle-live": "tackle-live", "tackle-replay": "tackle-replay",
             "background": "background"}
LABEL_COLOR = {  # BGR
    "tackle-live": (60, 60, 220),
    "tackle-replay": (40, 160, 255),
    "background": (140, 140, 140),
}


def load_frame_labels(clip_id: str, labels_dir: Path):
    """Return a function mapping a SOURCE frame index -> class name, or None
    if no label file exists. Frames outside any event are 'background'."""
    path = labels_dir / f"{clip_id}.json"
    if not path.exists():
        return None
    events = json.loads(path.read_text()).get("events", [])

    def classify(src_frame: int) -> str:
        for ev in events:
            if ev["frame_start"] <= src_frame <= ev["frame_end"]:
                return ev["type"]
        return "background"

    return classify


def annotate(tile: np.ndarray, frame_idx: int, t: float, label: str | None,
             is_centre: bool) -> np.ndarray:
    """Draw frame idx / time / label badge on a BGR tile. Outline the centre.

    Overlay sizes were originally tuned for 256x256 tiles; everything scales
    with min(h, w) / 256 so raw 1280x720 tiles get readable text/badges and
    256x256 preprocessed tiles render identically to before.
    """
    h, w = tile.shape[:2]
    s = max(min(h, w) / 256.0, 1.0) * 1.68
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs_top, fs_tag = 0.38 * s, 0.42 * s
    thick = max(1, round(s))
    top_bar_h = max(16, round(16 * s))
    tag_bar_h = max(18, round(18 * s))
    pad = max(3, round(3 * s))
    outline = max(3, round(3 * s))

    cv2.rectangle(tile, (0, 0), (w - 1, top_bar_h), (0, 0, 0), -1)
    top = f"f{frame_idx}  {t:.2f}s"
    if is_centre:
        top += "  ANCHOR (window label)"
    cv2.putText(tile, top, (pad, top_bar_h - max(4, round(4 * s))),
                font, fs_top, (0, 255, 255) if is_centre else (255, 255, 255),
                thick, cv2.LINE_AA)
    if label is not None:
        tag = LABEL_TAG.get(label, label)
        col = LABEL_COLOR.get(label, (200, 200, 200))
        (tw, _), _ = cv2.getTextSize(tag, font, fs_tag, thick)
        cv2.rectangle(tile, (w - tw - 2 * pad, h - tag_bar_h), (w, h), col, -1)
        cv2.putText(tile, tag, (w - tw - pad, h - max(5, round(5 * s))),
                    font, fs_tag, (0, 0, 0), thick, cv2.LINE_AA)
    if is_centre:
        cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (0, 255, 255), outline)
    return tile


def resolve_video(clip: str) -> Path:
    p = Path(clip)
    if p.suffix == ".mp4" and p.exists():
        return p
    candidate = DEFAULT_VIDEO_DIR / f"{clip}.mp4"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find video for '{clip}' (looked at {candidate})")


def rescale_frame_idx(frame_idx: int, extract_fps: float, src_fps: float) -> int:
    if extract_fps == src_fps:
        return frame_idx
    return round(frame_idx * src_fps / extract_fps)


def read_frame_rgb(video: Path, src_frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise RuntimeError(f"Could not read frame {src_frame_idx} from {video}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def apply_center_crop(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """Match HF processor: resize shortest edge to `size`, then centre-crop."""
    h, w = rgb.shape[:2]
    if h < w:
        new_h = size
        new_w = round(w * size / h)
    else:
        new_w = size
        new_h = round(h * size / w)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    return resized[y0:y0 + size, x0:x0 + size]


def apply_reflect(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """Match base_extractor._square_with_reflect + resize to (size, size)."""
    h, w = rgb.shape[:2]
    if h != w:
        if h < w:
            pad = w - h
            top, bot = pad // 2, pad - pad // 2
            rgb = cv2.copyMakeBorder(rgb, top, bot, 0, 0, cv2.BORDER_REFLECT_101)
        else:
            pad = h - w
            left, right = pad // 2, pad - pad // 2
            rgb = cv2.copyMakeBorder(rgb, 0, 0, left, right, cv2.BORDER_REFLECT_101)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


VIEWS = {
    "raw": lambda f: f,
    "center_crop": apply_center_crop,
    "reflect": apply_reflect,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="clip id (e.g. 3284_aqvzrpx4l4ug8) or path to .mp4")
    ap.add_argument("frame_idx", type=int, help="frame index from the misclassifications CSV")
    ap.add_argument("--view", choices=list(VIEWS), default="center_crop",
                    help="what to show: raw frame, centre-cropped 256x256, or reflect-padded 256x256 (default: center_crop)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output PNG path (default: /tmp/<clip>_f<idx>_<view>.png)")
    ap.add_argument("--extract-fps", type=float, default=25.0,
                    help="fps used at feature extraction time (default: 25.0)")
    ap.add_argument("--src-fps", type=float, default=25.0,
                    help="fps of the source video (default: 25.0)")
    ap.add_argument("--no-labels", action="store_true",
                    help="do not draw the frame-id / time bar and the label badge")
    ap.add_argument("--no-open", action="store_true", help="do not open the image after saving")
    args = ap.parse_args()

    video = resolve_video(args.clip)
    src_idx = rescale_frame_idx(args.frame_idx, args.extract_fps, args.src_fps)

    rgb = read_frame_rgb(video, src_idx)
    processed = VIEWS[args.view](rgb)

    clip_name = video.stem
    out = args.out or Path(f"/tmp/{clip_name}_f{args.frame_idx}_{args.view}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    if not args.no_labels:
        classify = load_frame_labels(clip_name, DEFAULT_LABELS_DIR)
        label = classify(src_idx) if classify else None
        t = src_idx / args.src_fps
        bgr = annotate(bgr, src_idx, t, label, is_centre=False)
    cv2.imwrite(str(out), bgr)

    t = src_idx / args.src_fps
    print(f"clip={clip_name} frame_idx={args.frame_idx} src_frame={src_idx} t={t:.3f}s "
          f"view={args.view} shape={processed.shape[1]}x{processed.shape[0]} -> {out}")

    if not args.no_open and sys.platform == "darwin":
        subprocess.run(["open", str(out)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
