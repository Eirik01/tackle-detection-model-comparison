"""Show the temporal window a probe actually sees around a centre/anchor frame.

Usage:
    python -m analysis.show_window <clip> <anchor_idx> [options]

`clip` is a clip id (resolved against data/TACDEC/videos/) or a path to .mp4.
`anchor_idx` is the centre-frame index in the TARGET (5 FPS) grid -- i.e. the
`frame_idx` column from a temporal misclassifications CSV.

The window frames are taken from the SAME source-of-truth used at extraction
and training time (`window_protocol.select_source_frames`), so what you see is
byte-identical to the probe's input. For the default 5 FPS / W=10 protocol the
window spans source-frame offsets [-20,-15,-10,-5,0,5,10,15,20,25] around the
centre (lower-middle convention), i.e. 1.8 s of video.

Outputs (default: both):
  - a contact-sheet PNG (2x5 grid) with per-tile frame index, time, and label,
    the centre tile outlined.
  - an .mp4 of the W frames (with --video / on by default; disable with --no-video).

Examples:
    python -m analysis.show_window 3266_a7t1iblhbw2sg 42
    python -m analysis.show_window 3266_a7t1iblhbw2sg 42 --view center_crop
    python -m analysis.show_window 3266_a7t1iblhbw2sg 42 --no-video --no-labels
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse the single-frame helpers and the shared window protocol.
from analysis.show_frame import (
    DEFAULT_LABELS_DIR,
    DEFAULT_VIDEO_DIR,
    REPO_ROOT,
    VIEWS,
    annotate,
    load_frame_labels,
    read_frame_rgb,
    resolve_video,
)

sys.path.insert(0, str(REPO_ROOT / "src"))
from window_protocol import select_source_frames  # noqa: E402


def video_frame_count(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def make_contact_sheet(tiles: list[np.ndarray], cols: int = 5, gap: int = 4) -> np.ndarray:
    n = len(tiles)
    rows = (n + cols - 1) // cols
    h, w = tiles[0].shape[:2]
    sheet = np.full((rows * h + (rows - 1) * gap,
                     cols * w + (cols - 1) * gap, 3), 30, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (h + gap), c * (w + gap)
        sheet[y:y + h, x:x + w] = tile
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="clip id or path to .mp4")
    ap.add_argument("anchor_idx", type=int,
                    help="centre-frame index in the TARGET (5 FPS) grid "
                         "(the frame_idx column of a temporal misclassifications CSV)")
    ap.add_argument("--view", choices=list(VIEWS), default="reflect",
                    help="how to render each frame (default: reflect, matches the run)")
    ap.add_argument("--target-fps", type=float, default=5.0)
    ap.add_argument("--source-fps", type=float, default=25.0)
    ap.add_argument("--window-size", type=int, default=10, help="W (default 10)")
    ap.add_argument("--cols", type=int, default=2,
                    help="contact-sheet columns (default 2 -> 2 wide; 5 -> 5x2; "
                         "3 -> 3,3,3,1 for W=10)")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp"),
                    help="where to write the PNG / MP4 (default: /tmp)")
    ap.add_argument("--play-fps", type=float, default=5.0,
                    help="playback fps of the output mp4 (default: 5)")
    ap.add_argument("--no-video", action="store_true", help="skip the mp4")
    ap.add_argument("--no-labels", action="store_true",
                    help="do not read the label file / draw label badges")
    ap.add_argument("--no-open", action="store_true",
                    help="do not open outputs after writing")
    args = ap.parse_args()

    if args.source_fps % args.target_fps != 0:
        ap.error("source-fps must be an integer multiple of target-fps")
    stride = int(args.source_fps // args.target_fps)

    video = resolve_video(args.clip)
    clip_name = video.stem
    n_src = video_frame_count(video)

    src_frames = select_source_frames(
        args.anchor_idx, n_src,
        anchor_stride=stride, intra_window_stride=stride,
        window_length=args.window_size, boundary="clamp",
    )
    centre_src = args.anchor_idx * stride

    classify = None if args.no_labels else load_frame_labels(clip_name, DEFAULT_LABELS_DIR)
    view_fn = VIEWS[args.view]

    tiles, raw_tiles = [], []
    for f in src_frames:
        rgb = read_frame_rgb(video, f)
        bgr = cv2.cvtColor(view_fn(rgb), cv2.COLOR_RGB2BGR)
        raw_tiles.append(bgr.copy())
        label = classify(f) if classify else None
        tiles.append(annotate(bgr, f, f / args.source_fps, label, f == centre_src))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{clip_name}_a{args.anchor_idx}_{args.view}_window"
    outputs = []

    sheet_path = args.out_dir / f"{stem}.png"
    cv2.imwrite(str(sheet_path), make_contact_sheet(tiles, cols=min(args.cols, len(tiles))))
    outputs.append(sheet_path)

    if not args.no_video:
        h, w = raw_tiles[0].shape[:2]
        mp4_path = args.out_dir / f"{stem}.mp4"
        vw = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.play_fps, (w, h))
        for bgr in raw_tiles:
            vw.write(bgr)
        vw.release()
        outputs.append(mp4_path)

    span = (src_frames[-1] - src_frames[0]) / args.source_fps
    print(f"clip={clip_name} anchor={args.anchor_idx} (src centre={centre_src}, "
          f"t={centre_src / args.source_fps:.2f}s)")
    print(f"window source frames ({args.window_size}): {src_frames}")
    print(f"span={span:.2f}s  view={args.view}")
    for p in outputs:
        print(f"  -> {p}")

    if not args.no_open and sys.platform == "darwin":
        for p in outputs:
            subprocess.run(["open", str(p)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
