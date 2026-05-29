"""
Quick sanity check: run YOLO + optical flow ball tracking on a sample video
and visualize the detections.

Usage: uv run python tests/yolo_detection.py [--video <filename or path>]
"""

import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import TACDEC_VIDEOS

WEIGHTS = ROOT / "yolov8m_forzasys_soccer.pt"
_video_dir = TACDEC_VIDEOS if TACDEC_VIDEOS.exists() else ROOT / "data" / "TACDEC" / "videos"
PLAYER_CONF = 0.5
BALL_CONF = 0.5
NUM_FRAMES = 20

PLAYER_CLASSES = {0}
BALL_CLASS = 1
SHOW_CLASSES = {0, 1}
MAX_PAIR_DIST_FRAC = 0.25

MAX_TRACKING_AGE_SEC = 1.0  # reset optical flow tracker if no YOLO reset within this window

# Optical flow params for Lucas-Kanade
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
)


def iqr_filter_ball(detections):
    """Remove spatially anomalous ball detections before optical flow tracking."""
    positions = {fi: pos for fi, pos in detections.items() if pos is not None}
    if len(positions) < 4:
        return detections

    xs = np.array([p[0] for p in positions.values()])
    ys = np.array([p[1] for p in positions.values()])

    def outlier_mask(arr):
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        return (arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)

    flags = outlier_mask(xs) | outlier_mask(ys)
    result = dict(detections)
    for idx, fi in enumerate(positions):
        if flags[idx]:
            result[fi] = None

    n = int(flags.sum())
    if n:
        print(f"IQR filter: removed {n} outlier ball detection(s)")
    return result


def get_player_crop(player_boxes, ball_pos, ball_boxes, frame_h, frame_w, padding=0.15):
    """
    Crop the union of the 2 players nearest the ball + the ball bounding box,
    ensuring the point of interaction is always inside the crop.
    Returns None (full frame fallback) if ball unknown, <2 players, or pair too far apart.
    """
    if ball_pos is None or len(player_boxes) < 2:
        return None

    bcx, bcy = ball_pos
    players = []
    for b in player_boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        players.append({"cx": (x1+x2)/2, "cy": (y1+y2)/2,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    players.sort(key=lambda p: (p["cx"]-bcx)**2 + (p["cy"]-bcy)**2)
    i, j = 0, 1

    dx = players[i]["cx"] - players[j]["cx"]
    dy = players[i]["cy"] - players[j]["cy"]
    if (dx**2 + dy**2) ** 0.5 > MAX_PAIR_DIST_FRAC * frame_w:
        return None

    # Union of the 2 selected players
    x1 = min(players[i]["x1"], players[j]["x1"])
    y1 = min(players[i]["y1"], players[j]["y1"])
    x2 = max(players[i]["x2"], players[j]["x2"])
    y2 = max(players[i]["y2"], players[j]["y2"])

    # Expand to include ball bounding box if detected
    if ball_boxes:
        bx1, by1, bx2, by2 = ball_boxes[0].xyxy[0].tolist()
        x1 = min(x1, bx1)
        y1 = min(y1, by1)
        x2 = max(x2, bx2)
        y2 = max(y2, by2)

    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)

    # Enforce minimum crop size — expands from centre if too small
    min_w = 0.20 * frame_w
    min_h = 0.30 * frame_h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if (x2 - x1) < min_w:
        x1 = max(0, cx - min_w / 2)
        x2 = min(frame_w, cx + min_w / 2)
    if (y2 - y1) < min_h:
        y1 = max(0, cy - min_h / 2)
        y2 = min(frame_h, cy + min_h / 2)

    return (int(x1), int(y1), int(x2), int(y2))


def _get_crop_from_coords(player_coords, ball_pos, ball_coords_list,
                          frame_h, frame_w, padding=0.15):
    """
    Same logic as get_player_crop but accepts raw coordinate tuples
    instead of YOLO box objects — used in the batched full-video pipeline.
    """
    if ball_pos is None or len(player_coords) < 2:
        return None

    bcx, bcy = ball_pos
    players = []
    for (x1, y1, x2, y2) in player_coords:
        players.append({"cx": (x1+x2)/2, "cy": (y1+y2)/2,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    players.sort(key=lambda p: (p["cx"]-bcx)**2 + (p["cy"]-bcy)**2)
    i, j = 0, 1

    dx = players[i]["cx"] - players[j]["cx"]
    dy = players[i]["cy"] - players[j]["cy"]
    if (dx**2 + dy**2) ** 0.5 > MAX_PAIR_DIST_FRAC * frame_w:
        return None

    x1 = min(players[i]["x1"], players[j]["x1"])
    y1 = min(players[i]["y1"], players[j]["y1"])
    x2 = max(players[i]["x2"], players[j]["x2"])
    y2 = max(players[i]["y2"], players[j]["y2"])

    if ball_coords_list:
        bx1, by1, bx2, by2 = ball_coords_list[0]
        x1, y1 = min(x1, bx1), min(y1, by1)
        x2, y2 = max(x2, bx2), max(y2, by2)

    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)

    min_w, min_h = 0.20 * frame_w, 0.30 * frame_h
    cx, cy = (x1+x2)/2, (y1+y2)/2
    if (x2 - x1) < min_w:
        x1, x2 = max(0, cx - min_w/2), min(frame_w, cx + min_w/2)
    if (y2 - y1) < min_h:
        y1, y2 = max(0, cy - min_h/2), min(frame_h, cy + min_h/2)

    return (int(x1), int(y1), int(x2), int(y2))


def smooth_crops(crop_dict, window=5, polyorder=2, max_gap=5):
    """
    Apply Savitzky-Golay smoothing to crop box coordinates over time.
    Only smooths within contiguous sequences of valid crops — resets across
    gaps larger than max_gap frames to avoid cross-contamination.
    """
    valid = [(fi, box) for fi, box in sorted(crop_dict.items()) if box is not None]
    if len(valid) < window:
        return crop_dict

    window = window if window % 2 == 1 else window + 1

    # Split valid frames into contiguous segments (no gap > max_gap)
    segments = []
    seg = [valid[0]]
    for k in range(1, len(valid)):
        if valid[k][0] - valid[k-1][0] <= max_gap:
            seg.append(valid[k])
        else:
            segments.append(seg)
            seg = [valid[k]]
    segments.append(seg)

    result = dict(crop_dict)
    for seg in segments:
        if len(seg) < window:
            continue  # too short to smooth, leave as-is
        indices = [fi for fi, _ in seg]
        coords = np.array([box for _, box in seg], dtype=float)
        smoothed = np.stack([
            savgol_filter(coords[:, i], window_length=window, polyorder=polyorder)
            for i in range(4)
        ], axis=1)
        for i, fi in enumerate(indices):
            x1, y1, x2, y2 = smoothed[i]
            result[fi] = (int(max(0, x1)), int(max(0, y1)), int(x2), int(y2))

    return result


def pad_to_square(frame, x1, y1, x2, y2):
    """
    Crop region from frame and zero-pad to square, preserving aspect ratio.
    This is what will be fed to DINOv3 instead of a distorted resize.
    Returns None if crop region is empty.
    """
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    dy = (size - h) // 2
    dx = (size - w) // 2
    padded[dy:dy + h, dx:dx + w] = crop
    return padded


import torch
YOLO_BATCH_SIZE = 16 if torch.cuda.is_available() else 1


def run_full_video(video, model, frame_w, frame_h, fps, total_frames, out_dir):
    """
    Three-pass pipeline — no frame storage between passes:
      Pass 1: batched YOLO → lightweight detections + IQR filter
      Pass 2: sequential optical flow + crop computation + Savitzky-Golay
      Pass 3: sequential video read → annotated MP4 write
    """
    out_path = out_dir / f"{video.stem}_annotated.mp4"

    # Stored between passes — coordinates only, not frames (~KB not GB)
    player_coords = {}   # fi -> [(x1,y1,x2,y2), ...]
    ball_coords = {}     # fi -> (x1,y1,x2,y2) or None
    raw_ball_detections = {}  # fi -> (cx,cy) or None

    # ------------------------------------------------------------------
    # Pass 1: batched YOLO — read frames in batches, never store them all
    # ------------------------------------------------------------------
    print(f"Pass 1: batched YOLO (batch={YOLO_BATCH_SIZE}) + storing frames in RAM...")
    cap = cv2.VideoCapture(str(video))
    all_frames = []
    batch, batch_indices = [], []
    fi = 0

    def _flush_batch(batch, batch_indices):
        results = model(batch, conf=BALL_CONF, verbose=False, classes=list(SHOW_CLASSES))
        for idx, res in enumerate(results):
            boxes = res.boxes
            frame_i = batch_indices[idx]
            player_coords[frame_i] = [
                tuple(b.xyxy[0].tolist())
                for b in boxes
                if int(b.cls[0]) in PLAYER_CLASSES and float(b.conf[0]) >= PLAYER_CONF
            ]
            bbs = [b for b in boxes if int(b.cls[0]) == BALL_CLASS]
            if bbs:
                bx1, by1, bx2, by2 = bbs[0].xyxy[0].tolist()
                ball_coords[frame_i] = (bx1, by1, bx2, by2)
                raw_ball_detections[frame_i] = ((bx1+bx2)/2, (by1+by2)/2)
            else:
                ball_coords[frame_i] = None
                raw_ball_detections[frame_i] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        batch.append(frame)
        batch_indices.append(fi)
        if len(batch) == YOLO_BATCH_SIZE:
            _flush_batch(batch, batch_indices)
            batch, batch_indices = [], []
        fi += 1

    if batch:
        _flush_batch(batch, batch_indices)
    cap.release()

    clean_ball = iqr_filter_ball(raw_ball_detections)
    n_det = sum(1 for v in clean_ball.values() if v is not None)
    print(f"  Ball detections after IQR: {n_det}/{fi}")

    # ------------------------------------------------------------------
    # Pass 2: optical flow + crop computation (in-memory, single iteration)
    # ------------------------------------------------------------------
    print("Pass 2: optical flow + crop computation...")
    ball_positions = {}
    raw_crops = {}
    prev_gray = None
    tracker_pt = None
    tracker_valid = False
    tracker_age = 0
    max_tracking_age = int(MAX_TRACKING_AGE_SEC * fps)

    for frame_i, frame in enumerate(all_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if tracker_valid and tracker_age > max_tracking_age:
            tracker_valid = False

        if tracker_valid and prev_gray is not None:
            pt = np.array([[[tracker_pt[0], tracker_pt[1]]]], dtype=np.float32)
            new_pt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pt, None, **LK_PARAMS)
            if status[0][0] == 1:
                nx, ny = float(new_pt[0][0][0]), float(new_pt[0][0][1])
                if 0 <= nx <= frame_w and 0 <= ny <= frame_h:
                    tracker_pt = (nx, ny)
                    tracker_age += 1
                else:
                    tracker_valid = False
            else:
                tracker_valid = False

        if clean_ball[frame_i] is not None:
            tracker_pt = clean_ball[frame_i]
            tracker_valid = True
            tracker_age = 0

        ball_positions[frame_i] = tracker_pt if tracker_valid else None
        bc = ball_coords[frame_i]
        raw_crops[frame_i] = _get_crop_from_coords(
            player_coords[frame_i], ball_positions[frame_i],
            [bc] if bc else [], frame_h, frame_w
        )
        prev_gray = gray

    smoothed_crops = smooth_crops(raw_crops, window=7)

    # ------------------------------------------------------------------
    # Pass 3: write annotated video (in-memory, single iteration)
    # ------------------------------------------------------------------
    print("Pass 3: writing annotated video...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    for frame_i, frame in enumerate(all_frames):
        ball_pos = ball_positions[frame_i]
        crop = smoothed_crops[frame_i]

        for x1, y1, x2, y2 in player_coords[frame_i]:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        if ball_pos:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            color = (0, 255, 0) if raw_ball_detections[frame_i] else (0, 165, 255)
            cv2.circle(frame, (bx, by), 10, color, 2)

        if crop:
            x1, y1, x2, y2 = crop
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        else:
            cv2.putText(frame, "full frame", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"Saved annotated video to {out_path}")


def verify_crop_functions():
    """
    Sanity check: get_player_crop and _get_crop_from_coords must produce
    identical results given equivalent inputs. Uses synthetic data only.
    """
    from types import SimpleNamespace
    import torch

    frame_h, frame_w = 720, 1280

    # Synthetic player boxes: (x1, y1, x2, y2)
    player_data = [
        (300, 200, 380, 420),  # player near ball
        (420, 210, 500, 430),  # player near ball
        (800, 300, 880, 520),  # distant player
    ]
    ball_data = (350, 380, 370, 400)  # ball between the first two players
    ball_pos = ((ball_data[0] + ball_data[2]) / 2, (ball_data[1] + ball_data[3]) / 2)

    # Mock YOLO box objects for get_player_crop
    def make_box(x1, y1, x2, y2, cls, conf=0.9):
        box = SimpleNamespace()
        box.xyxy = [torch.tensor([x1, y1, x2, y2], dtype=torch.float32)]
        box.cls = torch.tensor([cls])
        box.conf = torch.tensor([conf])
        return box

    player_boxes = [make_box(*p, cls=0) for p in player_data]
    ball_boxes = [make_box(*ball_data, cls=1)]

    result_a = get_player_crop(player_boxes, ball_pos, ball_boxes, frame_h, frame_w)
    result_b = _get_crop_from_coords(player_data, ball_pos, [ball_data], frame_h, frame_w)

    print("=== Crop function verification ===")
    print(f"get_player_crop:      {result_a}")
    print(f"_get_crop_from_coords:{result_b}")

    if result_a == result_b:
        print("✓ MATCH — functions are consistent")
    else:
        print("✗ MISMATCH — functions differ, check implementation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None,
                        help="Video filename inside TACDEC dir, or full path.")
    parser.add_argument("--full", action="store_true",
                        help="Process every frame and output an annotated MP4.")
    parser.add_argument("--verify-crop", action="store_true",
                        help="Verify get_player_crop and _get_crop_from_coords are consistent.")
    args = parser.parse_args()

    if args.verify_crop:
        verify_crop_functions()
        return

    if args.video:
        video = Path(args.video)
        if not video.is_absolute():
            video = _video_dir / video
    else:
        video = next(_video_dir.glob("*.mp4"))

    out_dir = ROOT / "yolo_test_frames" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    print(f"Classes: {model.names}")
    print(f"Video:   {video}")
    print(f"Output:  {out_dir}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"ERROR: could not open {video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Frames:  {total_frames} @ {frame_w}x{frame_h} @ {original_fps:.1f}fps")

    if args.full:
        run_full_video(video, model, frame_w, frame_h, original_fps,
                       total_frames, out_dir)
        return

    cap = cv2.VideoCapture(str(video))
    sample_indices = sorted(set(
        int(i * total_frames / NUM_FRAMES) for i in range(NUM_FRAMES)
    ))
    sample_set = set(sample_indices)

    # ------------------------------------------------------------------
    # Pass 1: YOLO on sampled frames — collect raw ball + player detections
    # ------------------------------------------------------------------
    print("\nPass 1: YOLO detection on sampled frames...")
    raw_frames = {}
    player_detections = {}
    ball_box_detections = {}   # frame_idx -> list of ball boxes (for crop union)
    raw_ball_detections = {}   # frame_idx -> (cx, cy) or None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in sample_set:
            results = model(frame, conf=BALL_CONF, verbose=False, classes=list(SHOW_CLASSES))
            boxes = results[0].boxes
            raw_frames[frame_idx] = frame.copy()
            player_detections[frame_idx] = [
                b for b in boxes
                if int(b.cls[0]) in PLAYER_CLASSES and float(b.conf[0]) >= PLAYER_CONF
            ]
            ball_boxes = [b for b in boxes if int(b.cls[0]) == BALL_CLASS]
            ball_box_detections[frame_idx] = ball_boxes
            if ball_boxes:
                bx1, by1, bx2, by2 = ball_boxes[0].xyxy[0].tolist()
                raw_ball_detections[frame_idx] = ((bx1+bx2)/2, (by1+by2)/2)
            else:
                raw_ball_detections[frame_idx] = None
        frame_idx += 1
    cap.release()

    # IQR filter on YOLO detections before using them as tracker resets
    clean_ball = iqr_filter_ball(raw_ball_detections)

    n_detected = sum(1 for v in clean_ball.values() if v is not None)
    print(f"Clean ball detections: {n_detected}/{NUM_FRAMES} sampled frames")

    # ------------------------------------------------------------------
    # Pass 2: Full sequential read — optical flow tracks ball between frames,
    #         clean YOLO detections reset the tracker when available.
    # ------------------------------------------------------------------
    print("\nPass 2: optical flow tracking through all frames...")
    ball_positions = {fi: None for fi in sample_indices}
    ball_source = {}  # frame_idx -> 'yolo' | 'optical_flow' | None

    cap = cv2.VideoCapture(str(video))
    prev_gray = None
    tracker_pt = None
    tracker_valid = False
    tracker_age = 0  # frames since last YOLO reset
    max_tracking_age = int(MAX_TRACKING_AGE_SEC * original_fps)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Expire tracker if it has gone too long without a YOLO reset
        if tracker_valid and tracker_age > max_tracking_age:
            tracker_valid = False

        # Propagate tracker via optical flow from previous frame
        if tracker_valid and prev_gray is not None:
            pt = np.array([[[tracker_pt[0], tracker_pt[1]]]], dtype=np.float32)
            new_pt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pt, None, **LK_PARAMS)
            if status[0][0] == 1:
                nx, ny = float(new_pt[0][0][0]), float(new_pt[0][0][1])
                if 0 <= nx <= frame_w and 0 <= ny <= frame_h:
                    tracker_pt = (nx, ny)
                    tracker_age += 1
                else:
                    tracker_valid = False
            else:
                tracker_valid = False

        # At sampled frames: YOLO detection resets/initialises the tracker
        if frame_idx in sample_set:
            if clean_ball[frame_idx] is not None:
                tracker_pt = clean_ball[frame_idx]
                tracker_valid = True
                tracker_age = 0
                ball_positions[frame_idx] = tracker_pt
                ball_source[frame_idx] = "yolo"
            else:
                ball_positions[frame_idx] = tracker_pt if tracker_valid else None
                ball_source[frame_idx] = "optical_flow" if tracker_valid else None

        prev_gray = gray
        frame_idx += 1

    cap.release()

    n_of = sum(1 for fi in sample_indices if ball_source.get(fi) == "optical_flow")
    print(f"Optical flow filled: {n_of} additional frames")

    # ------------------------------------------------------------------
    # Pass 3: Compute crops → smooth → visualise + save padded crops
    # ------------------------------------------------------------------
    print("\nPass 3: visualizing...")

    # Compute raw crops for all sampled frames
    raw_crops = {}
    for fi in sample_indices:
        raw_crops[fi] = get_player_crop(
            player_detections[fi], ball_positions[fi],
            ball_box_detections.get(fi, []), frame_h, frame_w
        )

    # Savitzky-Golay smoothing across the crop trajectory
    smoothed_crops = smooth_crops(raw_crops)
    n_smoothed = sum(1 for fi in sample_indices
                     if smoothed_crops.get(fi) != raw_crops.get(fi))
    print(f"Savitzky-Golay smoothed {n_smoothed} crop box(es)")

    saved = 0
    for frame_idx in sample_indices:
        frame = raw_frames[frame_idx]
        player_boxes = player_detections[frame_idx]
        ball_pos = ball_positions[frame_idx]
        source = ball_source.get(frame_idx)
        crop = smoothed_crops[frame_idx]

        results = model(frame, conf=BALL_CONF, verbose=False, classes=list(SHOW_CLASSES))
        annotated = results[0].plot()

        # Draw optical-flow predicted ball position (orange circle)
        if ball_pos and source == "optical_flow":
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(annotated, (bx, by), 14, (0, 165, 255), 3)
            cv2.putText(annotated, "optical flow", (bx + 16, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        if crop:
            x1, y1, x2, y2 = crop
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
            label = "yolo" if source == "yolo" else "opt-flow"
            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            method = label

            # Save aspect-ratio-preserving padded crop (what DINOv3 will see)
            padded = pad_to_square(frame, x1, y1, x2, y2)
            if padded is not None:
                resized = cv2.resize(padded, (256, 256))
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}_crop.jpg"), resized)
        else:
            method = "fallback"
            cv2.putText(annotated, "fallback: full frame", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        n_players = len(player_boxes)
        n_ball = 1 if raw_ball_detections[frame_idx] else 0
        print(f"Frame {frame_idx:5d}: {n_players} players, {n_ball} ball(YOLO), source={source}, method={method}")

        cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.jpg"), annotated)
        saved += 1

    print(f"\nSaved {saved} frames to {out_dir}/")
    print("Cyan box = smoothed crop | Orange circle = optical flow | *_crop.jpg = what DINOv3 sees")


if __name__ == "__main__":
    main()
