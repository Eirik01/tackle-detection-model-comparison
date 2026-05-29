"""
YOLO-based crop coordinate extractor for TACDEC videos.

Runs YOLO + optical flow + IQR filtering to compute per-frame crop boxes.
Saves lightweight .npz files (coordinates only) that both DINOv3 and V-JEPA2
extractors can load to apply spatially-grounded crops before backbone inference.

Crop format: int array [N_frames, 4] with (x1, y1, x2, y2).
Fallback frames (no ball / <2 players) stored as (-1, -1, -1, -1).
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

PLAYER_CONF = 0.5
BALL_CONF = 0.5
PLAYER_CLASS = 0
BALL_CLASS = 1
DETECT_CLASSES = [PLAYER_CLASS, BALL_CLASS]

MAX_PAIR_DIST_FRAC = 0.25
CROP_PADDING = 0.15
MIN_CROP_W_FRAC = 0.20
MIN_CROP_H_FRAC = 0.30
MAX_TRACKING_AGE_SEC = 1.0

YOLO_BATCH_SIZE = 16 if torch.cuda.is_available() else 1

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
)

FALLBACK = (-1, -1, -1, -1)


def _iqr_filter_ball(detections):
    """Remove spatially anomalous ball detections using IQR."""
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
    return result


def _compute_crop(player_coords, ball_pos, ball_box, frame_h, frame_w):
    """
    Compute crop box as union of 2 players nearest the ball + ball box.
    Returns FALLBACK if conditions not met.

    Args:
        player_coords: list of (x1, y1, x2, y2) for detected players
        ball_pos: (cx, cy) of ball or None
        ball_box: (x1, y1, x2, y2) of ball bounding box or None
        frame_h, frame_w: frame dimensions
    """
    if ball_pos is None or len(player_coords) < 2:
        return FALLBACK

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
        return FALLBACK

    x1 = min(players[i]["x1"], players[j]["x1"])
    y1 = min(players[i]["y1"], players[j]["y1"])
    x2 = max(players[i]["x2"], players[j]["x2"])
    y2 = max(players[i]["y2"], players[j]["y2"])

    if ball_box is not None:
        bx1, by1, bx2, by2 = ball_box
        x1, y1 = min(x1, bx1), min(y1, by1)
        x2, y2 = max(x2, bx2), max(y2, by2)

    pad_x = (x2 - x1) * CROP_PADDING
    pad_y = (y2 - y1) * CROP_PADDING
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)

    min_w = MIN_CROP_W_FRAC * frame_w
    min_h = MIN_CROP_H_FRAC * frame_h
    cx, cy = (x1+x2)/2, (y1+y2)/2
    if (x2 - x1) < min_w:
        x1, x2 = max(0, cx - min_w/2), min(frame_w, cx + min_w/2)
    if (y2 - y1) < min_h:
        y1, y2 = max(0, cy - min_h/2), min(frame_h, cy + min_h/2)

    return (int(x1), int(y1), int(x2), int(y2))


def extract_crops_for_video(video_path, yolo_model, fps, output_path):
    """
    Extract per-frame crop coordinates for a single video.

    Strategy:
    - Pass 1: Read all frames, run batched YOLO on sampled frames, store
              player + ball detections. Apply IQR to ball detections.
    - Pass 2: Sequential optical flow through all frames to track ball
              between sampled frames. Compute crop per sampled frame.
    - Save crops array [N_sampled_frames, 4] as .npz.

    Args:
        video_path (Path): Input video file
        yolo_model: Loaded YOLO model
        fps (float): Target extraction FPS (must match backbone extraction FPS)
        output_path (Path): Where to save the .npz crop file
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: could not open {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    skip_interval = max(1, round(original_fps / fps)) if fps else 1
    sampled_indices = set(range(0, total_frames, skip_interval))
    sorted_sampled = sorted(sampled_indices)

    # ------------------------------------------------------------------
    # Pass 1: Batched YOLO on sampled frames + store all frames for Pass 2
    # ------------------------------------------------------------------
    player_coords = {}   # fi -> [(x1,y1,x2,y2), ...]
    ball_coords = {}     # fi -> (x1,y1,x2,y2) or None
    raw_ball_detections = {}  # fi -> (cx,cy) or None
    all_frames = []

    cap = cv2.VideoCapture(str(video_path))
    batch, batch_indices = [], []

    def _flush(batch, batch_indices):
        results = yolo_model(batch, conf=BALL_CONF, verbose=False,
                             classes=DETECT_CLASSES)
        for idx, res in enumerate(results):
            boxes = res.boxes
            fi = batch_indices[idx]
            player_coords[fi] = [
                tuple(b.xyxy[0].tolist())
                for b in boxes
                if int(b.cls[0]) == PLAYER_CLASS and float(b.conf[0]) >= PLAYER_CONF
            ]
            bbs = [b for b in boxes if int(b.cls[0]) == BALL_CLASS]
            if bbs:
                bx1, by1, bx2, by2 = bbs[0].xyxy[0].tolist()
                ball_coords[fi] = (bx1, by1, bx2, by2)
                raw_ball_detections[fi] = ((bx1+bx2)/2, (by1+by2)/2)
            else:
                ball_coords[fi] = None
                raw_ball_detections[fi] = None

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        if fi in sampled_indices:
            batch.append(frame)
            batch_indices.append(fi)
            if len(batch) == YOLO_BATCH_SIZE:
                _flush(batch, batch_indices)
                batch, batch_indices = [], []
        fi += 1

    if batch:
        _flush(batch, batch_indices)
    cap.release()

    clean_ball = _iqr_filter_ball(raw_ball_detections)

    # ------------------------------------------------------------------
    # Pass 2: Optical flow through every frame, compute crop at sampled frames
    # ------------------------------------------------------------------
    max_tracking_age = int(MAX_TRACKING_AGE_SEC * original_fps)
    prev_gray = None
    tracker_pt = None
    tracker_valid = False
    tracker_age = 0

    crops_dict = {}  # fi -> (x1,y1,x2,y2) or FALLBACK

    for frame_i, frame in enumerate(all_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if tracker_valid and tracker_age > max_tracking_age:
            tracker_valid = False

        if tracker_valid and prev_gray is not None:
            pt = np.array([[[tracker_pt[0], tracker_pt[1]]]], dtype=np.float32)
            new_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, pt, None, **LK_PARAMS
            )
            if status[0][0] == 1:
                nx, ny = float(new_pt[0][0][0]), float(new_pt[0][0][1])
                if 0 <= nx <= frame_w and 0 <= ny <= frame_h:
                    tracker_pt = (nx, ny)
                    tracker_age += 1
                else:
                    tracker_valid = False
            else:
                tracker_valid = False

        if frame_i in sampled_indices:
            if clean_ball.get(frame_i) is not None:
                tracker_pt = clean_ball[frame_i]
                tracker_valid = True
                tracker_age = 0

            ball_pos = tracker_pt if tracker_valid else None
            bc = ball_coords.get(frame_i)
            crops_dict[frame_i] = _compute_crop(
                player_coords.get(frame_i, []),
                ball_pos, bc, frame_h, frame_w
            )

        prev_gray = gray

    # Save as ordered array matching sampled frame indices
    crops_array = np.array(
        [crops_dict.get(fi, FALLBACK) for fi in sorted_sampled],
        dtype=np.int32
    )

    n_valid = int((crops_array[:, 0] >= 0).sum())
    n_total = len(crops_array)
    print(f"  Crops: {n_valid}/{n_total} frames have valid crop ({n_valid/n_total*100:.1f}%)")

    np.savez_compressed(output_path, crops=crops_array)


class YOLOCropExtractor:
    """
    Extracts and saves YOLO crop coordinates for all TACDEC videos.
    Output files are consumed by DINOv3Extractor and VJEPA2Extractor.
    """

    def __init__(self, input_dir, output_dir, yolo_weights, device="cuda"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.yolo = YOLO(str(yolo_weights))
        if device == "cuda" and torch.cuda.is_available():
            self.yolo.to("cuda")
        print(f"YOLO loaded: {yolo_weights}")
        print(f"Classes: {self.yolo.names}")

    def extract(self, fps, override=False, start_idx=None, end_idx=None):
        videos = sorted(self.input_dir.glob("*.mp4"))
        if start_idx is not None or end_idx is not None:
            videos = videos[start_idx:end_idx]

        print(f"Processing {len(videos)} videos at {fps}fps...")

        for video_path in tqdm(videos, desc="Extracting crops"):
            output_path = self.output_dir / f"{video_path.stem}_yolo_{fps}fps_crops.npz"
            if output_path.exists() and not override:
                continue
            extract_crops_for_video(video_path, self.yolo, fps, output_path)

    @staticmethod
    def get_crop_path(crop_dir, video_stem, fps):
        """Return the expected crop file path for a given video."""
        return Path(crop_dir) / f"{video_stem}_yolo_{fps}fps_crops.npz"

    @staticmethod
    def load_crops(crop_path):
        """
        Load crop coordinates from .npz file.
        Returns int32 array [N_frames, 4]. Fallback frames have value -1.
        """
        return np.load(crop_path)["crops"]
