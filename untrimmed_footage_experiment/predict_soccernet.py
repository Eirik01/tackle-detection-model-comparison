"""
Run the trained DINOv3 attentive probe on a *single* unlabelled video (e.g. one
half of a SoccerNet game) and dump the fired tackle events with timestamps.

This is the inference half of ``eval_temporal.py`` Track 3a, stripped of every
dependency on TACDEC labels / game splits so it can run on a brand-new clip that
has no ground truth. It:

  1. loads the chosen attentive-probe checkpoint (same finder/rebuilder as eval),
  2. runs a stride-1 sliding window over every valid anchor of the clip, using
     the identical DINOv3 dense-token loader and window protocol as training,
  3. peak-detects events with the identical ``postprocess_clip`` call eval uses
     to feed the SoccerNet Average-mAP metric,
  4. writes the detections to a human-readable .txt and two .csv files so the
     timestamps can be checked by hand against the broadcast.

No metric is computed (there is no tackle ground truth for SoccerNet). The peak
list is exactly what the Average-mAP path would score, so a detection here is a
detection there.

Usage (see untrimmed_footage_experiment/run_predict.sh):
  uv run python untrimmed_footage_experiment/predict_soccernet.py \
      --features-dir /cluster/.../soccernet_thesis_experiment/features \
      --model-suffix centred_v1 \
      --min-confidence 0.5 \
      --out-dir /cluster/.../soccernet_thesis_experiment/predictions

The video_id is auto-detected from the single dense feature file in
--features-dir (the experiment runs on one clip), so it doesn't need to be
passed explicitly. Pass --video-id only if there are multiple dense files in
the dir and you want to pick one.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# This experiment lives outside src/; put src/ on the path so the flat imports
# below resolve the same way they do for eval_temporal.py (matches the shim used
# across analysis/ and visualization/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.temporal_loaders import CLASS_NAMES, DINOv3DenseLoader
from eval_temporal import find_checkpoint, rebuild_probe
from postprocess import postprocess_clip
from utils import set_seed
from window_protocol import valid_anchor_range


def _fmt_mmss(t: float) -> str:
    """Seconds-from-start -> 'MM:SS.s' for eyeballing against the video clock."""
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:04.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features-dir", required=True,
                    help="Directory holding the extracted dense .npy for this clip.")
    ap.add_argument("--video-id", default=None,
                    help="Feature-file stem (the .mp4 filename without extension). "
                         "If omitted, auto-detected from the single dense .npy in "
                         "--features-dir matching the configured fps + padding.")
    ap.add_argument("--model-suffix", default="centred_v1",
                    help="Checkpoint suffix: loads best_attn_dinov3_l_<suffix>.pth "
                         "from TACDEC_MODELS/dinov3_l/. Set to your best run.")
    ap.add_argument("--backbone-size", default="large",
                    choices=["base", "large"])
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--window-size", type=int, default=10, help="W (target-FPS frames).")
    ap.add_argument("--fps", type=float, default=5.0, help="Target FPS (anchor rate).")
    ap.add_argument("--source-fps", type=float, default=25.0,
                    help="FPS embedded in the on-disk feature filename "
                         "(extract_features.py --fps). Must match extraction.")
    ap.add_argument("--padding-mode", choices=["center_crop", "reflect"],
                    default="reflect",
                    help="Extraction flavour; 'reflect' loads *_reflect_dense_* "
                         "files (the active DINOv3 temporal protocol).")
    # Peak-detection knobs: defaults mirror eval_temporal.py / postprocess.py.
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="Confidence threshold for an event to count as 'fired' "
                         "in the .txt summary. The full peak list (threshold 0) "
                         "is always written to *_events_all.csv as well.")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian smoothing sigma in frames (peak detection).")
    ap.add_argument("--min-distance-sec", type=float, default=0.5,
                    help="Per-class minimum spacing between consecutive peaks.")
    ap.add_argument("--max-duration-sec", type=float, default=None,
                    help="If set, only score anchors in the first N seconds "
                         "of the clip (useful for a quick qualitative check on "
                         "long broadcasts). Default: score the whole valid range.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True,
                    help="Directory for the events .txt / .csv outputs.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_id = f"dinov3_{args.backbone_size[0]}"

    # --- Load probe (identical path to eval_temporal.py) ---------------------
    ckpt_path = find_checkpoint(backbone_id, args.model_suffix)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    train_window = int(ckpt.get("window_size", args.window_size))
    train_fps = float(ckpt.get("fps", args.fps))
    if train_window != args.window_size or train_fps != args.fps:
        print(f"NOTE: checkpoint trained at W={train_window}@{train_fps} FPS; "
              f"running at W={args.window_size}@{args.fps} FPS. Match them unless "
              "the mismatch is intentional.")
    model, backbone = rebuild_probe(ckpt)
    model = model.to(device)
    if backbone != "dinov3":
        raise SystemExit(f"This predictor is DINOv3-only; checkpoint is {backbone!r}.")
    print(f"Probe loaded ({backbone}), params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # --- Dense-token loader (identical protocol to training/eval) ------------
    src_stride = max(1, int(round(args.source_fps / args.fps)))
    dense_tag = "reflect" if args.padding_mode == "reflect" else ""

    # Auto-detect the clip if --video-id was omitted. DINOv3DenseLoader names
    # files as {video_id}_dinov3_l_{source_fps}fps[_{dense_tag}]_dense_features.npy
    # — invert that to recover video_id when there's exactly one match.
    features_dir = Path(args.features_dir)
    if args.video_id is None:
        tag = f"_{dense_tag}" if dense_tag else ""
        suffix = f"_dinov3_l_{args.source_fps}fps{tag}_dense_features.npy"
        matches = sorted(features_dir.glob(f"*{suffix}"))
        if not matches:
            raise SystemExit(f"No dense feature file matching *{suffix} in "
                             f"{features_dir}. Run extraction first.")
        if len(matches) > 1:
            names = "\n  ".join(p.name for p in matches)
            raise SystemExit(f"Multiple candidates in {features_dir}; pass "
                             f"--video-id to disambiguate:\n  {names}")
        args.video_id = matches[0].name[: -len(suffix)]
        print(f"Auto-detected video_id: {args.video_id}  ({matches[0].name})")

    loader = DINOv3DenseLoader(
        args.features_dir, args.fps, args.window_size,
        source_fps=args.source_fps, max_cached=2, dense_tag=dense_tag,
    )

    # Clip length straight from the feature file (no labels needed). _open_video
    # mmaps the array; shape[0] is the number of source (25-FPS) rows on disk.
    arr = loader._open_video(args.video_id)
    n_source = int(arr.shape[0])
    valid_lo, valid_hi = valid_anchor_range(
        video_length=n_source,
        anchor_stride=src_stride,
        intra_window_stride=src_stride,
        window_length=args.window_size,
    )
    if valid_hi < valid_lo:
        raise SystemExit(
            f"Clip too short: {n_source} source rows can't fit one W={args.window_size} "
            f"window at source_fps={args.source_fps}/target_fps={args.fps}."
        )
    duration_s = n_source / args.source_fps
    if args.max_duration_sec is not None:
        # Clamp valid_hi to the latest anchor whose source-frame index is
        # still within the requested duration. Anchors are at source-FPS
        # indices, so the latest in-bounds anchor is floor(max_sec * src_fps
        # / anchor_stride) * anchor_stride; equivalent to clamping the
        # anchor index by max_sec * src_fps.
        max_source_idx = int(args.max_duration_sec * args.source_fps)
        clamped_hi = min(valid_hi, max_source_idx)
        if clamped_hi < valid_lo:
            raise SystemExit(
                f"--max-duration-sec={args.max_duration_sec} too small: no valid "
                f"anchor fits in the requested window (valid_lo={valid_lo})."
            )
        if clamped_hi < valid_hi:
            print(f"Clamping anchor range to first {args.max_duration_sec:.0f}s: "
                  f"valid_hi {valid_hi} -> {clamped_hi}.")
            valid_hi = clamped_hi
    n_anchors = valid_hi - valid_lo + 1
    print(f"Clip '{args.video_id}': {n_source} source rows (~{duration_s/60:.1f} min), "
          f"valid anchors [{valid_lo}, {valid_hi}] = {n_anchors} windows.")

    # --- Stride-1 sliding-window inference -----------------------------------
    # logits row j corresponds to absolute anchor (valid_lo + j); building it
    # contiguous from 0 keeps postprocess_clip's `probs[:seq_len]` honest, and
    # we add valid_lo back when converting peak frame -> timestamp.
    logits = np.zeros((n_anchors, args.num_classes), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for j, anchor in enumerate(range(valid_lo, valid_hi + 1)):
            feats = loader.get_feature(args.video_id, anchor)  # [W*256, D]
            x = torch.from_numpy(feats).unsqueeze(0).to(device, non_blocking=True)
            logits[j] = model(x).squeeze(0).cpu().numpy()
            if j % 1000 == 0:
                print(f"  ...{j}/{n_anchors} windows", flush=True)
    mask = np.ones(n_anchors, dtype=np.float32)

    # --- Peak detection (identical call to eval_temporal Track 3a) -----------
    # Run once at threshold 0 to get the full peak list; the 'fired' subset is a
    # confidence filter on top of that, so we never re-run inference.
    detections = postprocess_clip(
        logits=logits,
        mask=mask,
        num_classes=args.num_classes,
        fps=args.fps,
        method="peak",
        labeling_mode="anchor",
        sigma=args.sigma,
        min_confidence=0.0,
        min_distance_sec=args.min_distance_sec,
        nms=False,
    )

    # Map peak frame (0-based within valid range) back to absolute time.
    for d in detections:
        d["anchor"] = valid_lo + int(d["frame"])
        d["timestamp_sec"] = d["anchor"] / args.fps
        d["class_name"] = CLASS_NAMES[int(d["class"])]
    detections.sort(key=lambda d: d["timestamp_sec"])
    fired = [d for d in detections if d["confidence"] >= args.min_confidence]

    n_live = sum(d["class_name"] == "tackle-live" for d in fired)
    n_replay = sum(d["class_name"] == "tackle-replay" for d in fired)
    print(f"\nFired events (conf >= {args.min_confidence}): {len(fired)} "
          f"(tackle-live={n_live}, tackle-replay={n_replay}); "
          f"{len(detections)} total peaks.")

    # --- Write outputs -------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video_id

    txt_path = out_dir / f"{stem}_dinov3_temporal_events.txt"
    with open(txt_path, "w") as f:
        f.write(f"# Fired tackle events for '{stem}'\n")
        f.write(f"# checkpoint        : {ckpt_path}\n")
        f.write(f"# min_confidence    : {args.min_confidence}\n")
        f.write(f"# sigma / min_dist  : {args.sigma} / {args.min_distance_sec}s\n")
        f.write(f"# W / target / src  : {args.window_size} / {args.fps} / {args.source_fps} FPS\n")
        f.write(f"# clip duration     : ~{duration_s/60:.1f} min ({n_source} src rows)\n")
        f.write(f"# fired / total     : {len(fired)} / {len(detections)} peaks\n")
        f.write("#\n")
        f.write("# timestamp (mm:ss) |   sec   | event          | confidence\n")
        f.write("# " + "-" * 58 + "\n")
        for d in fired:
            f.write(f"{_fmt_mmss(d['timestamp_sec']):>9} | "
                    f"{d['timestamp_sec']:7.2f} | "
                    f"{d['class_name']:<14} | {d['confidence']:.3f}\n")
    print(f"  wrote {txt_path}")

    def _write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "timestamp_sec", "mm_ss", "class_id", "class_name",
                "confidence", "anchor_idx",
            ])
            w.writeheader()
            for d in rows:
                w.writerow({
                    "timestamp_sec": round(d["timestamp_sec"], 3),
                    "mm_ss": _fmt_mmss(d["timestamp_sec"]),
                    "class_id": int(d["class"]),
                    "class_name": d["class_name"],
                    "confidence": round(float(d["confidence"]), 4),
                    "anchor_idx": d["anchor"],
                })

    csv_fired = out_dir / f"{stem}_dinov3_temporal_events.csv"
    csv_all = out_dir / f"{stem}_dinov3_temporal_events_all.csv"
    _write_csv(csv_fired, fired)
    _write_csv(csv_all, detections)
    print(f"  wrote {csv_fired}")
    print(f"  wrote {csv_all}  (every peak, re-threshold without re-running)")

    # Small machine-readable run summary.
    (out_dir / f"{stem}_dinov3_temporal_predict_summary.json").write_text(json.dumps({
        "video_id": stem,
        "checkpoint": str(ckpt_path),
        "n_source_rows": n_source,
        "duration_min": duration_s / 60.0,
        "valid_anchor_range": [valid_lo, valid_hi],
        "min_confidence": args.min_confidence,
        "sigma": args.sigma,
        "min_distance_sec": args.min_distance_sec,
        "window_size": args.window_size,
        "fps": args.fps,
        "source_fps": args.source_fps,
        "n_fired": len(fired),
        "n_total_peaks": len(detections),
        "n_fired_tackle_live": n_live,
        "n_fired_tackle_replay": n_replay,
    }, indent=2, default=float))
    print("\nDone.")


if __name__ == "__main__":
    main()
