"""
Run the trained DINOv3 linear spatial probe on a *single* unlabelled video
(e.g. one half of a SoccerNet game) and dump the fired tackle events with
timestamps.

This is the spatial-probe counterpart to ``predict_soccernet.py`` (attentive).
It mirrors the event-detection path used by ``eval_spatial.py`` Track 3
(SoccerNet Average-mAP), stripped of label-dependent code so it can run on a
brand-new clip without ground truth. It:

  1. loads the spatial probe checkpoint from a training ``--run-dir`` (same
     layout as eval_spatial.py: model.pt + config.json),
  2. runs per-frame inference on pre-extracted CLS features (the
     ``*_<backbone_id>_<fps>fps_features.npz`` produced by the SoccerNet
     extractor),
  3. stride-subsamples the per-frame logits to 5 FPS (parity with the
     attentive probes and with eval_spatial Track 3),
  4. segment-decodes events with the identical ``postprocess_clip`` call
     eval_spatial uses to feed Average-mAP,
  5. writes the detections to a human-readable .txt and two .csv files so the
     timestamps can be checked by hand against the broadcast.

No metric is computed (there is no tackle ground truth for SoccerNet).

Usage (see soccernet_experiment/run_predict_spatial.sh):
  uv run python src/predict_soccernet_spatial.py \
      --run-dir /cluster/.../results/dinov3_linear_spatial/<RUN_NAME> \
      --features-dir /cluster/.../soccernet_thesis_experiment/features \
      --min-confidence 0.5 \
      --out-dir /cluster/.../soccernet_thesis_experiment/predictions
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

# Run as `python src/predict_soccernet_spatial.py`, so sys.path[0] == src/.
from data.labels import CLASS_NAMES
from eval_spatial import infer_clip, load_clip_cls
from models.dinov3.linear_probe import DINOv3LinearProbe
from postprocess import postprocess_clip
from utils import set_seed


def _fmt_mmss(t: float) -> str:
    """Seconds-from-start -> 'MM:SS.s' for eyeballing against the video clock."""
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:04.1f}"


def _auto_detect_video_id(features_dir: Path, backbone_id: str, fps: float) -> str:
    """Recover the video_id from the single matching .npz in features_dir.

    Mirrors `_feature_path` in eval_spatial.py:
        {video_id}_{backbone_id}_{fps}fps_features.npz
    """
    suffix = f"_{backbone_id}_{fps}fps_features.npz"
    matches = sorted(features_dir.glob(f"*{suffix}"))
    if not matches:
        raise SystemExit(
            f"No CLS feature file matching *{suffix} in {features_dir}. "
            "Run the SoccerNet extractor first."
        )
    if len(matches) > 1:
        names = "\n  ".join(p.name for p in matches)
        raise SystemExit(
            f"Multiple candidates in {features_dir}; pass --video-id to disambiguate:\n  {names}"
        )
    return matches[0].name[: -len(suffix)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Training run directory (contains model.pt and config.json).")
    ap.add_argument("--features-dir", required=True, type=Path,
                    help="Directory holding the extracted CLS .npz for this clip.")
    ap.add_argument("--video-id", default=None,
                    help="Feature-file stem (the .mp4 filename without extension). "
                         "If omitted, auto-detected from the single CLS .npz in "
                         "--features-dir matching the run's backbone_id + fps.")
    ap.add_argument("--extraction-fps", type=float, default=None,
                    help="Override the run's extraction FPS when locating the "
                         "feature file. Use this when SoccerNet CLS was extracted "
                         "at a lower FPS than training (e.g. 5.0 vs 25.0). The "
                         "linear probe has no temporal state, so this is equivalent "
                         "to the stride-subsampled path eval_spatial uses for the "
                         "5-FPS Average-mAP — just without the resampling step.")
    # Event-decode knobs: defaults mirror eval_spatial.py.
    ap.add_argument("--min-segment-frames", type=int, default=2,
                    help="Minimum predicted-class run length to count as a segment.")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="Confidence threshold for the .txt 'fired' summary. "
                         "The full segment list (threshold 0) is always written "
                         "to *_events_all.csv as well.")
    ap.add_argument("--nms", action="store_true",
                    help="Apply cross-class NMS on top of segment detection "
                         "(off by default to match eval_spatial).")
    ap.add_argument("--nms-distance-sec", type=float, default=0.5,
                    help="Min temporal distance between cross-class detections "
                         "when --nms is set.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Directory for the events .txt / .csv outputs.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load run config + probe (same path as eval_spatial.py) --------------
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "config.json").read_text())
    backbone_id = cfg["backbone_id"]
    train_fps = float(cfg["fps"])
    fps = float(args.extraction_fps) if args.extraction_fps is not None else train_fps
    feature_dim = int(cfg["feature_dim"])
    if fps != train_fps:
        print(f"NOTE: probe was trained on {train_fps}-FPS extraction; reading "
              f"{fps}-FPS features instead. The probe is frame-independent so "
              "this is fine, but make sure that's what you intended.")

    model = DINOv3LinearProbe(embed_dim=feature_dim, num_classes=3).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded probe from {run_dir / 'model.pt'}")
    print(f"  backbone_id={backbone_id}  feature_dim={feature_dim}  extraction_fps={fps}")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Resolve feature file ------------------------------------------------
    features_dir = args.features_dir.resolve()
    if args.video_id is None:
        args.video_id = _auto_detect_video_id(features_dir, backbone_id, fps)
        print(f"Auto-detected video_id: {args.video_id}")
    feat_path = features_dir / f"{args.video_id}_{backbone_id}_{fps}fps_features.npz"
    if not feat_path.exists():
        raise SystemExit(f"Feature file not found: {feat_path}")

    cls = load_clip_cls(feat_path)
    n_source = int(cls.shape[0])
    duration_s = n_source / fps
    print(f"Clip '{args.video_id}': {n_source} CLS rows @ {fps} FPS "
          f"(~{duration_s/60:.1f} min)")

    # --- Per-frame inference, then stride-subsample to 5 FPS for decoding ----
    # Matches eval_spatial.py: events are scored at 5 FPS regardless of the
    # extraction FPS, so the cadence is comparable with the attentive probes.
    print("Running per-frame inference...")
    logits = infer_clip(model, cls, device)

    eval_fps = 5.0
    stride = max(1, int(round(fps / eval_fps)))
    logits_eval = logits[::stride]
    T_eval = logits_eval.shape[0]
    mask_eval = np.ones(T_eval, dtype=np.float32)
    print(f"Event decoding @ {eval_fps} FPS (stride={stride}): {T_eval} frames")

    # --- Segment decoding (identical call to eval_spatial.py Track 3) --------
    # Always decode at threshold 0 so we keep the full segment list; the
    # 'fired' subset is a confidence filter on top.
    detections = postprocess_clip(
        logits=logits_eval,
        mask=mask_eval,
        num_classes=3,
        fps=eval_fps,
        method="segment",
        labeling_mode="interval",
        min_segment_frames=args.min_segment_frames,
        min_confidence=0.0,
        nms=args.nms,
        min_distance_sec=args.nms_distance_sec,
    )

    for d in detections:
        d["timestamp_sec"] = float(d["frame"]) / eval_fps
        d["class_name"] = CLASS_NAMES[int(d["class"])]
    detections.sort(key=lambda d: d["timestamp_sec"])
    fired = [d for d in detections if d["confidence"] >= args.min_confidence]

    n_live = sum(d["class_name"] == "tackle-live" for d in fired)
    n_replay = sum(d["class_name"] == "tackle-replay" for d in fired)
    print(f"\nFired events (conf >= {args.min_confidence}): {len(fired)} "
          f"(tackle-live={n_live}, tackle-replay={n_replay}); "
          f"{len(detections)} total segments.")

    # --- Write outputs -------------------------------------------------------
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.video_id

    txt_path = out_dir / f"{stem}_spatial_events.txt"
    with open(txt_path, "w") as f:
        f.write(f"# Fired tackle events for '{stem}' (DINOv3 linear spatial probe)\n")
        f.write(f"# run-dir            : {run_dir}\n")
        f.write(f"# min_confidence     : {args.min_confidence}\n")
        f.write(f"# min_segment_frames : {args.min_segment_frames}\n")
        f.write(f"# nms                : {args.nms} (dist={args.nms_distance_sec}s)\n")
        f.write(f"# extraction / eval  : {fps} / {eval_fps} FPS  (stride={stride})\n")
        f.write(f"# clip duration      : ~{duration_s/60:.1f} min ({n_source} src rows)\n")
        f.write(f"# fired / total      : {len(fired)} / {len(detections)} segments\n")
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
                "confidence", "frame_idx",
            ])
            w.writeheader()
            for d in rows:
                w.writerow({
                    "timestamp_sec": round(d["timestamp_sec"], 3),
                    "mm_ss": _fmt_mmss(d["timestamp_sec"]),
                    "class_id": int(d["class"]),
                    "class_name": d["class_name"],
                    "confidence": round(float(d["confidence"]), 4),
                    "frame_idx": int(d["frame"]),
                })

    csv_fired = out_dir / f"{stem}_spatial_events.csv"
    csv_all = out_dir / f"{stem}_spatial_events_all.csv"
    _write_csv(csv_fired, fired)
    _write_csv(csv_all, detections)
    print(f"  wrote {csv_fired}")
    print(f"  wrote {csv_all}  (every segment, re-threshold without re-running)")

    (out_dir / f"{stem}_spatial_predict_summary.json").write_text(json.dumps({
        "video_id": stem,
        "run_dir": str(run_dir),
        "backbone_id": backbone_id,
        "feature_dim": feature_dim,
        "extraction_fps": fps,
        "eval_fps": eval_fps,
        "stride": stride,
        "n_source_rows": n_source,
        "n_eval_frames": T_eval,
        "duration_min": duration_s / 60.0,
        "min_confidence": args.min_confidence,
        "min_segment_frames": args.min_segment_frames,
        "nms": bool(args.nms),
        "nms_distance_sec": args.nms_distance_sec,
        "n_fired": len(fired),
        "n_total_segments": len(detections),
        "n_fired_tackle_live": n_live,
        "n_fired_tackle_replay": n_replay,
    }, indent=2, default=float))
    print("\nDone.")


if __name__ == "__main__":
    main()
