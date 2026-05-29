"""
Run the trained V-JEPA 2 attentive probe on a *single* unlabelled video
(e.g. one half of a SoccerNet game) and dump the fired tackle events with
timestamps.

V-JEPA 2 counterpart to ``predict_soccernet.py`` (DINOv3 attentive). It:

  1. loads the chosen V-JEPA 2 attentive-probe checkpoint via the same
     find_checkpoint / rebuild_probe path eval_temporal uses,
  2. iterates the valid anchor range emitted by the V-JEPA 2 dense extractor
     (the .npz is self-describing — it carries valid_lo/valid_hi/anchor_stride
     /window_length so no external config is needed),
  3. peak-detects events with the identical ``postprocess_clip`` call eval
     uses to feed Average-mAP,
  4. writes the detections to a human-readable .txt and two .csv files.

No metric is computed (there is no tackle ground truth for SoccerNet).

Usage (see untrimmed_footage_experiment/run_predict_vjepa2.sh):
  uv run python untrimmed_footage_experiment/predict_soccernet_vjepa2.py \
      --features-dir /cluster/.../soccernet_thesis_experiment/features \
      --model-suffix v_attn_v1 \
      --window-size 10 \
      --fps 5.0 \
      --padding-mode reflect \
      --min-confidence 0.5 \
      --out-dir /cluster/.../soccernet_thesis_experiment/predictions
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
# below resolve (matches the shim used across analysis/ and visualization/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.temporal_loaders import CLASS_NAMES, VJEPA2DenseLoader
from eval_temporal import find_checkpoint, rebuild_probe
from postprocess import postprocess_clip
from utils import set_seed


def _fmt_mmss(t: float) -> str:
    m = int(t // 60)
    s = t - 60 * m
    return f"{m:02d}:{s:04.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features-dir", required=True,
                    help="Directory holding the extracted V-JEPA 2 dense .npz for this clip.")
    ap.add_argument("--video-id", default=None,
                    help="Feature-file stem (the .mp4 filename without extension). "
                         "If omitted, auto-detected from the single dense .npz in "
                         "--features-dir matching the configured fps + W + padding.")
    ap.add_argument("--model-suffix", required=True,
                    help="Checkpoint suffix: loads best_attn_vjepa2_l_<suffix>.pth "
                         "from TACDEC_MODELS/vjepa2_l/. Set to your best run.")
    ap.add_argument("--backbone-size", default="large", choices=["base", "large"])
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--window-size", type=int, default=10,
                    help="W (V-JEPA 2 raw-frame window length, must match extraction).")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="Target FPS (anchor rate). Must match extraction.")
    ap.add_argument("--padding-mode", choices=["center_crop", "reflect"],
                    default="reflect",
                    help="Extraction flavour; 'reflect' loads "
                         "*_reflect_dense_w{W}*.npz files (the active V-JEPA 2 "
                         "temporal protocol).")
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
    backbone_id = f"vjepa2_{args.backbone_size[0]}"

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
    if backbone != "vjepa2":
        raise SystemExit(f"This predictor is V-JEPA 2-only; checkpoint is {backbone!r}.")
    print(f"Probe loaded ({backbone}), params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # --- Dense loader + auto-detect video_id ---------------------------------
    dense_tag = "reflect" if args.padding_mode == "reflect" else ""
    features_dir = Path(args.features_dir)
    if args.video_id is None:
        tag = f"_{dense_tag}" if dense_tag else ""
        # Streaming-layout files end in .npy with a .meta.npz sidecar; legacy
        # files are a single .npz. Prefer .npy when both exist.
        stem_glob = f"*_{backbone_id}_{args.fps}fps{tag}_dense_w{args.window_size}"
        matches = sorted(features_dir.glob(f"{stem_glob}*.npy"))
        if not matches:
            matches = [p for p in sorted(features_dir.glob(f"{stem_glob}*.npz"))
                       if not p.name.endswith(".meta.npz")]
        if not matches:
            raise SystemExit(f"No V-JEPA 2 dense file matching {stem_glob}*.npy "
                             f"(or legacy .npz) in {features_dir}. Run extraction first.")
        if len(matches) > 1:
            names = "\n  ".join(p.name for p in matches)
            raise SystemExit(f"Multiple candidates in {features_dir}; pass "
                             f"--video-id to disambiguate:\n  {names}")
        # Strip everything from "_{backbone_id}_..." onward to recover the stem.
        prefix = matches[0].name.split(f"_{backbone_id}_")[0]
        args.video_id = prefix
        print(f"Auto-detected video_id: {args.video_id}  ({matches[0].name})")

    loader = VJEPA2DenseLoader(
        args.features_dir, args.fps, args.window_size,
        model_id=backbone_id, max_cached=2, dense_tag=dense_tag,
    )

    # The dense file is self-describing: read valid_lo/valid_hi/anchor_stride
    # and let the loader hand us [N_tokens, D] for each anchor.
    arr = loader._open_video(args.video_id)
    valid_lo = int(arr["valid_lo"])
    valid_hi = int(arr["valid_hi"])
    n_rows = valid_hi - valid_lo + 1
    if n_rows <= 0:
        raise SystemExit(f"Empty dense file for {args.video_id}; no anchors to score.")

    # Pull anchor_stride / n_source_frames from the metadata source: for the
    # streaming layout that's the .meta.npz sidecar, for the legacy layout
    # the bundled single-.npz.
    anchor_stride = None
    n_source_frames = None
    candidates = loader._candidate_paths(args.video_id)
    primary = max(candidates, key=lambda p: p.name.count("_"))
    meta_source = (primary.parent / (primary.stem + ".meta.npz")
                   if primary.suffix == ".npy" else primary)
    with np.load(meta_source) as raw:
        if "anchor_stride" in raw.files:
            anchor_stride = int(raw["anchor_stride"])
        if "n_source_frames" in raw.files:
            n_source_frames = int(raw["n_source_frames"])
    if anchor_stride is None:
        # Legacy files don't store it; fall back to the protocol assumption
        # source_fps = anchor_stride * target_fps, so anchor_stride = source_fps/fps
        # and the SoccerNet experiment runs at source==25 / target==5 -> stride=5.
        anchor_stride = max(1, int(round(25.0 / args.fps)))
        print(f"NOTE: anchor_stride absent from .npz; falling back to {anchor_stride}.")
    source_fps = float(anchor_stride) * float(args.fps)
    duration_s = (n_source_frames / source_fps) if n_source_frames else (n_rows / args.fps)
    if args.max_duration_sec is not None:
        # Anchors here are target-FPS indices (anchor i -> source frame
        # i*anchor_stride). Clamp by max_sec * target_fps.
        max_anchor = int(args.max_duration_sec * args.fps)
        clamped_hi = min(valid_hi, max_anchor)
        if clamped_hi < valid_lo:
            raise SystemExit(
                f"--max-duration-sec={args.max_duration_sec} too small: no valid "
                f"anchor fits in the requested window (valid_lo={valid_lo})."
            )
        if clamped_hi < valid_hi:
            print(f"Clamping anchor range to first {args.max_duration_sec:.0f}s: "
                  f"valid_hi {valid_hi} -> {clamped_hi}.")
            valid_hi = clamped_hi
            n_rows = valid_hi - valid_lo + 1
    print(f"Clip '{args.video_id}': {n_rows} window rows "
          f"(anchors [{valid_lo}, {valid_hi}], stride={anchor_stride}, "
          f"~{duration_s/60:.1f} min)")

    # --- One forward per row -------------------------------------------------
    logits = np.zeros((n_rows, args.num_classes), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for j, anchor in enumerate(range(valid_lo, valid_hi + 1)):
            feats = loader.get_feature(args.video_id, anchor)  # [N_tokens, D]
            x = torch.from_numpy(feats).unsqueeze(0).to(device, non_blocking=True)
            logits[j] = model(x).squeeze(0).cpu().numpy()
            if j % 1000 == 0:
                print(f"  ...{j}/{n_rows} windows", flush=True)
    mask = np.ones(n_rows, dtype=np.float32)

    # --- Peak detection (identical call to eval_temporal Track 3a) -----------
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

    # Peak frame (0-based within valid range) -> absolute timestamp via the
    # protocol: anchor i is centred on source frame i*anchor_stride, so
    # timestamp_sec = (valid_lo + frame) / target_fps.
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

    txt_path = out_dir / f"{stem}_vjepa2_temporal_events.txt"
    with open(txt_path, "w") as f:
        f.write(f"# Fired tackle events for '{stem}' (V-JEPA 2 attentive probe)\n")
        f.write(f"# checkpoint        : {ckpt_path}\n")
        f.write(f"# min_confidence    : {args.min_confidence}\n")
        f.write(f"# sigma / min_dist  : {args.sigma} / {args.min_distance_sec}s\n")
        f.write(f"# W / target / src  : {args.window_size} / {args.fps} / {source_fps} FPS\n")
        f.write(f"# clip duration     : ~{duration_s/60:.1f} min ({n_rows} window rows)\n")
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

    csv_fired = out_dir / f"{stem}_vjepa2_temporal_events.csv"
    csv_all = out_dir / f"{stem}_vjepa2_temporal_events_all.csv"
    _write_csv(csv_fired, fired)
    _write_csv(csv_all, detections)
    print(f"  wrote {csv_fired}")
    print(f"  wrote {csv_all}  (every peak, re-threshold without re-running)")

    (out_dir / f"{stem}_vjepa2_temporal_predict_summary.json").write_text(json.dumps({
        "video_id": stem,
        "checkpoint": str(ckpt_path),
        "backbone": backbone,
        "window_size": args.window_size,
        "fps": args.fps,
        "source_fps": source_fps,
        "anchor_stride": anchor_stride,
        "padding_mode": args.padding_mode,
        "n_window_rows": n_rows,
        "valid_anchor_range": [valid_lo, valid_hi],
        "duration_min": duration_s / 60.0,
        "min_confidence": args.min_confidence,
        "sigma": args.sigma,
        "min_distance_sec": args.min_distance_sec,
        "n_fired": len(fired),
        "n_total_peaks": len(detections),
        "n_fired_tackle_live": n_live,
        "n_fired_tackle_replay": n_replay,
    }, indent=2, default=float))
    print("\nDone.")


if __name__ == "__main__":
    main()
