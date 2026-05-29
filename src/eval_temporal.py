"""
Evaluate a trained attentive probe on TACDEC. Three tracks, all keyed off the
same game-disjoint split as training (splits.split_games, shared
with the spatial probe).

Tracks (selected via --metric):

  1. 'balanced'   — frame-level on the protocol's TEST pool (the same window
     sampler used at training time, but on test-split clips). For
     --protocol centered, this is build_balanced_windows on the test clips:
     one window per event + class-balanced background, strict center-frame
     label. Reports per-class P/R/F1 + confusion matrix.

  2. 'frame_full' — frame-level on FULL test clips with stride-1 sliding.
     Background dominates (≈90% of frames), so macro-F1 / per-class F1 are
     the relevant metrics — accuracy is uninformative. Comparable to
     Kassab TempTAC's frame-level eval.

  3. 'event'      — SoccerNet Average-mAP on full test clips (delta in 1..5 s,
     peak-detected detections).

  Use --metric all to run all three (recommended) and write a single JSON.

Usage:
  uv run python src/eval_temporal.py \
      --backbone-type dinov3 --window-size 10 --fps 5.0 \
      --protocol centered --model-suffix centered_v1 \
      --metric all --seed 42 \
      --save-json results/temporal/dinov3_l_centered_v1_test.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import RESULTS_DIR, TACDEC_FEATURES, TACDEC_LABELS, TACDEC_MODELS
from data.balanced_temporal_dataset import (
    get_balanced_temporal_dataloaders,
)
from data.temporal_loaders import (
    CLASS_NAMES,
    DINOv3DenseLoader,
    VJEPA2DenseLoader,
    _load_video_labels_at_target_fps,
)
from data.kassab_concat_dataset import (
    get_kassab_concat_temporal_dataloaders,
)
from data.splits import (
    split_games as split_games_by_clip,
    split_games_from_file,
)
from data.temporal_protocol import STRIDE_S, extract_events_5fps
from head_efficiency import PROFILE_BATCH_SIZE, profile_head
from models.dinov3.attentive_probe import DINOv3AttentiveProbe
from models.vjepa2.attentive_pooler import AttentiveClassifier
from postprocess import extract_ground_truth_events, postprocess_clip
from soccernet_eval import evaluate_average_map
from utils import set_seed
from window_protocol import valid_anchor_range


# ---- Class ordering ---------------------------------------------------------
#
# Our scheme:    0 = tackle-live, 1 = tackle-replay, 2 = background.
# Kassab TempTAC: 0 = background,  1 = tackle-live,  2 = tackle-replay.
#
# Same merge semantics, different class-id ordering. KASSAB_PERM[i] is the
# Kassab-space id for our class i (live=0->1, replay=1->2, bg=2->0). Used to
# emit a side-by-side classification_report in Kassab's row order so the
# numbers line up column-for-column with TempTAC.ipynb's output.
KASSAB_PERM = np.array([1, 2, 0], dtype=np.int64)
KASSAB_CLASS_NAMES = ["background", "tackle-live", "tackle-replay"]


def _kassab_relabel(arr: np.ndarray) -> np.ndarray:
    """Remap an int array of our-scheme labels into Kassab's class-id ordering."""
    return KASSAB_PERM[arr.astype(np.int64)]


def _print_kassab_order_report(labels: np.ndarray, preds: np.ndarray,
                               num_classes: int, header: str) -> dict:
    """Print + return a classification_report in Kassab's class ordering."""
    if num_classes != 3:
        # Permutation is only defined for the 3-class scheme used by Kassab.
        return {}
    lk = _kassab_relabel(labels)
    pk = _kassab_relabel(preds)
    print("\n" + "-" * 60)
    print(f"{header} -- Kassab class ordering (0=bg, 1=live, 2=replay)")
    print("-" * 60)
    print(classification_report(lk, pk, target_names=KASSAB_CLASS_NAMES,
                                 digits=4, zero_division=0,
                                 labels=[0, 1, 2]))
    cm_k = confusion_matrix(lk, pk, labels=[0, 1, 2])
    print("Confusion matrix (rows=true, cols=pred), Kassab order:")
    print(f"  {'':<16}" + "".join(f"{n:>16}" for n in KASSAB_CLASS_NAMES))
    for i, name in enumerate(KASSAB_CLASS_NAMES):
        print(f"  {name:<16}" + "".join(f"{cm_k[i, j]:>16d}"
                                          for j in range(num_classes)))
    return {
        "classification_report": classification_report(
            lk, pk, target_names=KASSAB_CLASS_NAMES,
            digits=4, output_dict=True, zero_division=0,
            labels=[0, 1, 2],
        ),
        "confusion_matrix": cm_k.tolist(),
        "class_order": KASSAB_CLASS_NAMES,
    }


# ---- IO ---------------------------------------------------------------------


def find_checkpoint(backbone_id: str, model_suffix: str) -> Path:
    name = f"{backbone_id}_{model_suffix}" if model_suffix else backbone_id
    candidates = [
        TACDEC_MODELS / backbone_id / f"best_attn_{name}.pth",
        TACDEC_MODELS / backbone_id / f"{name}.pth",
        TACDEC_MODELS / f"best_attn_{name}.pth",
        TACDEC_MODELS / f"{name}.pth",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Checkpoint not found. Tried:\n" + "\n".join(f"  - {p}" for p in candidates)
    )


# ---- Model rebuild ----------------------------------------------------------


def rebuild_probe(ckpt: dict):
    backbone = ckpt["backbone_type"]
    feature_dim = int(ckpt["feature_dim"])
    num_classes = int(ckpt["num_classes"])
    if backbone == "dinov3":
        m = DINOv3AttentiveProbe(
            in_dim=feature_dim,
            probe_dim=feature_dim,
            num_classes=num_classes,
            num_heads=int(ckpt.get("probe_num_heads", 16)),
            num_blocks=int(ckpt.get("probe_depth", 4)),
            t_size=int(ckpt["window_size"]),
            h_size=int(ckpt.get("patch_h", 16)),
            w_size=int(ckpt.get("patch_w", 16)),
        )
    elif backbone == "vjepa2":
        m = AttentiveClassifier(
            embed_dim=feature_dim,
            num_heads=int(ckpt.get("probe_num_heads", 16)),
            depth=int(ckpt.get("probe_depth", 4)),
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"unknown backbone in checkpoint: {backbone!r}")
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, backbone


# ---- Report 1: per-window classification on the protocol's test pool --------
# Each test row is one window with one center-frame label and one argmax
# prediction, so the classification_report's support column is also the
# frame-prediction count on that pool. Under --protocol kassab_concat this is
# directly comparable to Kassab TempTAC's frame-level classification_report,
# both in scope (capped sequences) and in unit (one prediction per frame).


@torch.no_grad()
def run_per_window(model, backbone, loader, device):
    all_preds, all_labels = [], []
    all_video_ids: list[str] = []
    all_anchors: list[int] = []
    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(feats)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_video_ids.extend(batch["video_ids"])
        all_anchors.extend(int(a) for a in batch["anchors"].tolist())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        all_video_ids,
        np.asarray(all_anchors, dtype=np.int64),
    )


# ---- Report 2: per-clip stride-1 -> peak detection -> average-mAP -----------


def _build_per_clip_logits(model, backbone, loader_fn, video_id, n_target,
                            window_size, num_classes, device):
    """
    Run the probe on every stride-1 anchor row of one clip. Returns:
      logits : np.ndarray [n_target, num_classes]
      mask   : np.ndarray [n_target]   (1.0 for all rows; boundaries handled
                                         via zero-padding inside the loader)
    """
    feature_loader = loader_fn()
    logits = np.zeros((n_target, num_classes), dtype=np.float32)
    # Stride-1 over every anchor row; the loader handles boundary zero-pad.
    for anchor in range(n_target):
        feats = feature_loader.get_feature(video_id, anchor)
        x = torch.from_numpy(feats).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
        logits[anchor] = out.squeeze(0).cpu().numpy()
    mask = np.ones(n_target, dtype=np.float32)
    return logits, mask


def run_per_clip_map(
    model,
    backbone: str,
    test_video_ids: list[str],
    labels_dir: Path,
    features_dir: Path,
    target_fps: float,
    window_size: int,
    num_classes: int,
    device: torch.device,
    feature_cache_size: int,
    delta_grid: list[float],
    min_distance_sec: float,
    sigma: float,
    source_fps: float | None = None,
    dense_tag: str = "",
):
    """
    Iterate every test-game video, run stride-1 inference, peak-detect events,
    aggregate, then evaluate average-mAP at the requested deltas.
    """
    eff_source = source_fps if source_fps is not None else target_fps
    if backbone == "dinov3":
        loader_fn = lambda: DINOv3DenseLoader(features_dir, target_fps,
                                               window_size,
                                               source_fps=eff_source,
                                               max_cached=feature_cache_size,
                                               dense_tag=dense_tag)
    else:
        loader_fn = lambda: VJEPA2DenseLoader(features_dir, target_fps,
                                               window_size,
                                               max_cached=feature_cache_size,
                                               dense_tag=dense_tag)

    all_detections, all_gts, all_seq_lens = [], [], []
    print(f"Running stride-1 mAP eval on {len(test_video_ids)} clips...")

    feat_loader = loader_fn()  # one shared loader for the whole sweep

    for video_id in test_video_ids:
        # GT and clip length at target FPS.
        label_path = labels_dir / f"{video_id}.json"
        if not label_path.exists():
            print(f"  skip {video_id}: missing label file")
            continue
        labels_target, _, n_target, src_stride = _load_video_labels_at_target_fps(
            label_path, target_fps,
        )
        if n_target == 0:
            continue

        # Kassab no-pad: only anchors whose window fits entirely inside the
        # video produce predictions. Boundary anchors are dropped (mask=0)
        # so peak detection / Average-mAP ignores them.
        n_source = n_target * src_stride
        valid_lo, valid_hi = valid_anchor_range(
            video_length=n_source,
            anchor_stride=src_stride,
            intra_window_stride=src_stride,
            window_length=window_size,
        )
        if valid_hi < valid_lo:
            print(f"  skip {video_id}: video too short for W={window_size} at "
                  f"target_fps={target_fps}")
            continue

        # Build full-length logits/mask in target-FPS space; non-valid
        # anchors stay zero with mask=0.
        logits = np.zeros((n_target, num_classes), dtype=np.float32)
        seq_mask = np.zeros(n_target, dtype=np.float32)
        for anchor in range(valid_lo, valid_hi + 1):
            feats = feat_loader.get_feature(video_id, anchor)
            x = torch.from_numpy(feats).unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                out = model(x)
            logits[anchor] = out.squeeze(0).cpu().numpy()
            seq_mask[anchor] = 1.0

        # Peak-based detections (we trained on anchor-style center-frame
        # labels; logits peak around event centers, peak detection is correct).
        detections = postprocess_clip(
            logits=logits,
            mask=seq_mask,
            num_classes=num_classes,
            fps=target_fps,
            method="peak",
            labeling_mode="anchor",
            sigma=sigma,
            min_confidence=0.0,         # 0 = let SoccerNet sweep all thresholds
            min_distance_sec=min_distance_sec,
            nms=False,                  # SoccerNet protocol: per-class only
        )
        gts = extract_ground_truth_events(
            labels=labels_target,
            mask=seq_mask,
            fps=target_fps,
            num_classes=num_classes,
        )

        all_detections.append(detections)
        all_gts.append(gts)
        all_seq_lens.append(int(n_target))

    print(f"  collected detections for {len(all_detections)} clips.")
    # SoccerNet's tight metric uses delta in {1..5}s; we want a custom grid
    # {0.5, 1, 2, 3}. soccernet_eval supports tight/loose/at1..at5; for a
    # custom delta-curve we pass metric='tight' and post-filter the per-
    # tolerance breakdown (or call evaluate_average_map per delta with 'atK').
    # Run 'tight' once to get the standard panel, then 'at1' etc. if needed.
    map_results = evaluate_average_map(
        all_detections=all_detections,
        all_ground_truths=all_gts,
        all_seq_lens=all_seq_lens,
        num_classes=num_classes,
        fps=target_fps,
        metric="tight",
        class_names=[CLASS_NAMES[c] for c in range(num_classes - 1)],
        verbose=False,
    )
    return map_results, len(all_detections)


# ---- Per-clip stride-1 frame predictions (feeds track 2) -------------------


def run_per_clip_frame_eval(
    model,
    backbone: str,
    test_video_ids: list[str],
    labels_dir: Path,
    features_dir: Path,
    target_fps: float,
    window_size: int,
    num_classes: int,
    device: torch.device,
    feature_cache_size: int,
    source_fps: float | None = None,
    dense_tag: str = "",
) -> list[dict]:
    """
    Run stride-1 inference across every test clip and return per-clip frame
    predictions. One forward per anchor row inside the clip's valid range
    (the same `valid_anchor_range` used at training time).

    Returned list, one entry per processed clip:
        {
            "video_id":   str,
            "n_target":   int,            # 5-FPS rows in the clip
            "valid_lo":   int,            # inclusive
            "valid_hi":   int,            # inclusive
            "predictions": np.ndarray[n_target, int64],   # argmax per anchor;
                                                          # -1 outside [valid_lo, valid_hi]
            "labels_5fps": np.ndarray[n_target, int64],   # full 5-FPS label timeline
            "events_5fps": list[(cls, s_5, e_5)],         # ground-truth events
        }

    Used by track 2 (frame_full, flatten predictions/labels over valid range).
    """
    eff_source = source_fps if source_fps is not None else target_fps
    if backbone == "dinov3":
        loader_fn = lambda: DINOv3DenseLoader(features_dir, target_fps,
                                               window_size,
                                               source_fps=eff_source,
                                               max_cached=feature_cache_size,
                                               dense_tag=dense_tag)
    else:
        loader_fn = lambda: VJEPA2DenseLoader(features_dir, target_fps,
                                               window_size,
                                               max_cached=feature_cache_size,
                                               dense_tag=dense_tag)

    feat_loader = loader_fn()
    per_clip: list[dict] = []
    print(f"Running stride-1 per-clip frame eval on {len(test_video_ids)} clips...")

    for video_id in test_video_ids:
        label_path = labels_dir / f"{video_id}.json"
        if not label_path.exists():
            print(f"  skip {video_id}: missing label file")
            continue
        labels_target, _, n_target, src_stride = _load_video_labels_at_target_fps(
            label_path, target_fps,
        )
        if n_target == 0:
            continue

        n_source = n_target * src_stride
        valid_lo, valid_hi = valid_anchor_range(
            video_length=n_source,
            anchor_stride=src_stride,
            intra_window_stride=src_stride,
            window_length=window_size,
        )
        if valid_hi < valid_lo:
            continue

        preds = np.full(n_target, -1, dtype=np.int64)
        for anchor in range(valid_lo, valid_hi + 1):
            feats = feat_loader.get_feature(video_id, anchor)
            x = torch.from_numpy(feats).unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                out = model(x)
            preds[anchor] = int(out.squeeze(0).argmax(dim=-1).cpu().item())

        # Ground-truth events at 5 FPS (target_fps). extract_events_5fps
        # assumes target_fps == 5; for other fps the stride conversion still
        # works since STRIDE_S = 25 / target_fps.
        events_5fps = extract_events_5fps(label_path)

        per_clip.append({
            "video_id":   video_id,
            "n_target":   int(n_target),
            "valid_lo":   int(valid_lo),
            "valid_hi":   int(valid_hi),
            "predictions": preds,
            "labels_5fps": labels_target.astype(np.int64),
            "events_5fps": events_5fps,
        })

    print(f"  collected frame predictions for {len(per_clip)} clips.")
    return per_clip


# ---- Misclassification dump (mirrors eval_spatial.py schema) ----------------


def _write_misclassifications_csv(
    out_path: Path,
    video_ids: list[str],
    anchors: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    fps: float,
) -> int:
    """Write one row per misclassified sample. Schema matches eval_spatial.py:
    clip_id, frame_idx, time_sec, true_label, pred_label, true_class, pred_class.
    For temporal: frame_idx is the 5-FPS anchor index. Returns rows written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "frame_idx", "time_sec",
            "true_label", "pred_label", "true_class", "pred_class",
        ])
        writer.writeheader()
        for i in range(len(labels)):
            true_label = int(labels[i])
            pred_label = int(preds[i])
            if pred_label == true_label:
                continue
            anchor = int(anchors[i])
            writer.writerow({
                "clip_id": video_ids[i],
                "frame_idx": anchor,
                "time_sec": float(anchor) / float(fps),
                "true_label": true_label,
                "pred_label": pred_label,
                "true_class": CLASS_NAMES[true_label],
                "pred_class": CLASS_NAMES[pred_label],
            })
            rows += 1
    return rows


# ---- CLI --------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backbone-type", choices=["dinov3", "vjepa2"], required=True)
    ap.add_argument("--backbone-size", default="large",
                    choices=["base", "large", "huge", "giant"])
    ap.add_argument("--model-suffix", required=True)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--fps", type=float, default=5.0,
                    help="Target FPS (default 5.0; shared 5 FPS protocol).")
    ap.add_argument("--window-size", type=int, default=10,
                    help="W (default 10; 5 FPS * 2 s, even).")
    ap.add_argument("--protocol",
                    choices=["centered", "kassab_concat"],
                    default="centered",
                    help="Test-pool sampler for the 'balanced' track only. Should "
                         "match the protocol used at training time. 'kassab_concat' "
                         "is strict Kassab TempTAC parity at 5 FPS (cross-clip "
                         "concat-and-slide, DINOv3-only) -- use this for direct "
                         "comparison against Kassab's frame-level "
                         "classification_report. The 'frame_full' and 'event' "
                         "tracks are protocol-independent (always dense stride-1 "
                         "per clip).")
    ap.add_argument("--replay-cap", type=int, default=280,
                    help="'kassab_concat' protocol, 'balanced' track: replay-sequence "
                         "cap for the test pool.")
    ap.add_argument("--bg-count", type=int, default=500,
                    help="'kassab_concat' protocol, 'balanced' track: background-"
                         "sequence count for the test pool.")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="'balanced' track batch size.")
    ap.add_argument("--feature-cache", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="'balanced' track DataLoader workers. The 'frame_full' "
                         "and 'event' tracks iterate anchors one at a time and "
                         "ignore this flag.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source-fps", type=float, default=None,
                    help="DINOv3 only: FPS embedded in the on-disk feature filename. "
                         "Defaults to the value saved in the checkpoint (or --fps "
                         "for legacy checkpoints). Set explicitly to override.")
    ap.add_argument("--metric",
                    choices=["balanced", "frame_full", "event", "all"],
                    default="all",
                    help="Which eval track(s) to run. 'balanced' = frame-level on "
                         "the protocol's test pool. 'frame_full' = frame-level "
                         "stride-1 on full test clips. 'event' = SoccerNet "
                         "Avg-mAP (peak-detected). 'all' runs all three.")

    # mAP knobs (SoccerNet defaults preserved upstream)
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian smoothing sigma for peak detection (Avg-mAP).")
    ap.add_argument("--min-distance-sec", type=float, default=0.5,
                    help="Per-class min distance between peaks (Avg-mAP).")

    ap.add_argument("--save-json", type=str, default=None)
    ap.add_argument("--padding-mode", choices=["center_crop", "reflect"],
                    default="center_crop",
                    help="Extraction flavour: 'center_crop' loads the default "
                         "dense files; 'reflect' loads *_reflect_dense_* files "
                         "produced by extract_features.py --padding-mode reflect.")
    ap.add_argument("--split-file", type=str, default=None,
                    help="Optional JSON file with explicit train/val/test "
                         "clip-ID lists. Overrides the seeded game-disjoint "
                         "split for all three tracks. Must match the split "
                         "used at training time. Use "
                         "scripts/dump_kassab_split.py for Kassab TempTAC "
                         "parity.")
    ap.add_argument("--split-mode",
                    choices=["kassab_bug", "correct"],
                    default="kassab_bug",
                    help="kassab_concat protocol only. Must match the value "
                         "used at training time. 'kassab_bug' (default) "
                         "evaluates on the positional-slice pool that matches "
                         "Kassab's reported numbers; 'correct' evaluates on "
                         "the game-disjoint partition.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_id = f"{args.backbone_type}_{args.backbone_size[0]}"

    # Find + load checkpoint
    ckpt_path = find_checkpoint(backbone_id, args.model_suffix)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    train_window = int(ckpt.get("window_size", args.window_size))
    train_fps = float(ckpt.get("fps", args.fps))
    train_source_fps = float(ckpt.get("source_fps", train_fps))
    if train_window != args.window_size or train_fps != args.fps:
        print(f"NOTE: model trained with W={train_window}@{train_fps} FPS; evaluating "
              f"at W={args.window_size}@{args.fps} FPS. Set --window-size/--fps to "
              "match training unless you intend the mismatch.")
    # source_fps from checkpoint controls which on-disk file is read. Falls
    # back to train_fps when the checkpoint is from before the stride-indexing
    # change (back-compat).
    eff_source_fps = (args.source_fps
                      if args.source_fps is not None else train_source_fps)

    model, backbone = rebuild_probe(ckpt)
    model = model.to(device)
    print(f"Probe loaded ({backbone}), params: {sum(p.numel() for p in model.parameters()):,}")

    dense_tag = "reflect" if args.padding_mode == "reflect" else ""
    print(f"Padding:     {args.padding_mode}"
          + (f"  (loading *_{dense_tag}_dense_* files)" if dense_tag else ""))

    target_names = [CLASS_NAMES[c] for c in range(args.num_classes)]
    features_dir = TACDEC_FEATURES / f"{args.backbone_type}_{args.backbone_size}"

    bal_report = bal_cm = None
    ff_report = ff_cm = None
    bal_kassab: dict = {}
    ff_kassab: dict = {}
    map_results = None
    n_test_clips = 0

    want_balanced   = args.metric in ("balanced", "all")
    want_frame_full = args.metric in ("frame_full", "all")
    want_event      = args.metric in ("event", "all")

    # ---- Track 1: frame-level on the protocol's test pool -------------------
    if want_balanced:
        if args.protocol == "centered":
            _, _, test_loader, info = get_balanced_temporal_dataloaders(
                labels_dir=TACDEC_LABELS,
                features_dir=features_dir,
                backbone=backbone,
                window_size=args.window_size,
                target_fps=args.fps,
                source_fps=eff_source_fps,
                seed=args.seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                feature_loader_cache=args.feature_cache,
                dense_tag=dense_tag,
                split_file=args.split_file,
            )
        elif args.protocol == "kassab_concat":
            if backbone != "dinov3":
                raise NotImplementedError(
                    "kassab_concat eval is DINOv3-only. V-JEPA 2 features "
                    "can't form cross-clip windows."
                )
            _, _, test_loader, info = get_kassab_concat_temporal_dataloaders(
                labels_dir=TACDEC_LABELS,
                features_dir=features_dir,
                backbone=backbone,
                window_size=args.window_size,
                target_fps=args.fps,
                source_fps=eff_source_fps,
                seed=args.seed,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                feature_loader_cache=args.feature_cache,
                dense_tag=dense_tag,
                replay_cap=args.replay_cap,
                bg_count=args.bg_count,
                split_file=args.split_file,
                split_mode=args.split_mode,
            )
        else:
            raise NotImplementedError(
                f"--protocol {args.protocol!r} not wired in eval."
            )
        test_windows = info["_splits"]["test"]
        print(f"\n[balanced] test pool: {len(test_windows)} windows "
              f"({len(info['game_ids']['test'])} games)")
        preds, labels, bal_video_ids, bal_anchors = run_per_window(
            model, backbone, test_loader, device,
        )

        print("\n" + "=" * 60)
        if args.protocol == "kassab_concat":
            unit_note = ("cross-clip concat-and-slide windows; support column "
                         "= one prediction per global-stream window (strict "
                         "Kassab TempTAC parity at 5 FPS)")
        else:
            unit_note = "support column = one frame prediction per window"
        print(f"Track 1: balanced test pool -- frame-level "
              f"(protocol={args.protocol}, {len(preds)} predictions)")
        print(f"         [{unit_note}]")
        print("=" * 60)
        print(classification_report(labels, preds, target_names=target_names,
                                     digits=4, zero_division=0))
        cm = confusion_matrix(labels, preds, labels=list(range(args.num_classes)))
        print("Confusion matrix (rows=true, cols=pred):")
        print(f"  {'':<16}" + "".join(f"{n:>16}" for n in target_names))
        for i, name in enumerate(target_names):
            print(f"  {name:<16}" + "".join(f"{cm[i, j]:>16d}"
                                              for j in range(args.num_classes)))
        bal_report = classification_report(labels, preds, target_names=target_names,
                                           digits=4, output_dict=True, zero_division=0)
        bal_cm = cm
        bal_kassab = _print_kassab_order_report(
            labels, preds, args.num_classes, "Track 1: balanced",
        )

        # Per-misclassification dump (balanced track). Same schema as
        # eval_spatial.py; frame_idx is the 5-FPS anchor index.
        if args.save_json:
            save_json = Path(args.save_json)
            miscls_path = save_json.with_name(
                save_json.stem + ".misclassifications_balanced.csv"
            )
            n_miscls = _write_misclassifications_csv(
                miscls_path,
                video_ids=bal_video_ids,
                anchors=bal_anchors,
                labels=labels,
                preds=preds,
                fps=args.fps,
            )
            print(f"       wrote {n_miscls} misclassifications -> {miscls_path.name}")
        else:
            print("       --save-json not set; skipping misclassifications CSV")

    # ---- Track 2 stride-1 per-clip frame predictions -----------------------
    per_clip_frames: list[dict] = []
    if want_frame_full or want_event:
        if args.split_file is not None:
            splits = split_games_from_file(args.split_file)
        else:
            splits = split_games_by_clip(TACDEC_LABELS, val_frac=0.15,
                                          test_frac=0.15, seed=args.seed)
        test_video_ids = sorted(splits["test"])
        print(f"\n[frame_full/event] test clips: {len(test_video_ids)}")

    if want_frame_full:
        per_clip_frames = run_per_clip_frame_eval(
            model=model,
            backbone=backbone,
            test_video_ids=test_video_ids,
            labels_dir=TACDEC_LABELS,
            features_dir=features_dir,
            target_fps=args.fps,
            source_fps=eff_source_fps,
            window_size=args.window_size,
            num_classes=args.num_classes,
            device=device,
            feature_cache_size=args.feature_cache,
            dense_tag=dense_tag,
        )

    # ---- Track 2: frame-level on full test clips ----------------------------
    if want_frame_full:
        preds_ff = []
        labels_ff = []
        for clip in per_clip_frames:
            lo, hi = clip["valid_lo"], clip["valid_hi"]
            preds_ff.append(clip["predictions"][lo:hi + 1])
            labels_ff.append(clip["labels_5fps"][lo:hi + 1])
        preds_ff = np.concatenate(preds_ff) if preds_ff else np.array([], dtype=np.int64)
        labels_ff = np.concatenate(labels_ff) if labels_ff else np.array([], dtype=np.int64)

        print("\n" + "=" * 60)
        print(f"Track 2: frame-level on full test clips "
              f"({len(per_clip_frames)} clips, {len(preds_ff)} valid frames)")
        print("=" * 60)
        print(classification_report(labels_ff, preds_ff, target_names=target_names,
                                     digits=4, zero_division=0))
        cm_ff = confusion_matrix(labels_ff, preds_ff,
                                 labels=list(range(args.num_classes)))
        print("Confusion matrix (rows=true, cols=pred):")
        print(f"  {'':<16}" + "".join(f"{n:>16}" for n in target_names))
        for i, name in enumerate(target_names):
            print(f"  {name:<16}" + "".join(f"{cm_ff[i, j]:>16d}"
                                              for j in range(args.num_classes)))
        ff_report = classification_report(labels_ff, preds_ff,
                                          target_names=target_names,
                                          digits=4, output_dict=True,
                                          zero_division=0)
        ff_cm = cm_ff
        ff_kassab = _print_kassab_order_report(
            labels_ff, preds_ff, args.num_classes, "Track 2: frame_full",
        )

        # Per-misclassification dump (frame_full track). One row per anchor
        # in [valid_lo, valid_hi] where pred != label, across all test clips.
        if args.save_json:
            ff_video_ids: list[str] = []
            ff_anchors_list: list[int] = []
            ff_labels_list: list[int] = []
            ff_preds_list: list[int] = []
            for clip in per_clip_frames:
                lo, hi = clip["valid_lo"], clip["valid_hi"]
                clip_preds = clip["predictions"][lo:hi + 1]
                clip_labels = clip["labels_5fps"][lo:hi + 1]
                for offset, (p, l) in enumerate(zip(clip_preds, clip_labels)):
                    if int(p) == int(l):
                        continue
                    ff_video_ids.append(clip["video_id"])
                    ff_anchors_list.append(lo + offset)
                    ff_labels_list.append(int(l))
                    ff_preds_list.append(int(p))
            save_json = Path(args.save_json)
            miscls_path = save_json.with_name(
                save_json.stem + ".misclassifications_frame_full.csv"
            )
            n_miscls = _write_misclassifications_csv(
                miscls_path,
                video_ids=ff_video_ids,
                anchors=np.asarray(ff_anchors_list, dtype=np.int64),
                labels=np.asarray(ff_labels_list, dtype=np.int64),
                preds=np.asarray(ff_preds_list, dtype=np.int64),
                fps=args.fps,
            )
            print(f"       wrote {n_miscls} misclassifications -> {miscls_path.name}")
        else:
            print("       --save-json not set; skipping misclassifications CSV")

    # ---- Track 3: event-level (SoccerNet Avg-mAP) --------------------------
    if want_event:
        # Peak-detected Avg-mAP via SoccerNet. Re-runs the forward pass because
        # peak detection needs the full softmax, not just argmax.
        # (Future optimization: thread softmax through run_per_clip_frame_eval.)
        map_results, n_test_clips = run_per_clip_map(
            model=model,
            backbone=backbone,
            test_video_ids=test_video_ids,
            labels_dir=TACDEC_LABELS,
            features_dir=features_dir,
            target_fps=args.fps,
            source_fps=eff_source_fps,
            window_size=args.window_size,
            num_classes=args.num_classes,
            device=device,
            feature_cache_size=args.feature_cache,
            delta_grid=[1.0, 2.0, 3.0, 4.0, 5.0],
            min_distance_sec=args.min_distance_sec,
            sigma=args.sigma,
            dense_tag=dense_tag,
        )

        print("\n" + "=" * 60)
        print(f"Track 3: Average-mAP (SoccerNet tight, delta in 1..5 s; "
              f"{n_test_clips} clips)")
        print("=" * 60)
        print(f"  Average-mAP          : {map_results['average_mAP']:.4f}")
        print(f"  Per-class avg AP     :")
        action_names = [CLASS_NAMES[c] for c in range(args.num_classes - 1)]
        for c, name in enumerate(action_names):
            print(f"    {name:<16}: {map_results['per_class_avg_ap'][c]:.4f}")
        print(f"  Per-tolerance mAP    :")
        for entry in map_results["per_tolerance"]:
            print(f"    delta={entry['tolerance_sec']:>4.1f}s  mAP={entry['mAP']:.4f}")

    # ---- Save JSON ----------------------------------------------------------
    if args.save_json:
        out = {
            "args": vars(args),
            "checkpoint": str(ckpt_path),
            "backbone": backbone,
            "backbone_id": backbone_id,
            "feature_dim": int(ckpt["feature_dim"]),
            "train_window": train_window,
            "train_fps": train_fps,
            "n_test_clips": n_test_clips,
            "balanced": {
                "classification_report": bal_report,
                "confusion_matrix": bal_cm.tolist() if bal_cm is not None else None,
                "kassab_order": bal_kassab,
            },
            "frame_full": {
                "classification_report": ff_report,
                "confusion_matrix": ff_cm.tolist() if ff_cm is not None else None,
                "kassab_order": ff_kassab,
            },
            "event": {
                "avg_map": map_results,
            },
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(out, indent=2, default=float))
        print(f"\nResults saved to {args.save_json}")

    # ---- Head-only efficiency profile (always on when CUDA is available) ----
    if not torch.cuda.is_available():
        print("\n[profile] Skipped head-only efficiency profile (no CUDA device).")
    else:
        print("\n" + "=" * 60)
        print("Head-only efficiency profile (bs=16, fp32)")
        print("=" * 60)

        # Fresh DataLoader at the profile batch size, reusing the same balanced
        # test pool used by Track 1. Centered protocol is the canonical one for
        # both backbones; kassab_concat (DINOv3-only) is also supported.
        if args.protocol == "centered":
            _, _, prof_loader, _ = get_balanced_temporal_dataloaders(
                labels_dir=TACDEC_LABELS,
                features_dir=features_dir,
                backbone=backbone,
                window_size=args.window_size,
                target_fps=args.fps,
                source_fps=eff_source_fps,
                seed=args.seed,
                batch_size=PROFILE_BATCH_SIZE,
                num_workers=args.num_workers,
                feature_loader_cache=args.feature_cache,
                dense_tag=dense_tag,
                split_file=args.split_file,
            )
        elif args.protocol == "kassab_concat":
            _, _, prof_loader, _ = get_kassab_concat_temporal_dataloaders(
                labels_dir=TACDEC_LABELS,
                features_dir=features_dir,
                backbone=backbone,
                window_size=args.window_size,
                target_fps=args.fps,
                source_fps=eff_source_fps,
                seed=args.seed,
                batch_size=PROFILE_BATCH_SIZE,
                num_workers=args.num_workers,
                feature_loader_cache=args.feature_cache,
                dense_tag=dense_tag,
                split_file=args.split_file,
                split_mode=args.split_mode,
            )
        else:
            raise NotImplementedError(
                f"head efficiency profile not wired for --protocol {args.protocol!r}"
            )

        # The DataLoader yields dict batches; the profiler wants raw feature
        # tensors. Wrap as a generator that pulls out the 'features' field.
        def _feat_iter():
            for batch in prof_loader:
                yield batch["features"]

        pipeline = (
            "dinov3_attentive" if backbone == "dinov3" else "vjepa2_attentive"
        )
        csv_path = RESULTS_DIR / "head_efficiency.csv"
        prof = profile_head(
            head=model,
            feature_batch_iter=_feat_iter(),
            pipeline=pipeline,
            csv_path=csv_path,
        )
        print(f"  pipeline:                 {prof['pipeline']}")
        print(f"  trainable params (M):     {prof['trainable_params_M']:.6f}")
        print(f"  mean batch latency (ms):  {prof['mean_batch_latency_ms']:.4f}  "
              f"(bs={prof['batch_size']}, warmup={prof['n_warmup']}, timed={prof['n_timed']})")
        print(f"  peak head VRAM (MiB):     {prof['peak_vram_mib']:.2f}")
        print(f"  device:                   {prof['device']}")
        print(f"  appended to:              {prof['csv_path']}")


if __name__ == "__main__":
    main()
