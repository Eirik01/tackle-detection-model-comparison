"""Evaluation for the DINOv3 linear spatial probe.

Runs the trained probe on every frame of every held-out test clip (no
subsampling) and reports three views:

1. Frame-level on the natural distribution (~89% background) — per-class
   accuracy, F1, confusion matrix. Honest "how does the probe see real
   broadcast video".
2. Frame-level on a class-balanced subsample of the test split — same
   undersampling procedure (`balance_split`) used during training, applied
   to the test clips with the run's `seed_balance`. Matches the balanced
   train/val view.
3. Event-level Average-mAP, SoccerNet tight metric (deltas in {1..5}s) —
   evaluated at 5 FPS by stride-subsampling the per-frame logits, so the
   cadence matches the attentive probes. Detections come from the
   segment-based postprocessor (interval-trained probe -> plateau output);
   results are written to `eval_events.json`.

Run from tackle-detection-model-comparison/ as:
    python src/eval_spatial.py --run-dir results/dinov3_linear_spatial/run_<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

import config
from data.labels import (
    BACKGROUND,
    CLASS_NAMES,
    CLASS_ORDER,
    LIVE_TYPES as _LIVE_TYPES,
    REPLAY_TYPES as _REPLAY_TYPES,
    TACKLE_LIVE,
    TACKLE_REPLAY,
)
from data.splits import balance_split, build_frame_labels
from head_efficiency import PROFILE_BATCH_SIZE, profile_head
from models.dinov3.linear_probe import DINOv3LinearProbe
from postprocess import postprocess_clip
from soccernet_eval import evaluate_average_map


ACTION_CLASS_NAMES = [CLASS_NAMES[c] for c in (TACKLE_LIVE, TACKLE_REPLAY)]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _feature_path(cache_dir: Path, clip_id: str, backbone_id: str, fps: float) -> Path:
    return cache_dir / f"{clip_id}_{backbone_id}_{fps}fps_features.npz"


def load_clip_cls(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["cls"].astype(np.float32)


def extract_gt_events(label_path: Path) -> List[Dict]:
    """Per-clip ground-truth events with merged 3-class labels.

    Returns one dict per annotated event with keys: class, frame (center),
    using the same merging convention as build_frame_labels (live + live-inc
    -> 0, replay + replay-inc -> 1). Background and unknown types are skipped.
    """
    with open(label_path) as f:
        data = json.load(f)

    frame_count = data["media_attributes"]["frame_count"]
    out: List[Dict] = []
    for event in data["events"]:
        t = event["type"]
        if t in _LIVE_TYPES:
            cls = TACKLE_LIVE
        elif t in _REPLAY_TYPES:
            cls = TACKLE_REPLAY
        else:
            continue
        start = event["frame_start"]
        end = min(event["frame_end"], frame_count - 1)
        center = (start + end) // 2
        out.append({"class": cls, "frame": center})
    return out


def print_confusion_matrix(cm: np.ndarray, indent: str = "       ") -> None:
    names = [CLASS_NAMES[c] for c in CLASS_ORDER]
    col_w = max(14, max(len(n) for n in names) + 2)
    row_label_w = max(len(n) for n in names)
    print(f"{indent}confusion matrix (rows=true, cols=pred):")
    print(f"{indent}{'':<{row_label_w}}" + "".join(f"{n:>{col_w}}" for n in names))
    for i, name in enumerate(names):
        print(f"{indent}{name:<{row_label_w}}" + "".join(f"{cm[i, j]:>{col_w}d}" for j in range(len(names))))


def save_confusion_matrix(cm: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels([CLASS_NAMES[c] for c in CLASS_ORDER], rotation=30, ha="right")
    ax.set_yticklabels([CLASS_NAMES[c] for c in CLASS_ORDER])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def infer_clip(
    model: torch.nn.Module,
    cls_features: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Per-frame logits for one clip. Returns float32 array of shape (T, 3)."""
    model.eval()
    T = cls_features.shape[0]
    out = np.empty((T, 3), dtype=np.float32)
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        batch = torch.from_numpy(cls_features[start:end]).to(device)
        out[start:end] = model(batch).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="Training run directory (contains model.pt, splits.json, config.json).")
    p.add_argument("--label-dir", type=Path, default=None, help="Override label-dir from run config.")
    p.add_argument("--feature-cache-dir", type=Path, default=None, help="Override feature-cache-dir from run config.")

    # Event-detection knobs (postprocess_clip)
    p.add_argument("--min-segment-frames", type=int, default=2)
    p.add_argument("--min-confidence", type=float, default=0.0, help="0.0 keeps full PR curve for canonical SoccerNet mAP.")
    p.add_argument("--metric", type=str, default="tight", choices=["tight", "loose"])
    p.add_argument("--nms", action="store_true", help="Apply cross-class NMS on top of segment detection.")


    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg = json.loads((run_dir / "config.json").read_text())
    splits = json.loads((run_dir / "splits.json").read_text())

    label_dir = Path(args.label_dir or cfg["label_dir"])
    feature_cache_dir = Path(args.feature_cache_dir or cfg["feature_cache_dir"])
    backbone_id = cfg["backbone_id"]
    fps = cfg["fps"]
    feature_dim = cfg["feature_dim"]

    test_clips: List[str] = splits["test"]
    print(f"[setup] run-dir: {run_dir}")
    print(f"[setup] test clips: {len(test_clips)}")
    print(f"[setup] backbone={backbone_id} fps={fps} feature_dim={feature_dim}")
    print(f"[setup] event detection: method=segment, min_seg={args.min_segment_frames}, "
          f"min_conf={args.min_confidence}, metric={args.metric}, nms={args.nms}, eval_fps=5.0")

    device = torch.device(config.DEVICE if torch.cuda.is_available() or config.DEVICE == "cpu" else "cpu")

    # --- Load model -----------------------------------------------------
    model = DINOv3LinearProbe(embed_dim=feature_dim, num_classes=3).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- Inference on every frame of every test clip --------------------
    print("\n[1/4] Running probe on every frame of every test clip")
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    labels_by_clip: Dict[str, np.ndarray] = {}
    logits_by_clip: Dict[str, np.ndarray] = {}
    # Event-level evaluation runs at 5 FPS (parity with the attentive probes);
    # logits are stride-subsampled from the native extraction FPS before decoding.
    eval_fps = 5.0
    stride = max(1, int(round(fps / eval_fps)))
    all_detections: List[List[Dict]] = []
    all_ground_truths: List[List[Dict]] = []
    all_seq_lens: List[int] = []

    for clip_idx, clip_id in enumerate(tqdm(test_clips, desc="test clips")):
        feat_path = _feature_path(feature_cache_dir, clip_id, backbone_id, fps)
        cls = load_clip_cls(feat_path)
        labels = build_frame_labels(label_dir / f"{clip_id}.json")

        # Source/extraction-FPS mismatch -> use the shorter length so indexing is safe.
        T = min(len(cls), len(labels))
        cls = cls[:T]
        labels = labels[:T]

        logits = infer_clip(model, cls, device)
        all_logits.append(logits)
        all_labels.append(labels)
        labels_by_clip[clip_id] = labels
        logits_by_clip[clip_id] = logits

        # Event detections at 5 FPS: stride-subsample logits, then segment-decode.
        logits_eval = logits[::stride]
        T_eval = logits_eval.shape[0]
        mask_eval = np.ones(T_eval, dtype=np.float32)
        detections = postprocess_clip(
            logits=logits_eval,
            mask=mask_eval,
            num_classes=3,
            fps=eval_fps,
            method="segment",
            labeling_mode="interval",
            min_segment_frames=args.min_segment_frames,
            min_confidence=args.min_confidence,
            nms=args.nms,
        )
        for det in detections:
            det["clip_id"] = clip_idx
        all_detections.append(detections)

        gt_centers = extract_gt_events(label_dir / f"{clip_id}.json")
        gt_events = [
            {
                "class": gt["class"],
                "frame": min(int(round(gt["frame"] / stride)), T_eval - 1),
                "timestamp_sec": min(int(round(gt["frame"] / stride)), T_eval - 1) / eval_fps,
                "clip_id": clip_idx,
            }
            for gt in gt_centers
        ]
        all_ground_truths.append(gt_events)
        all_seq_lens.append(T_eval)

    # --- Frame-level on the natural distribution ------------------------
    print("\n[2/4] Frame-level metrics (natural distribution)")
    concat_preds = np.concatenate([logits.argmax(axis=1) for logits in all_logits])
    concat_labels = np.concatenate(all_labels)

    overall_acc = float((concat_preds == concat_labels).mean())
    per_cls_acc: Dict[int, float] = {}
    for cls in CLASS_ORDER:
        m = concat_labels == cls
        per_cls_acc[cls] = float((concat_preds[m] == cls).mean()) if m.any() else float("nan")

    macro_f1 = float(f1_score(concat_labels, concat_preds, labels=CLASS_ORDER, average="macro"))
    per_cls_f1 = f1_score(concat_labels, concat_preds, labels=CLASS_ORDER, average=None)
    cm = confusion_matrix(concat_labels, concat_preds, labels=CLASS_ORDER)

    print(f"       total frames: {len(concat_labels):,}")
    print(f"       overall accuracy: {overall_acc:.4f}  (note: dominated by background majority)")
    print(f"       macro F1:         {macro_f1:.4f}")
    for cls in CLASS_ORDER:
        n = int((concat_labels == cls).sum())
        f1_cls = float(per_cls_f1[CLASS_ORDER.index(cls)])
        print(f"         {CLASS_NAMES[cls]:14s} n={n:6d}  acc={per_cls_acc[cls]:.4f}  f1={f1_cls:.4f}")

    print_confusion_matrix(cm)
    print("       classification report (precision / recall / f1 / support):")
    report = classification_report(
        concat_labels, concat_preds,
        labels=CLASS_ORDER,
        target_names=[CLASS_NAMES[c] for c in CLASS_ORDER],
        digits=4, zero_division=0,
    )
    for line in report.splitlines():
        print(f"       {line}")
    save_confusion_matrix(cm, run_dir / "confusion_matrix_natural.png", "Confusion matrix (natural test distribution)")

    frame_natural = {
        "n_clips": len(test_clips),
        "n_frames": int(len(concat_labels)),
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "per_class_accuracy": {CLASS_NAMES[c]: per_cls_acc[c] for c in CLASS_ORDER},
        "per_class_f1": {CLASS_NAMES[c]: float(per_cls_f1[CLASS_ORDER.index(c)]) for c in CLASS_ORDER},
        "per_class_frame_count": {CLASS_NAMES[c]: int((concat_labels == c).sum()) for c in CLASS_ORDER},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[c] for c in CLASS_ORDER],
    }
    (run_dir / "eval_frame_natural.json").write_text(json.dumps(frame_natural, indent=2))

    # --- Frame-level on a class-balanced test subsample -----------------
    print("\n[3/4] Frame-level metrics (balanced test subsample)")
    seed_balance = int(cfg["seed_balance"])
    balanced_pool = balance_split(test_clips, labels_by_clip, seed=seed_balance)

    bal_preds = np.empty(len(balanced_pool), dtype=np.int64)
    bal_labels = np.empty(len(balanced_pool), dtype=np.int64)
    for i, (clip_id, frame_idx, cls) in enumerate(balanced_pool):
        bal_preds[i] = int(logits_by_clip[clip_id][frame_idx].argmax())
        bal_labels[i] = int(cls)

    bal_overall_acc = float((bal_preds == bal_labels).mean())
    bal_per_cls_acc: Dict[int, float] = {}
    for cls in CLASS_ORDER:
        m = bal_labels == cls
        bal_per_cls_acc[cls] = float((bal_preds[m] == cls).mean()) if m.any() else float("nan")

    bal_macro_f1 = float(f1_score(bal_labels, bal_preds, labels=CLASS_ORDER, average="macro"))
    bal_per_cls_f1 = f1_score(bal_labels, bal_preds, labels=CLASS_ORDER, average=None)
    bal_cm = confusion_matrix(bal_labels, bal_preds, labels=CLASS_ORDER)

    print(f"       seed_balance: {seed_balance}")
    print(f"       total frames: {len(bal_labels):,}  (class-balanced)")
    print(f"       overall accuracy: {bal_overall_acc:.4f}")
    print(f"       macro F1:         {bal_macro_f1:.4f}")
    for cls in CLASS_ORDER:
        n = int((bal_labels == cls).sum())
        f1_cls = float(bal_per_cls_f1[CLASS_ORDER.index(cls)])
        print(f"         {CLASS_NAMES[cls]:14s} n={n:6d}  acc={bal_per_cls_acc[cls]:.4f}  f1={f1_cls:.4f}")

    print_confusion_matrix(bal_cm)
    print("       classification report (precision / recall / f1 / support):")
    bal_report = classification_report(
        bal_labels, bal_preds,
        labels=CLASS_ORDER,
        target_names=[CLASS_NAMES[c] for c in CLASS_ORDER],
        digits=4, zero_division=0,
    )
    for line in bal_report.splitlines():
        print(f"       {line}")
    save_confusion_matrix(bal_cm, run_dir / "confusion_matrix_balanced.png", "Confusion matrix (balanced test subsample)")

    frame_balanced = {
        "n_clips": len(test_clips),
        "n_frames": int(len(bal_labels)),
        "seed_balance": seed_balance,
        "overall_accuracy": bal_overall_acc,
        "macro_f1": bal_macro_f1,
        "per_class_accuracy": {CLASS_NAMES[c]: bal_per_cls_acc[c] for c in CLASS_ORDER},
        "per_class_f1": {CLASS_NAMES[c]: float(bal_per_cls_f1[CLASS_ORDER.index(c)]) for c in CLASS_ORDER},
        "per_class_frame_count": {CLASS_NAMES[c]: int((bal_labels == c).sum()) for c in CLASS_ORDER},
        "confusion_matrix": bal_cm.tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[c] for c in CLASS_ORDER],
    }
    (run_dir / "eval_frame_balanced.json").write_text(json.dumps(frame_balanced, indent=2))

    # --- Per-misclassification dump (balanced subsample) ----------------
    # One row per misclassified frame in the balanced pool. Used downstream
    # for error analysis and (in k-fold) rolled up by aggregate_kfold_spatial.
    miscls_rows: List[Dict] = []
    for i, (clip_id, frame_idx, _) in enumerate(balanced_pool):
        true_label = int(bal_labels[i])
        pred_label = int(bal_preds[i])
        if pred_label == true_label:
            continue
        miscls_rows.append({
            "clip_id": clip_id,
            "frame_idx": int(frame_idx),
            "time_sec": float(frame_idx) / float(fps),
            "true_label": true_label,
            "pred_label": pred_label,
            "true_class": CLASS_NAMES[true_label],
            "pred_class": CLASS_NAMES[pred_label],
        })
    miscls_path = run_dir / "misclassifications.csv"
    with open(miscls_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "frame_idx", "time_sec",
            "true_label", "pred_label", "true_class", "pred_class",
        ])
        writer.writeheader()
        for row in miscls_rows:
            writer.writerow(row)
    print(f"       wrote {len(miscls_rows)} misclassifications -> {miscls_path.name}")

    # --- Event-level: SoccerNet Average-mAP at 5 FPS --------------------
    print(f"\n[4/4] Event-level Average-mAP at {eval_fps} FPS (stride={stride} on {fps}-FPS logits)")
    n_pred = sum(len(d) for d in all_detections)
    n_gt = sum(len(g) for g in all_ground_truths)
    print(f"       predicted events: {n_pred}   ground-truth events: {n_gt}")

    event_results = evaluate_average_map(
        all_detections=all_detections,
        all_ground_truths=all_ground_truths,
        all_seq_lens=all_seq_lens,
        num_classes=3,
        fps=eval_fps,
        metric=args.metric,
        class_names=ACTION_CLASS_NAMES,
        verbose=True,
    )

    # Persist a compact, JSON-friendly version.
    serializable_event_results = {
        "eval_fps": eval_fps,
        "source_fps": float(fps),
        "stride": stride,
        "metric": event_results["metric"],
        "tolerances_sec": [float(t) for t in event_results["tolerances"]],
        "average_mAP": float(event_results["average_mAP"]),
        "per_class_avg_ap": {
            ACTION_CLASS_NAMES[int(k)]: float(v)
            for k, v in event_results["per_class_avg_ap"].items()
        },
        "per_tolerance": [
            {
                "tolerance_sec": float(item["tolerance_sec"]),
                "mAP": float(item["mAP"]),
                "per_class_ap": {
                    ACTION_CLASS_NAMES[i]: float(item["per_class_ap"][i])
                    for i in range(len(ACTION_CLASS_NAMES))
                },
            }
            for item in event_results["per_tolerance"]
        ],
        "n_predicted_events": int(n_pred),
        "n_ground_truth_events": int(n_gt),
        "postprocess": {
            "method": "segment",
            "min_segment_frames": args.min_segment_frames,
            "min_confidence": args.min_confidence,
            "nms": args.nms,
        },
    }
    (run_dir / "eval_events.json").write_text(json.dumps(serializable_event_results, indent=2))

    print(f"\n[done] outputs written to {run_dir}")
    print(f"        eval_frame_natural.json")
    print(f"        eval_frame_balanced.json")
    print(f"        eval_events.json")
    print(f"        confusion_matrix_natural.png")
    print(f"        confusion_matrix_balanced.png")

    # --- Head-only efficiency profile (always on when CUDA is available) ---
    if torch.cuda.is_available():
        print("\n[profile] Head-only efficiency (bs=16, fp32)")
        # Lazily yield per-clip CLS features (already cached on disk; same files
        # consumed by the normal eval loop). Each yielded tensor is a chunk of
        # rows shaped [B_src, D]; the profiler re-chunks to bs=16 internally.
        def _feature_batch_iter():
            for clip_id in test_clips:
                feat_path = _feature_path(feature_cache_dir, clip_id, backbone_id, fps)
                cls = load_clip_cls(feat_path)
                if cls.shape[0] == 0:
                    continue
                yield torch.from_numpy(cls.astype(np.float32))

        csv_path = config.RESULTS_DIR / "head_efficiency.csv"
        prof = profile_head(
            head=model,
            feature_batch_iter=_feature_batch_iter(),
            pipeline="dinov3_linear",
            csv_path=csv_path,
        )
        print(f"  pipeline:                 {prof['pipeline']}")
        print(f"  trainable params (M):     {prof['trainable_params_M']:.6f}")
        print(f"  mean batch latency (ms):  {prof['mean_batch_latency_ms']:.4f}  "
              f"(bs={prof['batch_size']}, warmup={prof['n_warmup']}, timed={prof['n_timed']})")
        print(f"  peak head VRAM (MiB):     {prof['peak_vram_mib']:.2f}")
        print(f"  device:                   {prof['device']}")
        print(f"  appended to:              {prof['csv_path']}")
    else:
        print("\n[profile] Skipped head-only efficiency profile (no CUDA device).")


if __name__ == "__main__":
    main()
