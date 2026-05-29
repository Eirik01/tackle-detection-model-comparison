"""Evaluate the DINOv3 linear (spatial) probe on the centre frames of the
temporal test windows.

The spatial probe's default test set (`eval_spatial.py`) is a balanced 4 725-
frame pool sampled per clip. The two attentive probes (DINOv3 and V-JEPA 2)
are tested on 207 centred W=10 windows, with the centre frame's label as the
window label. Those two test sets share neither items nor size, so the
headline F1 numbers cannot be compared directly.

This script rebuilds the same 207 windows the attentive probes see at test
time and evaluates the spatial probe on the centre frame of each window. The
centre frame at the 25 FPS extraction grid is `anchor_5fps * stride`, where
`stride = round(extraction_fps / target_fps)` matches the temporal pipeline's
convention. All other model and data settings mirror the existing spatial /
temporal evals, so the produced JSON can be paired with a temporal
`eval_test.json` for a window-by-window analysis.

Run from thesis_code/ as:
    uv run python -u src/eval_spatial_centred.py \\
        --spatial-checkpoint <path-to-model.pt> \\
        --seed 42 \\
        --save-json results/spatial/<run>_centred_test.json

Use the file-path invocation (not `python -m src.X`) so that ``src/`` is on
``sys.path`` and the absolute ``from config import ...`` style used by the
sibling train/eval modules resolves. This matches how ``train_temporal.py``
and ``eval_temporal.py`` are run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config
from data.balanced_temporal_dataset import get_balanced_temporal_dataloaders
from data.labels import CLASS_NAMES, CLASS_ORDER
from models.dinov3.linear_probe import DINOv3LinearProbe
from utils import set_seed


def _feature_path(cache_dir: Path, clip_id: str, backbone_id: str, fps: float) -> Path:
    return cache_dir / f"{clip_id}_{backbone_id}_{fps}fps_features.npz"


def _load_cls_array(cache: Dict[str, np.ndarray], cache_dir: Path, clip_id: str,
                    backbone_id: str, extraction_fps: float) -> np.ndarray:
    if clip_id not in cache:
        path = _feature_path(cache_dir, clip_id, backbone_id, extraction_fps)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing CLS feature cache for {clip_id}: {path}"
            )
        with np.load(path) as data:
            cache[clip_id] = data["cls"].astype(np.float32)
    return cache[clip_id]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--spatial-checkpoint", type=Path, required=True,
                    help="Path to the trained spatial probe state_dict (model.pt).")
    ap.add_argument("--labels-dir", type=Path, default=config.TACDEC_LABELS,
                    help="Directory of per-clip label JSON files.")
    ap.add_argument("--feature-cache-dir", type=Path,
                    default=config.TACDEC_FEATURES_DINOV3,
                    help="Directory of cached DINOv3 CLS features (.npz).")
    ap.add_argument("--backbone-id", type=str, default="dinov3_l",
                    help="Backbone identifier embedded in CLS feature filenames.")
    ap.add_argument("--extraction-fps", type=float, default=25.0,
                    help="FPS encoded in the CLS feature filename "
                         "(`{clip}_{backbone}_{fps}fps_features.npz`).")
    ap.add_argument("--target-fps", type=float, default=5.0,
                    help="FPS of the temporal anchor index. The centre frame "
                         "at extraction FPS is anchor * round(extraction_fps "
                         "/ target_fps).")
    ap.add_argument("--window-size", type=int, default=10,
                    help="Must match the temporal protocol the attentive "
                         "probes were tested with.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--padding-mode", choices=["center_crop", "reflect"],
                    default="reflect",
                    help="Selects the *_reflect_dense_* extraction flavour "
                         "used by the temporal probes. Only affects which "
                         "windows pass the temporal pipeline's validity "
                         "checks; the centre-frame CLS features are the same.")
    ap.add_argument("--save-json", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Reuse the temporal test split. We won't iterate the dense features,
    #    only the window metadata (video_id, anchor, label).
    print("[1/3] Resolving the temporal test windows ...")
    features_dir = config.TACDEC_FEATURES / "dinov3_large"
    dense_tag = "reflect" if args.padding_mode == "reflect" else ""

    # Diagnostic: confirm we see the same splits the temporal training used.
    from data.splits import split_games  # noqa: WPS433
    _splits_check = split_games(args.labels_dir, val_frac=0.15,
                                test_frac=0.15, seed=args.seed)
    print(f"       labels_dir = {args.labels_dir}")
    print(f"       (exists: {args.labels_dir.exists()})")
    print(f"       split sizes (clips): "
          + ", ".join(f"{k}={len(v)}" for k, v in _splits_check.items()))
    if any(len(v) == 0 for v in _splits_check.values()):
        raise SystemExit(
            "One or more splits is empty. The labels_dir likely contains "
            "fewer JSON files than the training pipeline saw. Pass "
            "--labels-dir explicitly to point at the same directory the "
            "temporal training used."
        )

    _, _, _, info = get_balanced_temporal_dataloaders(
        labels_dir=args.labels_dir,
        features_dir=features_dir,
        backbone="dinov3",
        window_size=args.window_size,
        target_fps=args.target_fps,
        source_fps=args.extraction_fps,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=0,
        feature_loader_cache=1,
        dense_tag=dense_tag,
    )
    test_windows: List[dict] = info["_splits"]["test"]
    counts = {CLASS_NAMES[c]: 0 for c in CLASS_ORDER}
    for w in test_windows:
        counts[CLASS_NAMES[int(w["class"])]] += 1
    print(f"       {len(test_windows)} windows  ::  "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))

    # 2. Load the spatial probe.
    print(f"[2/3] Loading spatial probe from {args.spatial_checkpoint}")
    model = DINOv3LinearProbe(embed_dim=config.FEATURE_DIM, num_classes=3)
    state = torch.load(args.spatial_checkpoint, map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    # 3. Centre-frame inference.
    print("[3/3] Predicting on centre frames ...")
    stride = int(round(args.extraction_fps / args.target_fps))
    if abs(stride - args.extraction_fps / args.target_fps) > 1e-6:
        raise ValueError(
            f"extraction_fps/target_fps must be integer; got "
            f"{args.extraction_fps}/{args.target_fps} = "
            f"{args.extraction_fps / args.target_fps}"
        )
    cache: Dict[str, np.ndarray] = {}
    feats_batch: List[np.ndarray] = []
    labels_batch: List[int] = []
    preds: List[int] = []
    labels: List[int] = []
    video_ids: List[str] = []
    anchors_5fps: List[int] = []

    def flush() -> None:
        if not feats_batch:
            return
        x = torch.from_numpy(np.stack(feats_batch)).to(device)
        with torch.no_grad():
            logits = model(x)
            p = logits.argmax(dim=-1).cpu().numpy()
        preds.extend(int(v) for v in p)
        labels.extend(labels_batch)
        feats_batch.clear()
        labels_batch.clear()

    for w in test_windows:
        clip_id = str(w["video_id"])
        anchor = int(w["anchor"])
        label = int(w["class"])
        frame_idx_25 = anchor * stride
        cls_arr = _load_cls_array(cache, args.feature_cache_dir, clip_id,
                                  args.backbone_id, args.extraction_fps)
        if frame_idx_25 >= len(cls_arr):
            raise IndexError(
                f"{clip_id}: anchor={anchor} (5 FPS) -> {frame_idx_25} "
                f"(25 FPS) OOB for CLS array of length {len(cls_arr)}."
            )
        feats_batch.append(cls_arr[frame_idx_25])
        labels_batch.append(label)
        video_ids.append(clip_id)
        anchors_5fps.append(anchor)
        if len(feats_batch) >= args.batch_size:
            flush()
    flush()

    preds_np = np.asarray(preds, dtype=np.int64)
    labels_np = np.asarray(labels, dtype=np.int64)

    macro_f1 = float(f1_score(labels_np, preds_np, labels=CLASS_ORDER,
                              average="macro", zero_division=0))
    f1_per_class = f1_score(labels_np, preds_np, labels=CLASS_ORDER,
                            average=None, zero_division=0)
    cm = confusion_matrix(labels_np, preds_np, labels=CLASS_ORDER)
    per_cls_acc: Dict[int, float] = {}
    for c in CLASS_ORDER:
        mask = labels_np == c
        per_cls_acc[c] = (float((preds_np[mask] == c).mean())
                          if mask.any() else float("nan"))
    overall_acc = float((preds_np == labels_np).mean())

    print()
    print(f"       n_windows = {len(test_windows)}")
    print(f"       accuracy  = {overall_acc:.4f}")
    print(f"       macro F1  = {macro_f1:.4f}")
    print("       per-class accuracy / f1:")
    for c in CLASS_ORDER:
        print(f"         {CLASS_NAMES[c]:14s}  acc={per_cls_acc[c]:.4f}  "
              f"f1={f1_per_class[CLASS_ORDER.index(c)]:.4f}")
    print("       confusion matrix (rows=true, cols=pred):")
    header = " " * 16 + "".join(f"{CLASS_NAMES[c]:>15s}" for c in CLASS_ORDER)
    print(header)
    for i, c in enumerate(CLASS_ORDER):
        row = "".join(f"{int(cm[i, j]):>15d}" for j in range(len(CLASS_ORDER)))
        print(f"       {CLASS_NAMES[c]:13s} {row}")
    print()
    print(classification_report(labels_np, preds_np, labels=CLASS_ORDER,
                                target_names=[CLASS_NAMES[c] for c in CLASS_ORDER],
                                digits=4, zero_division=0))

    out = {
        "spatial_checkpoint": str(args.spatial_checkpoint),
        "test_protocol": "temporal-centred (anchor_5fps centre frame)",
        "n_windows": len(test_windows),
        "seed": args.seed,
        "window_size": args.window_size,
        "extraction_fps": args.extraction_fps,
        "target_fps": args.target_fps,
        "stride": stride,
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "per_class_accuracy": {CLASS_NAMES[c]: per_cls_acc[c] for c in CLASS_ORDER},
        "per_class_f1": {CLASS_NAMES[c]: float(f1_per_class[CLASS_ORDER.index(c)])
                         for c in CLASS_ORDER},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [CLASS_NAMES[c] for c in CLASS_ORDER],
        "predictions": preds_np.tolist(),
        "labels": labels_np.tolist(),
        "video_ids": video_ids,
        "anchor_5fps": anchors_5fps,
    }
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(out, indent=2))
    print(f"[done] {args.save_json}")

    # Misclassifications CSV — mirrors the temporal probe schema so the three
    # probes' per-window errors can be diffed directly on the same 207 events.
    miscls_path = args.save_json.with_name(
        args.save_json.stem + ".misclassifications.csv"
    )
    n_miscls = 0
    with open(miscls_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "frame_idx", "time_sec",
            "true_label", "pred_label", "true_class", "pred_class",
        ])
        writer.writeheader()
        for i in range(len(labels_np)):
            true_label = int(labels_np[i])
            pred_label = int(preds_np[i])
            if pred_label == true_label:
                continue
            anchor = int(anchors_5fps[i])
            writer.writerow({
                "clip_id": video_ids[i],
                "frame_idx": anchor,
                "time_sec": float(anchor) / float(args.target_fps),
                "true_label": true_label,
                "pred_label": pred_label,
                "true_class": CLASS_NAMES[true_label],
                "pred_class": CLASS_NAMES[pred_label],
            })
            n_miscls += 1
    print(f"[done] wrote {n_miscls} misclassifications -> {miscls_path}")


if __name__ == "__main__":
    main()
