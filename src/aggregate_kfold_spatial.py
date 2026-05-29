"""Aggregate k-fold results for the DINOv3 linear spatial probe.

Reads each fold's eval JSONs under `<base-dir>/fold_*/` and writes a single
`<base-dir>/aggregate.json` plus a printed summary (mean +/- std across folds)
for:

* Event-level Average-mAP and per-tolerance mAP (from eval_events.json)
* Frame-level macro-F1 and per-class precision / recall / F1 / accuracy on the
  natural distribution (from eval_frame_natural.json)
* Same set on the balanced test subsample (from eval_frame_balanced.json)

Per-class precision and recall are derived from the confusion matrix stored in
each eval JSON, so no upstream change is required.

Run from thesis_code/ as:
    python -m src.aggregate_kfold_spatial --base-dir <path>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.labels import CLASS_NAMES_ORDER


def _mean_std(values: List[float]) -> Dict[str, float | List[float]]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "per_fold": [float(v) for v in arr],
    }


def _per_class_metrics_from_cm(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Derive precision, recall, F1, accuracy, and support per class from a
    rows=true / cols=pred confusion matrix."""
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(CLASS_NAMES_ORDER):
        tp = float(cm[i, i])
        row = float(cm[i, :].sum())
        col = float(cm[:, i].sum())
        precision = tp / col if col > 0 else 0.0
        recall = tp / row if row > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / row if row > 0 else 0.0
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "support": row,
        }
    return out


def _aggregate_confusion_matrix(fold_jsons: List[dict]) -> dict:
    """Mean and per-cell std of the confusion matrix across folds."""
    cms = np.array([f["confusion_matrix"] for f in fold_jsons], dtype=np.float64)
    mean = cms.mean(axis=0)
    std = cms.std(axis=0, ddof=1) if cms.shape[0] > 1 else np.zeros_like(mean)
    return {
        "labels": fold_jsons[0]["confusion_matrix_labels"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "per_fold": [[[int(x) for x in row] for row in cm] for cm in cms.astype(int)],
    }


def _aggregate_frame(fold_jsons: List[dict]) -> dict:
    overall_accs = [f["overall_accuracy"] for f in fold_jsons]
    macro_f1s = [f["macro_f1"] for f in fold_jsons]
    per_class: Dict[str, Dict[str, List[float]]] = {
        name: {"precision": [], "recall": [], "f1": [], "accuracy": [], "support": []}
        for name in CLASS_NAMES_ORDER
    }
    for f in fold_jsons:
        cm = np.asarray(f["confusion_matrix"], dtype=np.float64)
        labels = f["confusion_matrix_labels"]
        if labels != CLASS_NAMES_ORDER:
            raise ValueError(
                f"Unexpected confusion_matrix_labels {labels}; expected {CLASS_NAMES_ORDER}."
            )
        derived = _per_class_metrics_from_cm(cm)
        for name in CLASS_NAMES_ORDER:
            for key in per_class[name]:
                per_class[name][key].append(derived[name][key])

    return {
        "overall_accuracy": _mean_std(overall_accs),
        "macro_f1": _mean_std(macro_f1s),
        "per_class": {
            name: {key: _mean_std(values) for key, values in metrics.items()}
            for name, metrics in per_class.items()
        },
        "confusion_matrix": _aggregate_confusion_matrix(fold_jsons),
    }


def _save_mean_cm_plot(cm_agg: dict, output_path: Path, title: str) -> None:
    """Mean confusion matrix with per-cell std annotated ('mean\\n±std')."""
    mean = np.asarray(cm_agg["mean"])
    std = np.asarray(cm_agg["std"])
    labels = cm_agg["labels"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(mean, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            ax.text(j, i, f"{mean[i, j]:.1f}\n±{std[i, j]:.1f}",
                    ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _concatenate_misclassifications(fold_dirs: List[Path], output_csv: Path) -> int:
    """Concat per-fold misclassifications.csv into one, adding a 'fold' column.

    Returns the number of rows written. Folds missing the CSV (e.g. legacy
    runs) are skipped with a warning.
    """
    fieldnames = [
        "fold", "clip_id", "frame_idx", "time_sec",
        "true_label", "pred_label", "true_class", "pred_class",
    ]
    rows_written = 0
    missing: List[str] = []
    with open(output_csv, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for fd in fold_dirs:
            csv_path = fd / "misclassifications.csv"
            if not csv_path.exists():
                missing.append(fd.name)
                continue
            with open(csv_path, newline="") as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    row["fold"] = fd.name
                    writer.writerow(row)
                    rows_written += 1
    if missing:
        print(f"  WARNING: {len(missing)} fold(s) missing misclassifications.csv: "
              f"{', '.join(missing)}")
    return rows_written


def _aggregate_events(fold_jsons: List[dict]) -> dict:
    avg_maps = [f["average_mAP"] for f in fold_jsons]

    # Per-tolerance: tolerances must match across folds.
    tolerance_keys = [tuple(t["tolerance_sec"] for t in f["per_tolerance"]) for f in fold_jsons]
    if len(set(tolerance_keys)) != 1:
        raise ValueError(f"Per-tolerance lists differ across folds: {tolerance_keys}")
    tolerances = tolerance_keys[0]

    per_tolerance = []
    for t_idx, t in enumerate(tolerances):
        maps = [f["per_tolerance"][t_idx]["mAP"] for f in fold_jsons]
        per_class_aps = {
            name: [f["per_tolerance"][t_idx]["per_class_ap"][name] for f in fold_jsons]
            for name in CLASS_NAMES_ORDER if name in fold_jsons[0]["per_tolerance"][t_idx]["per_class_ap"]
        }
        per_tolerance.append({
            "tolerance_sec": float(t),
            "mAP": _mean_std(maps),
            "per_class_ap": {n: _mean_std(v) for n, v in per_class_aps.items()},
        })

    per_class_avg_ap_names = list(fold_jsons[0]["per_class_avg_ap"].keys())
    per_class_avg_ap = {
        name: _mean_std([f["per_class_avg_ap"][name] for f in fold_jsons])
        for name in per_class_avg_ap_names
    }

    return {
        "average_mAP": _mean_std(avg_maps),
        "per_class_avg_ap": per_class_avg_ap,
        "per_tolerance": per_tolerance,
    }


def _fmt(stat: Dict[str, float]) -> str:
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def _print_summary(agg: dict, fold_names: List[str]) -> None:
    print("=" * 70)
    print(f"K-fold aggregate ({len(fold_names)} folds: {', '.join(fold_names)})")
    print("=" * 70)

    print("\nEvent-level (SoccerNet-canonical, tight):")
    print(f"  Average-mAP              : {_fmt(agg['events']['average_mAP'])}")
    print(f"  Per-tolerance mAP:")
    for entry in agg["events"]["per_tolerance"]:
        print(f"    ±{entry['tolerance_sec']:.0f}s                   : {_fmt(entry['mAP'])}")
    print(f"  Per-class Avg-AP:")
    for name, stat in agg["events"]["per_class_avg_ap"].items():
        print(f"    {name:<22} : {_fmt(stat)}")

    for label, key in (("natural distribution", "frame_natural"),
                       ("balanced test subsample", "frame_balanced")):
        print(f"\nFrame-level ({label}):")
        print(f"  Overall accuracy         : {_fmt(agg[key]['overall_accuracy'])}")
        print(f"  Macro F1                 : {_fmt(agg[key]['macro_f1'])}")
        print(f"  {'class':<14}  {'precision':>16}  {'recall':>16}  {'f1':>16}")
        for name in CLASS_NAMES_ORDER:
            pc = agg[key]["per_class"][name]
            print(f"  {name:<14}  {_fmt(pc['precision']):>16}  {_fmt(pc['recall']):>16}  {_fmt(pc['f1']):>16}")
    print("=" * 70)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", type=Path, required=True,
                   help="Directory containing fold_*/ subdirs from a k-fold spatial run.")
    p.add_argument("--output", type=Path, default=None,
                   help="Path to write aggregate.json. Defaults to <base-dir>/aggregate.json.")
    args = p.parse_args()

    base_dir: Path = args.base_dir
    fold_dirs = sorted(d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("fold_"))
    if not fold_dirs:
        raise SystemExit(f"No fold_*/ subdirectories under {base_dir}.")

    natural, balanced, events = [], [], []
    fold_names = []
    for fd in fold_dirs:
        nat_path = fd / "eval_frame_natural.json"
        bal_path = fd / "eval_frame_balanced.json"
        evt_path = fd / "eval_events.json"
        missing = [p for p in (nat_path, bal_path, evt_path) if not p.exists()]
        if missing:
            raise SystemExit(f"Fold {fd.name} is missing {[str(m) for m in missing]}.")
        natural.append(json.loads(nat_path.read_text()))
        balanced.append(json.loads(bal_path.read_text()))
        events.append(json.loads(evt_path.read_text()))
        fold_names.append(fd.name)

    aggregate = {
        "n_folds": len(fold_dirs),
        "fold_names": fold_names,
        "frame_natural": _aggregate_frame(natural),
        "frame_balanced": _aggregate_frame(balanced),
        "events": _aggregate_events(events),
    }

    out_path = args.output or (base_dir / "aggregate.json")
    out_path.write_text(json.dumps(aggregate, indent=2))

    _save_mean_cm_plot(
        aggregate["frame_natural"]["confusion_matrix"],
        base_dir / "confusion_matrix_natural_mean.png",
        f"Mean confusion matrix ({len(fold_dirs)} folds, natural distribution)",
    )
    _save_mean_cm_plot(
        aggregate["frame_balanced"]["confusion_matrix"],
        base_dir / "confusion_matrix_balanced_mean.png",
        f"Mean confusion matrix ({len(fold_dirs)} folds, balanced subsample)",
    )

    miscls_csv = base_dir / "all_misclassifications.csv"
    n_miscls = _concatenate_misclassifications(fold_dirs, miscls_csv)

    _print_summary(aggregate, fold_names)
    print(f"\n[done] aggregate written to {out_path}")
    print(f"        confusion_matrix_natural_mean.png")
    print(f"        confusion_matrix_balanced_mean.png")
    print(f"        all_misclassifications.csv  ({n_miscls} rows)")


if __name__ == "__main__":
    main()
