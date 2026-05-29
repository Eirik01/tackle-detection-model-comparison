"""DINOv3 linear-probe training for the spatial approach (split-then-balance).

Protocol:
1. Game-level 70/15/15 split (seed = --seed-split).
2. Within each split, undersample background and tackle-replay frame pools to
   the tackle-live count for that split (seed = --seed-balance). All
   tackle-live frames are kept.
3. Load DINOv3-L CLS features from the .npz cache for the chosen frames.
4. Train the DINOv3LinearProbe with cross-entropy + SGD (momentum 0.9), the
   DINOv3 linear-probe recipe. Decoupled stop/save: the checkpoint is taken
   at the epoch of maximum validation macro-F1; early-stopping triggers when
   validation loss has not improved for --patience epochs.
5. Evaluate the macro-F1-best checkpoint on the balanced test pool.

Run from tackle-detection-model-comparison/ as:
    python src/train_spatial.py [--flags ...]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import config
from data.labels import CLASS_NAMES, CLASS_ORDER
from data.splits import (
    balance_split,
    build_frame_labels,
    kfold_split_games,
    split_games,
)
from models.dinov3.linear_probe import DINOv3LinearProbe
from utils import set_seed


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def _feature_path(cache_dir: Path, clip_id: str, backbone_id: str, fps: float) -> Path:
    return cache_dir / f"{clip_id}_{backbone_id}_{fps}fps_features.npz"


def load_pool_tensors(
    pool: List[Tuple[str, int, int]],
    cache_dir: Path,
    backbone_id: str,
    fps: float,
    feature_cache: Dict[str, np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load CLS features and labels for a balanced pool into dense tensors.

    `feature_cache` is mutated to avoid re-reading the same .npz across splits.
    """
    feats = np.empty((len(pool), config.FEATURE_DIM), dtype=np.float32)
    labels = np.empty(len(pool), dtype=np.int64)

    for i, (clip_id, frame_idx, cls) in enumerate(pool):
        if clip_id not in feature_cache:
            path = _feature_path(cache_dir, clip_id, backbone_id, fps)
            if not path.exists():
                raise FileNotFoundError(f"Missing feature cache for {clip_id}: {path}")
            with np.load(path) as data:
                feature_cache[clip_id] = data["cls"].astype(np.float32)

        cls_array = feature_cache[clip_id]
        if frame_idx >= len(cls_array):
            raise IndexError(
                f"{clip_id}: frame_idx {frame_idx} out of bounds for CLS array of length "
                f"{len(cls_array)}. Source-FPS / extraction-FPS mismatch?"
            )
        feats[i] = cls_array[frame_idx]
        labels[i] = cls

    return torch.from_numpy(feats), torch.from_numpy(labels)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """One pass over `loader`. If `optimizer` is None, runs in eval mode."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_n = 0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.set_grad_enabled(train_mode):
        for feats, targets in loader:
            feats = feats.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(feats)
            loss = criterion(logits, targets)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * feats.size(0)
            total_n += feats.size(0)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    acc = float((preds == targets).mean())
    return total_loss / total_n, acc, preds, targets


def per_class_accuracy(targets: np.ndarray, preds: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for cls in CLASS_ORDER:
        mask = targets == cls
        out[cls] = float((preds[mask] == cls).mean()) if mask.any() else float("nan")
    return out


def print_confusion_matrix(cm: np.ndarray, indent: str = "       ") -> None:
    names = [CLASS_NAMES[c] for c in CLASS_ORDER]
    col_w = max(14, max(len(n) for n in names) + 2)
    row_label_w = max(len(n) for n in names)
    print(f"{indent}confusion matrix (rows=true, cols=pred):")
    print(f"{indent}{'':<{row_label_w}}" + "".join(f"{n:>{col_w}}" for n in names))
    for i, name in enumerate(names):
        print(f"{indent}{name:<{row_label_w}}" + "".join(f"{cm[i, j]:>{col_w}d}" for j in range(len(names))))


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels([CLASS_NAMES[c] for c in CLASS_ORDER], rotation=30, ha="right")
    ax.set_yticklabels([CLASS_NAMES[c] for c in CLASS_ORDER])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Seeds
    p.add_argument("--seed-split", type=int, default=42, help="Seed for game-level 70/15/15 split.")
    p.add_argument("--seed-balance", type=int, default=42, help="Seed for per-split undersampling.")
    p.add_argument("--seed-train", type=int, default=42, help="Seed for model init + dataloader shuffle.")

    # Data
    p.add_argument("--label-dir", type=Path, default=config.TACDEC_LABELS, help="Directory of TACDEC label JSONs.")
    p.add_argument("--feature-cache-dir", type=Path, default=config.TACDEC_FEATURES_DINOV3, help="Directory of cached DINOv3 features.")
    p.add_argument("--backbone-id", type=str, default="dinov3_l", help="Backbone identifier in feature filenames.")
    p.add_argument("--fps", type=float, default=25.0, help="Extraction FPS used in feature filenames.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--n-folds", type=int, default=None,
                   help="If set together with --fold-idx, replace the 70/15/15 split with a "
                        "k-fold partition: shuffle games with --seed-split, chunk into "
                        "n_folds blocks, hold out block --fold-idx as test. val_frac is "
                        "carved from the remainder.")
    p.add_argument("--fold-idx", type=int, default=None,
                   help="Test-fold index in [0, n_folds). Requires --n-folds.")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9,
                   help="SGD momentum. Linear probing follows the DINOv3 recipe: "
                        "SGD + momentum 0.9 (the LR grid runs up to 5.0, which only "
                        "behaves under SGD, not AdamW).")
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stop on val_loss plateau (epochs without "
                        "val_loss improvement). Checkpoint and selection key "
                        "on val macro-F1; this only halts training.")
    p.add_argument("--num-workers", type=int, default=0)

    # Output
    p.add_argument("--output-dir", type=Path, default=None, help="Default: <TACDEC_RESULTS>/dinov3_linear_spatial/<timestamp>.")
    p.add_argument("--run-name", type=str, default=None, help="Override the auto-generated run name.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if (args.n_folds is None) != (args.fold_idx is None):
        raise SystemExit("--n-folds and --fold-idx must be set together (or both omitted).")

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or (config.TACDEC_RESULTS / "dinov3_linear_spatial" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed (and set CUBLAS_WORKSPACE_CONFIG) before any CUDA touch: the device
    # line below calls torch.cuda.is_available(), which can initialise the CUDA
    # context, after which CUBLAS_WORKSPACE_CONFIG no longer takes effect.
    set_seed(args.seed_train)
    device = torch.device(config.DEVICE if torch.cuda.is_available() or config.DEVICE == "cpu" else "cpu")

    print(f"[setup] output_dir: {output_dir}")
    print(f"[setup] device: {device}")
    print(f"[setup] seeds: split={args.seed_split}, balance={args.seed_balance}, train={args.seed_train}")
    if args.n_folds is not None:
        print(f"[setup] k-fold: fold {args.fold_idx + 1}/{args.n_folds} (val_frac={args.val_frac})")

    # --- 1. Split ---------------------------------------------------------
    if args.n_folds is not None:
        print(f"\n[1/4] Game-level k-fold split (fold {args.fold_idx + 1}/{args.n_folds})")
        splits = kfold_split_games(
            args.label_dir,
            n_folds=args.n_folds,
            fold_idx=args.fold_idx,
            val_frac=args.val_frac,
            seed=args.seed_split,
        )
    else:
        print("\n[1/4] Game-level split")
        splits = split_games(args.label_dir, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed_split)
    for name in ("train", "val", "test"):
        print(f"       {name}: {len(splits[name])} clips")

    # --- 2. Frame labels + balance ---------------------------------------
    print("\n[2/4] Building frame labels")
    labels_by_clip: Dict[str, np.ndarray] = {}
    for clip_id in splits["train"] + splits["val"] + splits["test"]:
        label_path = args.label_dir / f"{clip_id}.json"
        labels_by_clip[clip_id] = build_frame_labels(label_path)

    print("\n[2/4] Balancing each split")
    pools: Dict[str, List[Tuple[str, int, int]]] = {}
    for name in ("train", "val", "test"):
        pool = balance_split(splits[name], labels_by_clip, seed=args.seed_balance)
        pools[name] = pool
        counts = Counter(c for _, _, c in pool)
        per_class = ", ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in CLASS_ORDER)
        print(f"       {name}: {len(pool):,} frames  ({per_class})")

    # --- 3. Load features ------------------------------------------------
    print("\n[3/4] Loading CLS features")
    feature_cache: Dict[str, np.ndarray] = {}
    tensors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name in ("train", "val", "test"):
        t0 = time.time()
        tensors[name] = load_pool_tensors(pools[name], args.feature_cache_dir, args.backbone_id, args.fps, feature_cache)
        print(f"       {name}: features={tuple(tensors[name][0].shape)} in {time.time() - t0:.1f}s")

    # Explicit seeded generator so the train shuffle order is reproducible
    # independently of how many global-RNG draws happened before this point
    # (e.g. probe weight init), rather than relying on global RNG ordering.
    train_gen = torch.Generator()
    train_gen.manual_seed(args.seed_train)
    loaders = {
        "train": DataLoader(TensorDataset(*tensors["train"]), batch_size=args.batch_size, shuffle=True, generator=train_gen, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "val": DataLoader(TensorDataset(*tensors["val"]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
        "test": DataLoader(TensorDataset(*tensors["test"]), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"),
    }

    # --- 4. Train --------------------------------------------------------
    print("\n[4/4] Training")
    model = DINOv3LinearProbe(embed_dim=config.FEATURE_DIM, num_classes=3).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Decoupled stop/save: checkpoint on val macro-F1, early-stop on val_loss.
    # Loss is a smoother plateau signal; macro-F1 is the metric we report.
    best_val_loss = float("inf")
    best_val_loss_epoch = -1
    f1_at_best_val_loss = -1.0
    best_val_acc_at_best_f1 = -1.0
    val_loss_at_best_f1 = float("inf")
    best_val_macro_f1 = -1.0
    best_val_macro_f1_epoch = -1
    epochs_since_improve = 0
    best_state: dict[str, torch.Tensor] = {}
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_targets = run_epoch(model, loaders["val"], criterion, None, device)
        val_macro_f1 = float(f1_score(val_targets, val_preds, labels=CLASS_ORDER, average="macro"))
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_macro_f1})

        f1_improved = val_macro_f1 > best_val_macro_f1
        loss_improved = val_loss < best_val_loss
        marker = ""
        if f1_improved:
            marker += "  *new best F1"
        if loss_improved:
            marker += "  *new best loss"
        print(f"       epoch {epoch:3d}  train {train_loss:.4f}/{train_acc:.4f}  val {val_loss:.4f}/{val_acc:.4f}  val_f1 {val_macro_f1:.4f}{marker}")

        if f1_improved:
            best_val_macro_f1 = val_macro_f1
            best_val_macro_f1_epoch = epoch
            val_loss_at_best_f1 = val_loss
            best_val_acc_at_best_f1 = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if loss_improved:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            f1_at_best_val_loss = val_macro_f1
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"       early stop at epoch {epoch} (no val_loss improvement for {args.patience})")
                break

    # --- Final test ------------------------------------------------------
    print("\n[test] Loading best checkpoint and evaluating")
    model.load_state_dict(best_state)
    test_loss, test_acc, test_preds, test_targets = run_epoch(model, loaders["test"], criterion, None, device)
    per_cls_acc = per_class_accuracy(test_targets, test_preds)
    cm = confusion_matrix(test_targets, test_preds, labels=CLASS_ORDER)
    f1_macro = float(f1_score(test_targets, test_preds, labels=CLASS_ORDER, average="macro"))
    f1_per_class = f1_score(test_targets, test_preds, labels=CLASS_ORDER, average=None)

    print(f"       test loss/acc: {test_loss:.4f} / {test_acc:.4f}")
    print(f"       per-class accuracy:")
    for cls in CLASS_ORDER:
        print(f"         {CLASS_NAMES[cls]:14s} {per_cls_acc[cls]:.4f}   f1={f1_per_class[CLASS_ORDER.index(cls)]:.4f}")
    print(f"       macro F1: {f1_macro:.4f}")
    print_confusion_matrix(cm)
    print("       classification report (precision / recall / f1 / support):")
    test_report = classification_report(
        test_targets, test_preds,
        labels=CLASS_ORDER,
        target_names=[CLASS_NAMES[c] for c in CLASS_ORDER],
        digits=4, zero_division=0,
    )
    for line in test_report.splitlines():
        print(f"       {line}")

    # --- Persist outputs --------------------------------------------------
    (output_dir / "config.json").write_text(json.dumps({
        "seed_split": args.seed_split,
        "seed_balance": args.seed_balance,
        "seed_train": args.seed_train,
        "label_dir": str(args.label_dir),
        "feature_cache_dir": str(args.feature_cache_dir),
        "backbone_id": args.backbone_id,
        "fps": args.fps,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_folds": args.n_folds,
        "fold_idx": args.fold_idx,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "sgd",
        "momentum": args.momentum,
        "patience": args.patience,
        "feature_dim": config.FEATURE_DIM,
    }, indent=2))

    (output_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    (output_dir / "balanced_pool.json").write_text(json.dumps(
        {name: [list(t) for t in pools[name]] for name in pools}, indent=2
    ))
    (output_dir / "metrics.json").write_text(json.dumps({
        "history": history,
        "selection_metric": "val_macro_f1",
        "best_epoch": best_val_macro_f1_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_macro_f1_epoch": best_val_macro_f1_epoch,
        "val_loss_at_best_f1": val_loss_at_best_f1,
        "best_val_acc_at_best_f1": best_val_acc_at_best_f1,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "f1_at_best_val_loss": f1_at_best_val_loss,
        "test": {
            "loss": test_loss,
            "accuracy": test_acc,
            "macro_f1": f1_macro,
            "per_class_accuracy": {CLASS_NAMES[c]: per_cls_acc[c] for c in CLASS_ORDER},
            "per_class_f1": {CLASS_NAMES[c]: float(f1_per_class[CLASS_ORDER.index(c)]) for c in CLASS_ORDER},
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": [CLASS_NAMES[c] for c in CLASS_ORDER],
        },
    }, indent=2))

    save_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    torch.save(best_state, output_dir / "model.pt")

    print(f"\n[done] outputs written to {output_dir}")


if __name__ == "__main__":
    main()
