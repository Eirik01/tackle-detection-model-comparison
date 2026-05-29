"""
Train a paper-faithful attentive probe head on a frozen backbone.

Two probes are supported, one per backbone, each its own paper-faithful recipe:

  --backbone-type dinov3
      head = DINOv3AttentiveProbe (linear proj -> 3D sin-cos PE -> 4 SA blocks
             with 3D factorized RoPE + per-head random rotations -> 1 CA block
             with single learnable query -> linear classifier).
      input = patch tokens, gathered W consecutive frames per window:
              [W * num_patches, D].

  --backbone-type vjepa2
      head = bare AttentiveClassifier (V-JEPA2 paper recipe: 3 SA + 1 CA blocks,
             single learnable query, no RoPE, no input projection).
      input = spatio-temporal token grid emitted by one V-JEPA2 forward over
              the W-frame window: [N_tokens, D].

Training: per-window cross-entropy on the center-frame label, AdamW + cosine
schedule, optional inverse-frequency class weights (default on, train-split
only). Game-disjoint 70/15/15 split shared with the spatial probe
(splits.split_games).

Training-sampler protocol (selected via --protocol):
  'centered' (default)           : one window per event, class-balanced via
      build_balanced_windows in src/data/temporal_protocol.py. Backs the
      headline three-pipeline comparison.
  'kassab_concat' (DINOv3 only)  : strict Kassab TempTAC parity at 5 FPS:
      per-sequence retention rule (5-frame bg slice, whole tackle sequences)
      plus cross-clip concat-and-slide. Auxiliary parity experiment.

Usage:
  uv run python src/train_temporal.py \
      --backbone-type dinov3 --window-size 10 --fps 5.0 \
      --protocol centered \
      --num-epochs 30 --batch-size 64 --learning-rate 1e-4 \
      --model-suffix centered_v1 --seed 42 \
      --save-info results/temporal/dinov3_l_centered_v1_train.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from config import TACDEC_FEATURES, TACDEC_LABELS, TACDEC_MODELS
from data.balanced_temporal_dataset import (
    get_balanced_temporal_dataloaders,
)
from data.temporal_loaders import (
    CLASS_NAMES,
    compute_class_weights,
)
from data.kassab_concat_dataset import (
    get_kassab_concat_temporal_dataloaders,
)
from models.dinov3.attentive_probe import DINOv3AttentiveProbe
from models.vjepa2.attentive_pooler import AttentiveClassifier
from utils import set_seed


def forward_probe(model, backbone: str, features: torch.Tensor) -> torch.Tensor:
    """
    Both probes accept ``[B, N_tokens, D]`` and return ``[B, num_classes]``.
    Single dispatch so train and eval share the same forward path.
    """
    if backbone == "dinov3":
        return model(features)
    if backbone == "vjepa2":
        return model(features)
    raise ValueError(f"unknown backbone {backbone!r}")


# ---- Train / val loops ------------------------------------------------------


def evaluate(model, backbone, loader, device, criterion, num_classes):
    model.eval()
    total_loss, n_batches = 0.0, 0
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = forward_probe(model, backbone, feats)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    targets = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    macro_f1 = float(
        f1_score(targets, preds, labels=list(range(num_classes)),
                 average="macro", zero_division=0)
    )
    return total_loss / max(n_batches, 1), correct / max(total, 1), macro_f1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backbone-type", choices=["dinov3", "vjepa2"], required=True)
    ap.add_argument("--backbone-size", default="large",
                    choices=["base", "large", "huge", "giant"])
    ap.add_argument("--num-classes", type=int, default=3)

    # Data / window
    ap.add_argument("--fps", type=float, default=5.0,
                    help="Target (effective) FPS the dataset operates at (default 5.0). "
                         "Shared 5 FPS protocol => exact integer stride from 25 FPS source.")
    ap.add_argument("--source-fps", type=float, default=None,
                    help="DINOv3 only: FPS embedded in the on-disk feature filename. "
                         "If different from --fps, the loader stride-indexes the "
                         "source-FPS file so re-extraction isn't needed. Default = "
                         "--fps. Typical: --fps 5.0 --source-fps 25.0 to read the "
                         "existing 25 FPS DINOv3 dense files at a 5 FPS effective rate.")
    ap.add_argument("--window-size", type=int, default=10,
                    help="W: frames per window (5 FPS * 2 s = 10 default; even, required "
                         "by V-JEPA2 tubelet=2).")
    ap.add_argument("--bg-count", type=int, default=500,
                    help="'kassab_concat' protocol: number of random background "
                         "sequences sampled (without replacement).")
    ap.add_argument("--replay-cap", type=int, default=280,
                    help="'kassab_concat' protocol: first-come cap on tackle-replay "
                         "sequences (in data order).")
    ap.add_argument("--protocol",
                    choices=["centered", "kassab_concat"],
                    default="centered",
                    help="Training sampler. 'centered' (default) = one window per "
                         "event, class-balanced (build_balanced_windows); backs the "
                         "headline three-pipeline comparison. 'kassab_concat' = "
                         "strict Kassab TempTAC parity at 5 FPS: per-sequence "
                         "retention rule (5-frame bg slice, whole tackle sequences) "
                         "+ cross-clip concat-and-slide. DINOv3-only; auxiliary "
                         "parity experiment.")

    # Probe (DINOv3 paper recipe; reused for both backbones where applicable).
    ap.add_argument("--probe-num-heads", type=int, default=16)
    ap.add_argument("--probe-depth", type=int, default=4)
    ap.add_argument("--patch-h", type=int, default=16,
                    help="DINOv3 only: spatial patch grid height (256/patch_size).")
    ap.add_argument("--patch-w", type=int, default=16,
                    help="DINOv3 only: spatial patch grid width.")

    # Optim
    ap.add_argument("--num-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5,
                    help="Early-stop on val_loss plateau (epochs without "
                         "val_loss improvement). Checkpoint and ranking key on "
                         "val macro-F1 -- this only halts training.")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--ce-weight-style",
                    choices=["min1", "balanced"],
                    default="min1",
                    help="Inverse-frequency CE weight normalisation. 'min1' "
                         "(default) divides by min so the smallest weight is "
                         "1.0. 'balanced' matches sklearn's "
                         "compute_class_weight('balanced', ...) exactly -- "
                         "use this for Kassab TempTAC parity.")

    # Loading / IO
    ap.add_argument("--num-workers", type=int, default=0,
                    help="Per-loader workers. The lazy feature cache is per-loader, "
                         "so each worker re-builds it; keep 0 unless you have RAM "
                         "and the dataset throughput is the bottleneck.")
    ap.add_argument("--feature-cache", type=int, default=4,
                    help="LRU size (videos held in memory) per loader.")
    ap.add_argument("--model-suffix", default="centered")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-info", type=str, default=None)
    ap.add_argument("--padding-mode", choices=["center_crop", "reflect"],
                    default="center_crop",
                    help="Which extraction flavour to load: 'center_crop' uses "
                         "the default dense files; 'reflect' loads the "
                         "*_reflect_dense_* files written by extract_features.py "
                         "--padding-mode reflect.")
    ap.add_argument("--split-file", type=str, default=None,
                    help="Optional JSON file with explicit train/val/test "
                         "clip-ID lists (schema: {train: [...], val: [...], "
                         "test: [...]}). Overrides the seeded game-disjoint "
                         "split entirely. Use scripts/dump_kassab_split.py to "
                         "produce the Kassab TempTAC video-level split for "
                         "apples-to-apples comparison.")
    ap.add_argument("--split-mode",
                    choices=["kassab_bug", "correct"],
                    default="kassab_bug",
                    help="kassab_concat protocol only. 'kassab_bug' (default) "
                         "replicates Kassab TempTAC's extract_data flat-index "
                         "slice bug so the train/val/test pools positionally "
                         "match his reported eval pool (NOT actually game-"
                         "disjoint). 'correct' uses a real game-disjoint "
                         "partition; honest evaluation but numerically diverges "
                         "from Kassab's reported counts.")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_id = f"{args.backbone_type}_{args.backbone_size[0]}"

    print("=" * 60)
    print("Attentive-probe training (frozen backbone)")
    print("=" * 60)
    print(f"Backbone:    {backbone_id}")
    print(f"Probe:       "
          f"{'DINOv3AttentiveProbe (paper RoPE + sin-cos PE)' if args.backbone_type == 'dinov3' else 'AttentiveClassifier (V-JEPA2 paper)'}")
    if args.source_fps is not None and args.source_fps != args.fps:
        print(f"FPS / W:     target={args.fps}, source={args.source_fps} "
              f"(stride={int(round(args.source_fps / args.fps))}) / W={args.window_size}")
    else:
        print(f"FPS / W:     {args.fps} / {args.window_size}")
    print(f"Epochs:      {args.num_epochs}  batch={args.batch_size}  lr={args.learning_rate}  wd={args.weight_decay}")
    print(f"Class wts:   {('inverse-frequency [' + args.ce_weight_style + '] (train only)') if not args.no_class_weights else 'uniform (off)'}")
    print(f"Seed:        {args.seed}")
    print(f"Device:      {device}")
    print("=" * 60)

    # Data
    features_dir = TACDEC_FEATURES / f"{args.backbone_type}_{args.backbone_size}"
    eff_source_fps = args.source_fps if args.source_fps is not None else args.fps
    dense_tag = "reflect" if args.padding_mode == "reflect" else ""
    print(f"Protocol:    {args.protocol}")
    print(f"Padding:     {args.padding_mode}"
          + (f"  (loading *_{dense_tag}_dense_* files)" if dense_tag else ""))
    if args.protocol == "centered":
        train_loader, val_loader, test_loader, info = get_balanced_temporal_dataloaders(
            labels_dir=TACDEC_LABELS,
            features_dir=features_dir,
            backbone=args.backbone_type,
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
        if args.backbone_type != "dinov3":
            raise NotImplementedError(
                "kassab_concat is DINOv3-only. V-JEPA 2 pre-extracted dense "
                "features bake in single-clip temporal context, so cross-clip "
                "concat windows can't be assembled. Use --protocol centered "
                "for V-JEPA 2."
            )
        print(f"Caps:        replay_seqs<={args.replay_cap}, "
              f"bg_seqs={args.bg_count} (live seqs uncapped; strict Kassab "
              f"concat-and-slide at 5 FPS)")
        train_loader, val_loader, test_loader, info = get_kassab_concat_temporal_dataloaders(
            labels_dir=TACDEC_LABELS,
            features_dir=features_dir,
            backbone=args.backbone_type,
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
        print(f"Split mode:  {args.split_mode}"
              + ("  (replicates Kassab's extract_data bug; NOT game-disjoint)"
                 if args.split_mode == "kassab_bug" else
                 "  (real game-disjoint partition)"))
    else:
        raise NotImplementedError(f"--protocol {args.protocol!r} not wired.")
    print(f"Sequences per split: {info['n_sequences']}")
    for split, counts in info["frame_counts_per_split"].items():
        names = [CLASS_NAMES[c] for c in (0, 1, 2)]
        formatted = ", ".join(f"{n}={counts[c]}" for n, c in zip(names, (0, 1, 2)))
        print(f"  {split:5s} -> {formatted}")
    print(f"Game IDs (train, first 5): {info['game_ids']['train'][:5]}...")

    # Probe
    train_split = info["_splits"]["train"]
    if not train_split:
        raise RuntimeError("Empty train split; check Kassab subsampling parameters.")
    sample_feats = train_loader.dataset[0]["features"]
    feature_dim = int(sample_feats.shape[-1])
    print(f"Feature dim from cache:  {feature_dim}  (token shape per window: {tuple(sample_feats.shape)})")

    if args.backbone_type == "dinov3":
        # Paper recipe: probe_dim == in_dim for ViT-L; t_size == window_size so
        # the 3D sin-cos PE and RoPE caches match the (T=W, H, W) token grid.
        model = DINOv3AttentiveProbe(
            in_dim=feature_dim,
            probe_dim=feature_dim,
            num_classes=args.num_classes,
            num_heads=args.probe_num_heads,
            num_blocks=args.probe_depth,
            t_size=args.window_size,
            h_size=args.patch_h,
            w_size=args.patch_w,
        )
    else:
        model = AttentiveClassifier(
            embed_dim=feature_dim,
            num_heads=args.probe_num_heads,
            depth=args.probe_depth,
            num_classes=args.num_classes,
        )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Probe params: {n_params:,}")

    # Class weights (train split only)
    if args.no_class_weights:
        weights, counts = None, None
        weight_tensor = None
    else:
        weights, counts = compute_class_weights(
            train_split,
            num_classes=args.num_classes,
            normalization=args.ce_weight_style,
        )
        names = [CLASS_NAMES[c] for c in (0, 1, 2)]
        print("Train per-class counts: " + ", ".join(f"{n}={c}" for n, c in zip(names, counts)))
        print("CE weights:             " + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights)))
        weight_tensor = torch.tensor(weights, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6,
    )

    save_dir = TACDEC_MODELS / backbone_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"best_attn_{backbone_id}_{args.model_suffix}.pth"
    print(f"Will save best (max val macro-F1) to: {save_path}\n")

    best_val_loss = float("inf")
    best_val_loss_epoch: int | None = None
    f1_at_best_val_loss = -1.0
    best_val_macro_f1 = -1.0
    best_val_macro_f1_epoch: int | None = None
    val_loss_at_best_f1 = float("inf")
    epochs_since_improve = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        for batch in train_loader:
            feats = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = forward_probe(model, args.backbone_type, feats)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1
        avg_train = train_loss / max(n_batches, 1)

        val_loss, val_acc, val_macro_f1 = evaluate(
            model, args.backbone_type, val_loader, device, criterion, args.num_classes
        )
        msg = (f"Epoch {epoch:>3}/{args.num_epochs}  "
               f"train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  "
               f"val_acc={val_acc:.4f}  val_f1={val_macro_f1:.4f}")

        # Decoupled stop/save: checkpoint on val macro-F1, early-stop on val_loss.
        # Loss is a smoother plateau signal; macro-F1 is the metric we report.
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_macro_f1_epoch = epoch
            val_loss_at_best_f1 = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "selection_metric": "val_macro_f1",
                "best_val_macro_f1": best_val_macro_f1,
                "val_loss_at_best_f1": val_loss_at_best_f1,
                "best_val_loss": best_val_loss,
                "f1_at_best_val_loss": f1_at_best_val_loss,
                "backbone_type": args.backbone_type,
                "backbone_size": args.backbone_size,
                "backbone_id": backbone_id,
                "feature_dim": feature_dim,
                "num_classes": args.num_classes,
                "window_size": args.window_size,
                "fps": args.fps,
                "source_fps": eff_source_fps,
                "protocol": args.protocol,
                "probe_num_heads": args.probe_num_heads,
                "probe_depth": args.probe_depth,
                "patch_h": args.patch_h,
                "patch_w": args.patch_w,
                "class_weights": weights,
                "ce_weight_style": args.ce_weight_style,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "kassab_attentive_pipeline": True,
                "seed": args.seed,
            }, save_path)
            msg += "  -> new best F1, saved"

        # Early-stop tracker: val_loss plateau.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            f1_at_best_val_loss = val_macro_f1
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        scheduler.step()
        print(msg)

        if args.patience and epochs_since_improve >= args.patience:
            print(f"\nEarly stopping after {epoch} epochs "
                  f"(no val_loss improvement for {args.patience}).")
            break

    print(f"\nDone. Best val_macro_f1={best_val_macro_f1:.4f} "
          f"at epoch {best_val_macro_f1_epoch}. Checkpoint: {save_path}")

    if args.save_info:
        out = {
            "args": vars(args),
            "checkpoint": str(save_path),
            "selection_metric": "val_macro_f1",
            "best_val_macro_f1": best_val_macro_f1,
            "best_val_macro_f1_epoch": best_val_macro_f1_epoch,
            "val_loss_at_best_f1": val_loss_at_best_f1,
            "best_val_loss": best_val_loss,
            "best_val_loss_epoch": best_val_loss_epoch,
            "f1_at_best_val_loss": f1_at_best_val_loss,
            "feature_dim": feature_dim,
            "kassab_pipeline": {
                "frame_counts_per_split": {
                    split: {CLASS_NAMES[c]: v for c, v in d.items()}
                    for split, d in info["frame_counts_per_split"].items()
                },
                "n_sequences": info["n_sequences"],
                "target_fps": info["target_fps"],
                "window_size": info["window_size"],
                "game_ids": info["game_ids"],
            },
            "class_weights": weights,
            "class_counts_train": counts,
        }
        Path(args.save_info).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_info).write_text(json.dumps(out, indent=2))
        print(f"Run metadata saved to {args.save_info}")


if __name__ == "__main__":
    main()
