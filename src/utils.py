"""Active dataloader plumbing used by the kept analysis/visualization scripts.

`get_dataloaders` is the only public entry point; everything else in this file
exists to support it. Other helpers that previously lived here (FocalLoss,
compute_masked_loss, compute_class_weights, StaticBGUndersampledDataset, etc.)
served the retired k-fold / BiLSTM training pipeline and have moved to
`legacy/utils_old.py`.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.tacdec_dataset import TACDECSpottingDataset
from config import (
    BACKBONE_SIZE,
    BACKBONE_TYPE,
    NUM_CLASSES,
    TACDEC_FEATURES,
    TACDEC_LABELS,
    TRAIN_SEED,
)

DATA_SPLIT_SEED = TRAIN_SEED  # Use configurable seed for reproducible splits


def set_seed(seed: int):
    """Seed all RNGs and enable deterministic kernels for a reproducible run.

    Single source of truth for the temporal and spatial pipelines. Call this
    first thing in ``main()``, before any model build or dataloader iteration.
    For the spatial pipeline (three seeds), pass ``seed_train``; the split and
    balance seeds are fed separately to ``split_games`` / ``balance_split``,
    which use their own local RNGs and are unaffected by the global seeding here.
    """
    # CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA context use for
    # deterministic GEMMs. Authoritative source is setup.sh (exported before
    # Python starts); this line is a fallback for ad-hoc runs that don't source
    # it, and only takes effect if set_seed() runs before any CUDA work.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Force the math SDPA kernel: flash and memory-efficient attention have
    # non-deterministic backward passes on CUDA. Probe heads call
    # F.scaled_dot_product_attention through `with sdp_kernel():` (no args),
    # which picks the fastest available backend; disabling the two
    # non-deterministic backends globally leaves only math.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    # warn_only=True: fall back (with a warning) for any op lacking a
    # deterministic CUDA kernel rather than raising, so training never crashes.
    torch.use_deterministic_algorithms(True, warn_only=True)


def collate_with_windowing(batch, backbone_type='dinov3', temporal_head='lstm', window_size=16, num_classes=3):
    """
    Custom collate function with optional sliding window extraction for DINOv3 + BiLSTM.

    Applies sliding windows ONLY when:
      - backbone_type == 'dinov3'
      - temporal_head == 'lstm'

    Sliding windows with lower-middle center: for frame i, extracts [i - W//2 + 1 : i + W//2 + 1].
    For 16-frame window: [i-7 : i+9] with center at index 7 (lower middle of 0-15).
    Edges are zero-padded temporally. Returns per-frame windows and per-frame labels.
    The model will extract the center hidden state from the BiLSTM (at index W//2 - 1).

    Args:
        batch (list): List of samples from dataset, each with:
                     {'features': [max_len, feature_dim],
                      'labels': [max_len],
                      'mask': [max_len]}
        backbone_type (str): Backbone identifier ('dinov3' or 'vjepa2')
        temporal_head (str): Temporal head type ('lstm' or 'linear')
        window_size (int): Size of sliding window (default: 16, to match V-JEPA2)
        num_classes (int): Number of output classes

    Returns:
        dict: Batched tensors:
              - 'features': [batch, seq_len, window_size, feature_dim]  (if sliding window applied)
              - 'labels': [batch, seq_len]                              (per-frame, label of center frame)
              - 'mask': [batch, seq_len]                                (per-frame, mask of center frame)
    """
    features_list = []
    labels_list = []
    mask_list = []

    def _to_tensor(value, dtype=None):
        if torch.is_tensor(value):
            return value.to(dtype=dtype) if dtype is not None else value
        tensor = torch.as_tensor(value)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    # Stack batch
    gt_event_centers_batch = []
    video_id_batch = []

    for sample in batch:
        features_list.append(_to_tensor(sample['features'], dtype=torch.float32))
        labels_list.append(_to_tensor(sample['labels'], dtype=torch.long))
        mask_list.append(_to_tensor(sample['mask'], dtype=torch.float32))
        gt_event_centers_batch.append(sample.get('gt_event_centers', []))
        video_id_batch.append(sample.get('video_id', ''))

    # Batch: [batch_size, max_len, ...] (per-frame trailing dims preserved)
    batch_features = torch.stack(features_list)  # [B, max_len, D]  or  [B, max_len, num_tokens, D]
    batch_labels = torch.stack(labels_list)      # [B, max_len]
    batch_mask = torch.stack(mask_list)          # [B, max_len]

    # Apply sliding windows for:
    #   - DINOv3 + BiLSTM (original use-case)
    #   - any backbone + attpool / vjepa2_attpool over cls features
    # Skip when features are already dense per-clip token grids (4D: each frame
    # already carries its own [T*H*W, D] context).
    is_dense = batch_features.dim() == 4
    need_windows = (
        not is_dense
        and ((backbone_type == 'dinov3' and temporal_head == 'lstm')
             or temporal_head in ('attpool', 'vjepa2_attpool'))
    )
    if need_windows:
        batch_features, batch_labels, batch_mask = _apply_sliding_window(
            batch_features, batch_labels, batch_mask,
            window_size=window_size
        )

    return {
        'features': batch_features,
        'labels': batch_labels,
        'mask': batch_mask,
        'gt_event_centers': gt_event_centers_batch,
        'video_id': video_id_batch,
    }


def _apply_sliding_window(features, labels, mask, window_size=16):
    """
    Apply sliding window extraction with lower-middle center: for each frame i, extract [i - W//2 + 1 : i + W//2 + 1].

    For 16-frame window: [i-7 : i+9] with center at index 7 (lower middle of 0-15 indices).
    This creates per-frame context windows with zero-padding at sequence boundaries.
    Output shape allows model to process each frame with its temporal context,
    and extract the center hidden state at index W//2 - 1.

    Args:
        features: [batch, seq_len, feature_dim]
        labels: [batch, seq_len]
        mask: [batch, seq_len]
        window_size: Size of sliding window (default: 16 frames, to match V-JEPA2 temporal granularity)

    Returns:
        Tuple of (windowed_features, labels, mask)
        - windowed_features: [batch, seq_len, window_size, feature_dim]  (each frame gets a window)
        - labels: [batch, seq_len]                                        (per-frame labels, center frame)
        - mask: [batch, seq_len]                                          (per-frame mask, center frame)
    """
    batch_size, seq_len, feature_dim = features.shape
    half_window = window_size // 2  # For window_size=16: half_window=8

    # Pad features with zeros at the start for boundary handling
    # Padding on left: half_window - 1, on right: half_window
    padded_features = torch.nn.functional.pad(
        features,
        (0, 0, half_window - 1, half_window),  # (left, right) on seq_len dimension
        mode='constant',
        value=0.0
    )  # Shape: [batch, padded_len, feature_dim]

    # Vectorized extraction: efficiently create sliding windows using gather or stacking
    # Build indices for all window positions at once, then use gather
    # For each output position i, we extract padded_features[:, i:i+window_size, :]
    windowed_features_list = [
        padded_features[:, i:i+window_size, :].unsqueeze(1)
        for i in range(seq_len)
    ]
    windowed_features = torch.cat(windowed_features_list, dim=1)  # [batch, seq_len, window_size, feature_dim]

    # Labels and mask stay per-frame (center frame of each window)
    return windowed_features, labels, mask


class MaskedSubset(torch.utils.data.Dataset):
    """
    Applies pre-computed per-sample keep masks to a Subset.

    Used by the segment-level undersampling path: masks are computed on the
    full dataset before the train/val/test split and applied to all three
    splits, matching Evan's approach of reducing the dataset before splitting.

    Args:
        subset: A torch Subset (or any Dataset).
        keep_masks: dict mapping local index → np.ndarray [seq_len] of 0/1 floats.
                    1.0 = frame contributes to loss, 0.0 = dropped.
    """

    def __init__(self, subset, keep_masks: dict):
        self.subset = subset
        self.keep_masks = keep_masks

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, local_idx):
        sample = dict(self.subset[local_idx])
        if local_idx in self.keep_masks:
            km = torch.tensor(self.keep_masks[local_idx], dtype=torch.float32)
            sample['mask'] = sample['mask'] * km
        return sample


def compute_segment_balance_masks(dataset, bg_subsample_ratio: float, seed: int,
                                   num_classes: int = 3) -> dict:
    """
    Evan-equivalent pre-split undersampling.

    Treats every contiguous background segment in every clip as a droppable
    unit. Shuffles all segments globally, then drops enough to reduce the
    total background frame count by bg_subsample_ratio.

    Computed on the FULL dataset before any train/val/test split and applied
    to all three splits, matching Evan's approach of balancing before splitting.

    Args:
        dataset: Full TACDECSpottingDataset (not a Subset).
        bg_subsample_ratio: Fraction of total background frames to drop (e.g. 0.8).
        seed: RNG seed — use the run seed for reproducibility.
        num_classes: Number of output classes (background = num_classes - 1).

    Returns:
        dict mapping global sample index → np.ndarray keep_mask [seq_len]
        (1.0 = keep, 0.0 = dropped background segment frame).
        Only samples that have at least one dropped segment appear in the dict.
    """
    background_class = num_classes - 1

    # Collect all contiguous background segments across the full dataset
    all_segments = []  # (global_idx, seg_start, seg_end, n_frames)
    total_bg_frames = 0

    print("Computing segment-level background balance masks (full dataset)...")
    for global_idx in range(len(dataset)):
        sample = dataset[global_idx]
        labels = np.asarray(sample['labels'])
        mask   = np.asarray(sample['mask'])
        seq_len = int(mask.sum())

        is_bg  = (labels[:seq_len] == background_class).astype(np.int32)
        padded = np.concatenate([[0], is_bg, [0]])
        diff   = np.diff(padded)
        starts = np.where(diff ==  1)[0]
        ends   = np.where(diff == -1)[0]

        for s, e in zip(starts, ends):
            n = int(e - s)
            all_segments.append((global_idx, s, e, n))
            total_bg_frames += n

    n_to_drop = int(total_bg_frames * bg_subsample_ratio)
    print(f"  Total background frames: {total_bg_frames:,}")
    print(f"  Target drop:            {n_to_drop:,} ({bg_subsample_ratio*100:.0f}%)")

    # Shuffle segments globally and greedily drop until quota met
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_segments))

    dropped_ranges: dict[int, list] = {}
    frames_dropped = 0
    segs_dropped = 0

    for i in indices:
        if frames_dropped >= n_to_drop:
            break
        global_idx, s, e, n = all_segments[i]
        dropped_ranges.setdefault(global_idx, []).append((s, e))
        frames_dropped += n
        segs_dropped   += 1

    print(f"  Segments dropped:       {segs_dropped:,} across {len(dropped_ranges)} clips")
    print(f"  Frames dropped:         {frames_dropped:,} / {total_bg_frames:,} "
          f"({frames_dropped / max(total_bg_frames, 1) * 100:.1f}%)")

    # Build keep masks only for affected samples (others implicitly all-ones)
    keep_masks: dict[int, np.ndarray] = {}
    for global_idx, ranges in dropped_ranges.items():
        sample  = dataset[global_idx]
        seq_len = len(np.asarray(sample['labels']))
        km      = np.ones(seq_len, dtype=np.float32)
        for s, e in ranges:
            km[s:e] = 0.0
        keep_masks[global_idx] = km

    return keep_masks


def compute_target_count_masks(dataset, target_counts, seed: int,
                                num_classes: int = 3) -> dict:
    """
    Evan-style pre-split frame sampling: keep exactly target_counts[c] frames
    per class across the full dataset, drop everything else from the loss.

    Mirrors Evan's per-frame setup (e.g. 4000:2000:2000 for tackle-live:replay:bg)
    but adapted to clip-based temporal models — emits per-clip keep masks rather
    than reshuffling frames into independent samples.

    Computed on the FULL dataset before split, applied to all train/val/test
    splits — same convention as compute_segment_balance_masks.

    Args:
        dataset: Full TACDECSpottingDataset.
        target_counts: list[int] of length num_classes.
        seed: RNG seed.
        num_classes: Number of classes.

    Returns:
        dict mapping global sample index → np.ndarray keep_mask [seq_len].
    """
    if len(target_counts) != num_classes:
        raise ValueError(
            f"target_counts has length {len(target_counts)}, expected {num_classes}"
        )

    rng = np.random.default_rng(seed)

    per_class_frames = [[] for _ in range(num_classes)]
    sample_seq_lens = {}

    print("Computing target-count sampling masks (Evan-style)...")
    for global_idx in range(len(dataset)):
        sample = dataset[global_idx]
        labels = np.asarray(sample['labels'])
        mask = np.asarray(sample['mask'])
        sample_seq_lens[global_idx] = len(labels)
        valid_len = int(mask.sum())
        for f in range(valid_len):
            c = int(labels[f])
            if 0 <= c < num_classes:
                per_class_frames[c].append((global_idx, f))

    keep_per_idx = {}
    for c in range(num_classes):
        n_avail = len(per_class_frames[c])
        n_target = target_counts[c]
        if n_avail < n_target:
            print(f"  WARN: class {c} has only {n_avail:,} frames, target was {n_target:,}. Keeping all.")
            n_target = n_avail
        kept_pct = (n_target / max(n_avail, 1)) * 100
        print(f"  Class {c}: keep {n_target:,} / {n_avail:,} ({kept_pct:.1f}%)")

        chosen = (range(n_avail) if n_target >= n_avail
                  else rng.choice(n_avail, size=n_target, replace=False))
        for i in chosen:
            gi, fi = per_class_frames[c][int(i)]
            keep_per_idx.setdefault(gi, set()).add(fi)

    keep_masks = {}
    for global_idx, seq_len in sample_seq_lens.items():
        km = np.zeros(seq_len, dtype=np.float32)
        if global_idx in keep_per_idx:
            for f in keep_per_idx[global_idx]:
                km[f] = 1.0
        keep_masks[global_idx] = km

    total_kept = sum(int(k.sum()) for k in keep_masks.values())
    total_frames = sum(sample_seq_lens.values())
    print(f"  Total frames kept: {total_kept:,} / {total_frames:,} "
          f"({total_kept/max(total_frames,1)*100:.1f}%)")

    return keep_masks


def get_dataloaders(batch_size=8, backbone_type=None, backbone_size=None, num_classes=None,
                    labeling_mode='interval', tolerance_sec=0.0, temporal_head='lstm', extraction_fps=5.0,
                    bg_undersample_mode='none', bg_subsample_ratio=0.0, bg_undersample_seed=42,
                    target_counts=None,
                    window_size=16, fold_idx=None, n_folds=5,
                    val_frac=None,
                    feature_type='cls',
                    apply_balancing_to_eval=False):
    """
    Get train/val/test dataloaders for TACDEC dataset.

    Args:
        batch_size (int): Batch size for dataloaders
        backbone_type (str): Backbone type ('dinov3', 'vjepa2'). Uses config default if None.
        backbone_size (str): Backbone size ('base', 'large'). Uses config default if None.
        num_classes (int): Number of classes (3 or 5). Uses config default if None.
        labeling_mode (str): Labeling strategy - 'anchor' (center ± tolerance) or 'interval' (full range + padding). Default: 'interval'.
        tolerance_sec (float): Temporal tolerance in seconds. Default: 0.0.
        temporal_head (str): Temporal head type ('lstm' or 'linear'). Controls whether 16-frame windowing is applied.
                            Windowing is applied ONLY when backbone_type='dinov3' AND temporal_head='lstm'. Default: 'lstm'.
        extraction_fps (float): FPS used during feature extraction (default: 5.0). Must match the fps suffix in feature filenames.
    """
    # Use config defaults if not specified
    backbone_type = backbone_type or BACKBONE_TYPE
    backbone_size = backbone_size or BACKBONE_SIZE
    num_classes = num_classes or NUM_CLASSES

    # Construct path to backbone-specific features
    features_dir = TACDEC_FEATURES / f"{backbone_type}_{backbone_size}"
    MAX_CLIP_DURATION_SEC = 30.0  # Hard max is ~28s, 30s gives ~7% headroom
    dataset = TACDECSpottingDataset(
        features_dir=features_dir,
        labels_dir=TACDEC_LABELS,
        max_sequence_length=int(MAX_CLIP_DURATION_SEC * extraction_fps),
        extraction_fps=extraction_fps,
        tolerance_sec=tolerance_sec,
        num_classes=num_classes,
        labeling_mode=labeling_mode,
        feature_type=feature_type,
    )

    # ===== GAME-DISJOINT SPLIT (Prevents data leakage) =====
    # Build mapping: game_id -> list of sample indices
    # This ensures no game appears in multiple splits (train/val/test)
    game_to_indices = {}
    samples_with_invalid_game_id = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        game_id = sample['game_id']

        # Skip invalid game_ids (shouldn't happen if labels are correct)
        if game_id == -1:
            samples_with_invalid_game_id += 1
            continue

        if game_id not in game_to_indices:
            game_to_indices[game_id] = []
        game_to_indices[game_id].append(idx)

    if samples_with_invalid_game_id > 0:
        print(f"⚠️  Warning: {samples_with_invalid_game_id} samples have invalid game_id (-1). Check label metadata.")

    # Sort then shuffle games with a fixed seed so the global game ordering is
    # identical across runs. fold_idx then deterministically picks one chunk as
    # test; the remaining games are split train/val. When fold_idx is None we
    # fall back to the legacy 80/10/10 split.
    rng = np.random.default_rng(DATA_SPLIT_SEED)
    games = sorted(game_to_indices.keys())
    rng.shuffle(games)

    print(f"Game-disjoint split: {len(games)} unique games, {len(dataset)} total samples")

    if fold_idx is not None:
        if not (0 <= fold_idx < n_folds):
            raise ValueError(f"fold_idx must be in [0, {n_folds}), got {fold_idx}")
        folds = [list(f) for f in np.array_split(games, n_folds)]
        test_games = folds[fold_idx]
        remaining  = [g for i, f in enumerate(folds) if i != fold_idx for g in f]
        # val_frac is the fraction of TOTAL games allocated to val. Defaults to
        # ~1/8 of remaining → mirrors the legacy 80/10 train/val ratio at n_folds=5.
        # Set explicitly (e.g. 0.15) to control the val split independently.
        if val_frac is None:
            n_val = max(1, len(remaining) // 8)
        else:
            n_val = max(1, round(val_frac * len(games)))
            n_val = min(n_val, len(remaining) - 1)
        val_games   = remaining[-n_val:]
        train_games = remaining[:-n_val]
        print(f"  Fold {fold_idx + 1}/{n_folds} — test fold has {len(test_games)} games, "
              f"val has {len(val_games)} games, train has {len(train_games)} games")
    else:
        # Single-split path. Default 80/10/10; override val_frac to e.g. 0.15
        # for a 70/15/15 layout (test gets the remainder after train+val).
        n_games = len(games)
        v = 0.10 if val_frac is None else val_frac
        train_split_idx = int(0.8 * n_games) if val_frac is None else int((1.0 - 2 * v) * n_games)
        val_split_idx   = train_split_idx + max(1, int(v * n_games))
        train_games = games[:train_split_idx]
        val_games   = games[train_split_idx:val_split_idx]
        test_games  = games[val_split_idx:]

    # Collect sample indices for each split
    train_indices = [i for g in train_games for i in game_to_indices[g]]
    val_indices = [i for g in val_games for i in game_to_indices[g]]
    test_indices = [i for g in test_games for i in game_to_indices[g]]

    # Log split statistics
    print(f"  Train: {len(train_games)} games, {len(train_indices)} samples")
    print(f"  Val:   {len(val_games)} games, {len(val_indices)} samples")
    print(f"  Test:  {len(test_games)} games, {len(test_indices)} samples")

    # ===== SEGMENT-LEVEL BACKGROUND UNDERSAMPLING (Evan-equivalent) =====
    # Masks are computed on the FULL dataset before the split so balance targets
    # are based on global statistics. By default the masks are applied to TRAIN
    # ONLY: val/test see the natural class distribution, which is required for
    # Average-mAP to reflect deployment behavior. Set apply_balancing_to_eval=True
    # to reproduce the legacy behavior (mask all three splits).
    segment_keep_masks = None
    if bg_undersample_mode == 'segment' and bg_subsample_ratio > 0.0:
        segment_keep_masks = compute_segment_balance_masks(
            dataset, bg_subsample_ratio, bg_undersample_seed, num_classes
        )
    elif bg_undersample_mode == 'target_counts' and target_counts is not None:
        segment_keep_masks = compute_target_count_masks(
            dataset, list(target_counts), bg_undersample_seed, num_classes
        )

    def _make_split(indices, apply_masks):
        subset = Subset(dataset, indices)
        if segment_keep_masks is None or not apply_masks:
            return subset
        local_masks = {
            local_idx: segment_keep_masks[global_idx]
            for local_idx, global_idx in enumerate(indices)
            if global_idx in segment_keep_masks
        }
        return MaskedSubset(subset, local_masks)

    train_dataset = _make_split(train_indices, apply_masks=True)
    val_dataset   = _make_split(val_indices,   apply_masks=apply_balancing_to_eval)
    test_dataset  = _make_split(test_indices,  apply_masks=apply_balancing_to_eval)

    if segment_keep_masks is not None and not apply_balancing_to_eval:
        print("  Background balancing applied to TRAIN only (val/test on natural distribution).")
    elif segment_keep_masks is not None:
        print("  Background balancing applied to TRAIN, VAL, and TEST (legacy behavior).")

    # Report final per-class frame counts in each split (reflecting whether
    # segment masking is applied to that split).
    def _summarize_split(name, indices, apply_masks):
        counts = np.zeros(num_classes, dtype=np.int64)
        for gi in indices:
            sample = dataset[gi]
            labels = np.asarray(sample['labels'])
            base_mask = np.asarray(sample['mask']).astype(np.float32)
            if apply_masks and segment_keep_masks is not None and gi in segment_keep_masks:
                base_mask = base_mask * segment_keep_masks[gi].astype(np.float32)
            valid = base_mask > 0
            for c in range(num_classes):
                counts[c] += int(((labels == c) & valid).sum())
        total = int(counts.sum())
        pct = [f"{(c / max(total, 1)) * 100:.1f}%" for c in counts]
        print(f"  {name:5s}  per-class: {counts.tolist()}  (total {total:,})  → {pct}")

    if segment_keep_masks is not None:
        print("\nFinal frame split per class (train post-masking, val/test reflect "
              f"{'masking' if apply_balancing_to_eval else 'natural distribution'}):")
    else:
        print("\nFinal frame split per class:")
    _summarize_split("Train", train_indices, apply_masks=True)
    _summarize_split("Val",   val_indices,   apply_masks=apply_balancing_to_eval)
    _summarize_split("Test",  test_indices,  apply_masks=apply_balancing_to_eval)
    print()
    # ===== END GAME-DISJOINT SPLIT =====

    # Create custom collate function with sliding window parameters
    collate_fn = lambda batch: collate_with_windowing(
        batch,
        backbone_type=backbone_type,
        temporal_head=temporal_head,
        window_size=window_size,
        num_classes=num_classes
    )

    # Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
