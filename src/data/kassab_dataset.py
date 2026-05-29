"""
Kassab-style preprocessing for TACDEC frame classification (temporal approach).

Reproduces the pipeline from Kassab (2024) "Tackle detection in football":
  Step 1 — class merge 5 -> 3
  Step 2 — extract contiguous-class sequences (one per run-of-equal-labels)
  Step 3 — keep tackle sequences whole; sample 500 random 25-frame background
           chunks from segments >= 25 frames not overlapping tackles
  Step 4 — split sequences 70/15/15 by game ID (game-disjoint)
  Step 5 — concatenate test sequences and slide stride-1 W=50 windows for eval

Operates on pre-extracted frame-level features (DINOv3 CLS or any per-frame
[T, D] tensor) loaded via TACDECSpottingDataset. Reflective 1:1 padding from
the thesis is intentionally NOT applied here — that is a feature-extraction
side concern. Frames already represent whatever the extractor produced.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.tacdec_dataset import TACDECSpottingDataset


# 3-class indices after the 5 -> 3 merge done by TACDECSpottingDataset
CLASS_TACKLE_LIVE = 0
CLASS_TACKLE_REPLAY = 1
CLASS_BACKGROUND = 2

CLASS_NAMES = {
    CLASS_TACKLE_LIVE: "tackle-live",
    CLASS_TACKLE_REPLAY: "tackle-replay",
    CLASS_BACKGROUND: "background",
}

# Kassab thesis Table 7.6 totals (full dataset, post-subsample, pre-split)
EXPECTED_COUNTS = {
    CLASS_TACKLE_LIVE: 363,
    CLASS_TACKLE_REPLAY: 280,
    CLASS_BACKGROUND: 500,
}
EXPECTED_FRAMES = {
    CLASS_TACKLE_LIVE: 7941,
    CLASS_TACKLE_REPLAY: 12573,
    CLASS_BACKGROUND: 12500,
}


def extract_sequences(dataset: TACDECSpottingDataset) -> list[dict]:
    """
    Walk every clip, emit one record per contiguous run of frames sharing a
    class label. Each record carries source game/video IDs plus the slice of
    pre-extracted features for that run.
    """
    sequences = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        feats = sample["features"].numpy()
        labels = sample["labels"].numpy()
        mask = sample["mask"].numpy()
        game_id = sample["game_id"]
        video_id = sample["video_id"]

        valid_len = int(mask.sum())
        if valid_len == 0:
            continue
        feats = feats[:valid_len]
        labels = labels[:valid_len]

        run_start = 0
        for i in range(1, valid_len + 1):
            if i == valid_len or labels[i] != labels[run_start]:
                sequences.append({
                    "game_id": game_id,
                    "video_id": video_id,
                    "class": int(labels[run_start]),
                    "features": feats[run_start:i].copy(),
                    "n_frames": i - run_start,
                })
                run_start = i
    return sequences


def sample_background_chunks(sequences, target_count=500, chunk_len=25,
                             min_segment_len=70, seed=42):
    """
    Kassab-faithful background sampler (TempTAC.ipynb cell 7). Walks bg
    segments in DATA ORDER, takes the first `target_count` whose length is
    >= `min_segment_len`, picks a random start at randint(35, len - 34)
    so the chunk_len-frame chunk sits past a 35-frame leading buffer.

    Default `min_segment_len=70` and `randint(35, n-34)` reproduce cell 7
    exactly. Note the asymmetry: the sampler reserves 35 frames at the start
    but the chunk's tail (start+25) can land as close as 11 frames from the
    end -- matches the original code, not the prose.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for s in sequences:                       # already in data order
        if s["class"] != CLASS_BACKGROUND:
            continue
        if s["n_frames"] < min_segment_len:
            continue
        if len(chunks) >= target_count:
            break
        # Matches cell 7's `np.random.randint(35, len(bg_seq) - 34)`:
        # exclusive upper bound, so max start = n_frames - 35.
        start = int(rng.integers(35, s["n_frames"] - 34))
        chunks.append({
            "game_id": s["game_id"],
            "video_id": s["video_id"],
            "class": CLASS_BACKGROUND,
            "features": s["features"][start:start + chunk_len].copy(),
            "n_frames": chunk_len,
        })
    if len(chunks) < target_count:
        raise RuntimeError(
            f"Only {len(chunks)} bg segments with len >= {min_segment_len} "
            f"found in data order; need {target_count}."
        )
    return chunks


def build_kassab_sequences(dataset, seed=42, bg_count=500, bg_chunk_len=25,
                           bg_min_len=70, replay_cap=280):
    """
    Kassab-faithful sequence assembly (TempTAC.ipynb cell 7).

    Walks `extract_sequences(dataset)` in data order:
      - tackle-live   : kept entirely, NO cap (matches cell 7).
      - tackle-replay : capped at `replay_cap` first-come-first-served.
      - background    : delegated to `sample_background_chunks` (len >= 70,
                        first 500 with mid-segment 25-frame chunk).

    Step 1 (5 -> 3 class merge) is already performed by the underlying
    TACDECSpottingDataset when num_classes=3.
    """
    raw = extract_sequences(dataset)            # data order: clip-by-clip, in-clip temporal

    out = []
    replay_kept = 0
    for s in raw:
        if s["class"] == CLASS_TACKLE_LIVE:
            out.append(s)                       # uncapped (cell 7)
        elif s["class"] == CLASS_TACKLE_REPLAY:
            if replay_kept < replay_cap:
                out.append(s)
                replay_kept += 1
            # else: drop, cap reached (cell 7's `tackle_count2 >= 280: continue`)

    out.extend(sample_background_chunks(
        raw, target_count=bg_count, chunk_len=bg_chunk_len,
        min_segment_len=bg_min_len, seed=seed,
    ))
    return out


def verify_counts(sequences, hard_fail_pct=0.50):
    """
    Compare actual vs Kassab Table 7.6 totals.

    - background:    sampled to fixed target -> should match exactly.
    - tackle-replay: capped at 280 first-come-first-served -> should match
                     exactly if the source has >=280 replay sequences.
    - tackle-live:   UNCAPPED in cell 7 -> may differ because DINOv3 may have
                     extracted features from a slightly different clip set
                     than DINOv2. Reported but only sanity-checked, not
                     hard-failed.
    """
    counts = {c: 0 for c in CLASS_NAMES}
    frames = {c: 0 for c in CLASS_NAMES}
    for s in sequences:
        counts[s["class"]] += 1
        frames[s["class"]] += s["n_frames"]

    print("\nKassab Table 7.6 verification (full dataset, post-subsample, pre-split):")
    print(f"  {'class':<16}{'count':>6} (exp {'    ':>4}){'frames':>10}  (exp)")
    fatal = []
    # Strictness: tackle-live is uncapped so we only sanity-check it; the
    # other two are bounded by caps and should match closely.
    strict_classes = (CLASS_TACKLE_REPLAY, CLASS_BACKGROUND)
    for c in (CLASS_TACKLE_LIVE, CLASS_TACKLE_REPLAY, CLASS_BACKGROUND):
        ec, ef = EXPECTED_COUNTS[c], EXPECTED_FRAMES[c]
        dc = abs(counts[c] - ec) / max(ec, 1)
        df = abs(frames[c] - ef) / max(ef, 1)
        if c == CLASS_TACKLE_LIVE:
            note = "info only (uncapped in Kassab)"
            flag = note if max(dc, df) > 0.05 else f"OK [{note}]"
        else:
            flag = "OK" if max(dc, df) <= 0.05 else f"deviation count={dc:+.1%} frames={df:+.1%}"
        print(f"  {CLASS_NAMES[c]:<16}{counts[c]:>6} ({ec:>4}){'':>4}{frames[c]:>10}  ({ef})  {flag}")
        if c in strict_classes and max(dc, df) > hard_fail_pct:
            fatal.append((CLASS_NAMES[c], dc, df))

    if fatal:
        msg = "; ".join(f"{n} count={dc:+.1%} frames={df:+.1%}" for n, dc, df in fatal)
        raise RuntimeError(
            f"Pre-split counts deviate >{hard_fail_pct:.0%} from Kassab Table 7.6 "
            f"on capped classes -- probable bug in extraction. Details: {msg}"
        )
    print()
    return counts, frames


def split_by_game(sequences, train=0.70, val=0.15, seed=42):
    """
    Game-disjoint sequence split. Test fraction = 1 - train - val. Prints the
    literal game-ID assignment for audit.

    Uses legacy np.random (seed + shuffle) rather than the modern Generator
    API so the shuffle matches TempTAC.ipynb cell 9 exactly:
        np.random.seed(seed); np.random.shuffle(game_indices)
    Cell 9 also uses int() (truncation) for the partition sizes, not round().
    """
    games = sorted({s["game_id"] for s in sequences})
    np.random.seed(seed)
    np.random.shuffle(games)            # legacy in-place shuffle, matches cell 9
    n = len(games)
    n_train = int(n * train)            # int() truncation, matches cell 9
    n_val = int(n * val)
    train_games = set(games[:n_train])
    val_games = set(games[n_train:n_train + n_val])
    test_games = set(games[n_train + n_val:])

    splits = {
        "train": [s for s in sequences if s["game_id"] in train_games],
        "val":   [s for s in sequences if s["game_id"] in val_games],
        "test":  [s for s in sequences if s["game_id"] in test_games],
    }

    print(f"Game-disjoint sequence split (target {train:.0%}/{val:.0%}/"
          f"{1 - train - val:.0%}):")
    for name, gset in (("train", train_games), ("val", val_games), ("test", test_games)):
        print(f"  {name:<5} {len(gset):>2} games: {sorted(gset)}")
    print()

    print("Per-split sequence and frame counts:")
    for name, seqs in splits.items():
        per_class = {c: (0, 0) for c in CLASS_NAMES}
        for s in seqs:
            n_seq, n_fr = per_class[s["class"]]
            per_class[s["class"]] = (n_seq + 1, n_fr + s["n_frames"])
        print(f"  {name:<5} total {len(seqs):>4} sequences, "
              f"{sum(p[1] for p in per_class.values()):>6} frames")
        for c in (CLASS_TACKLE_LIVE, CLASS_TACKLE_REPLAY, CLASS_BACKGROUND):
            ns, nf = per_class[c]
            print(f"        {CLASS_NAMES[c]:<16} {ns:>4} sequences, {nf:>6} frames")
    print()

    return splits, {"train": sorted(train_games), "val": sorted(val_games),
                    "test": sorted(test_games)}


class KassabConcatDataset(Dataset):
    """
    Concatenate a list of sequence records along the time axis, expose one
    stride-1 sliding window of length `window_size`. Two boundary modes:

      'no-pad'  (default, Kassab-faithful per §4.5.4 "all but the first are
                 present again with a new slid-in"): emit only the
                 N - W + 1 windows that fit entirely inside the real stream.
                 No padding, mask is all ones, every timestep contributes.

      'pad-mask' (alternative): zero-pad features at the stream boundaries so
                 every real frame is centered in exactly one window. Padded
                 timesteps carry label = -1 and mask = 0 so they are excluded
                 from per-frame metrics. Gives every real frame equal
                 evaluation weight; differs from Kassab by <1% support.

    Each window emits, for the W timesteps inside it:
        features:      [W, D]  float32
        labels:        [W]     int64   (-1 sentinel only in pad-mask mode)
        mask:          [W]     float32 (1.0 valid, 0.0 padded; all 1.0 in no-pad)
        center_label:  int     class label of the window's center frame
                                (only valid in pad-mask, or in interior windows
                                in no-pad — always valid by construction since
                                center_idx maps inside the window)
    """

    def __init__(self, sequences, window_size=50, boundary="no-pad"):
        if not sequences:
            raise ValueError("KassabConcatDataset: no sequences provided")
        if boundary not in ("no-pad", "pad-mask"):
            raise ValueError(
                f"boundary must be 'no-pad' or 'pad-mask', got {boundary!r}")

        feats = np.concatenate([s["features"] for s in sequences], axis=0).astype(np.float32)
        labels = np.concatenate(
            [np.full(s["n_frames"], s["class"], dtype=np.int64) for s in sequences],
            axis=0,
        )
        self.window_size = window_size
        self.boundary = boundary
        # Lower-middle center, matches existing _apply_sliding_window in utils.py.
        self.center_idx = window_size // 2 - 1
        self.n_real_frames = int(len(labels))
        D = feats.shape[1]

        if boundary == "no-pad":
            if self.n_real_frames < window_size:
                raise RuntimeError(
                    f"Concatenated stream has {self.n_real_frames} real frames but "
                    f"W={window_size}. 'no-pad' boundary mode emits N - W + 1 "
                    "windows, which is non-positive. Use boundary='pad-mask' or "
                    "verify your split."
                )
            self.padded_feats = feats
            self.padded_labels = labels
            self.padded_mask = np.ones(self.n_real_frames, dtype=np.float32)
            self.n_windows = self.n_real_frames - window_size + 1
            self.total_valid_timesteps = self.n_windows * window_size
        else:  # pad-mask
            pad_left = window_size // 2 - 1
            pad_right = window_size // 2
            self.padded_feats = np.concatenate(
                [np.zeros((pad_left, D), dtype=np.float32),
                 feats,
                 np.zeros((pad_right, D), dtype=np.float32)],
                axis=0,
            )
            self.padded_labels = np.concatenate(
                [np.full(pad_left, -1, dtype=np.int64),
                 labels,
                 np.full(pad_right, -1, dtype=np.int64)],
                axis=0,
            )
            self.padded_mask = np.concatenate(
                [np.zeros(pad_left, dtype=np.float32),
                 np.ones(self.n_real_frames, dtype=np.float32),
                 np.zeros(pad_right, dtype=np.float32)],
                axis=0,
            )
            self.n_windows = self.n_real_frames
            # Closed-form total valid timesteps via cumsum: window i sums
            # padded_mask[i:i+W] = cs[i+W] - cs[i], summed over i in [0, N).
            cs = np.concatenate([[0.0], np.cumsum(self.padded_mask)])
            self.total_valid_timesteps = int(
                (cs[window_size:window_size + self.n_windows]
                 - cs[:self.n_windows]).sum()
            )

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        end = idx + self.window_size
        return {
            "features":     torch.from_numpy(self.padded_feats[idx:end].copy()),
            "labels":       torch.from_numpy(self.padded_labels[idx:end].copy()),
            "mask":         torch.from_numpy(self.padded_mask[idx:end].copy()),
            "center_label": int(self.padded_labels[idx + self.center_idx]),
        }


def kassab_collate(batch):
    return {
        "features":     torch.stack([b["features"] for b in batch], dim=0),    # [B, W, D]
        "labels":       torch.stack([b["labels"]   for b in batch], dim=0),    # [B, W]
        "mask":         torch.stack([b["mask"]     for b in batch], dim=0),    # [B, W]
        "center_label": torch.tensor([b["center_label"] for b in batch], dtype=torch.long),  # [B]
    }


def _build_kassab_splits(features_dir, labels_dir, backbone, num_classes,
                         extraction_fps, seed, max_clip_sec, train_frac, val_frac):
    """Shared pipeline up to (and including) the game-disjoint split."""
    if num_classes != 3:
        raise ValueError(f"Kassab pipeline is 3-class only (got {num_classes})")

    base = TACDECSpottingDataset(
        features_dir=features_dir,
        labels_dir=labels_dir,
        max_sequence_length=int(max_clip_sec * extraction_fps),
        extraction_fps=extraction_fps,
        backbone=backbone,
        tolerance_sec=0.0,
        num_classes=num_classes,
        labeling_mode="interval",
        feature_type="cls",
    )
    if len(base) == 0:
        raise RuntimeError(
            f"No matching feature files: features_dir={features_dir}, "
            f"backbone={backbone}, fps={extraction_fps}"
        )
    print(f"Loaded {len(base)} clips from {features_dir} at {extraction_fps} FPS "
          f"(backbone filter: {backbone}).")

    seqs = build_kassab_sequences(base, seed=seed)
    counts, frames = verify_counts(seqs)
    splits, game_ids = split_by_game(seqs, train=train_frac, val=val_frac, seed=seed)

    # Per-split per-class frame counts (serializable; suitable for JSON dump).
    frame_counts_per_split = {}
    for name, seq_list in splits.items():
        per_class = {c: 0 for c in CLASS_NAMES}
        for s in seq_list:
            per_class[s["class"]] += s["n_frames"]
        frame_counts_per_split[name] = per_class

    info = {
        "counts_total": counts,
        "frames_total": frames,
        "n_sequences": {k: len(v) for k, v in splits.items()},
        "game_ids": game_ids,
        "frame_counts_per_split": frame_counts_per_split,
        # Underscore-prefixed: contains numpy arrays; do NOT serialize directly.
        "_splits": splits,
    }
    return splits, info


def get_kassab_test_loader(features_dir, labels_dir, backbone, num_classes=3,
                           extraction_fps=25.0, window_size=50, batch_size=128,
                           seed=42, max_clip_sec=30.0,
                           train_frac=0.70, val_frac=0.15,
                           boundary="no-pad"):
    """
    One-call entrypoint: build the underlying clip dataset, run the Kassab
    pipeline, return a DataLoader over stride-1 windows of the test stream.

    Args:
        boundary: 'no-pad' (Kassab-faithful, emits N - W + 1 windows) or
                  'pad-mask' (zero-pad + per-timestep mask, every real frame
                  centered once). See KassabConcatDataset.
    """
    splits, info = _build_kassab_splits(
        features_dir, labels_dir, backbone, num_classes, extraction_fps,
        seed, max_clip_sec, train_frac, val_frac,
    )
    test_ds = KassabConcatDataset(splits["test"], window_size=window_size,
                                  boundary=boundary)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=kassab_collate)
    info["n_test_frames"] = test_ds.n_real_frames
    info["n_test_windows"] = test_ds.n_windows
    info["test_total_valid_timesteps"] = test_ds.total_valid_timesteps
    info["window_size"] = window_size
    info["boundary"] = boundary
    return test_loader, info


def get_kassab_dataloaders(features_dir, labels_dir, backbone, num_classes=3,
                           extraction_fps=25.0, window_size=50, batch_size=128,
                           seed=42, max_clip_sec=30.0,
                           train_frac=0.70, val_frac=0.15,
                           num_workers=0,
                           boundary="no-pad"):
    """
    Train / val / test loaders from the Kassab pipeline. Train shuffled,
    val/test sequential. Same `boundary` mode applied to all three.
    """
    splits, info = _build_kassab_splits(
        features_dir, labels_dir, backbone, num_classes, extraction_fps,
        seed, max_clip_sec, train_frac, val_frac,
    )
    train_ds = KassabConcatDataset(splits["train"], window_size=window_size, boundary=boundary)
    val_ds   = KassabConcatDataset(splits["val"],   window_size=window_size, boundary=boundary)
    test_ds  = KassabConcatDataset(splits["test"],  window_size=window_size, boundary=boundary)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=kassab_collate, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=kassab_collate, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=kassab_collate, num_workers=num_workers)

    info["n_frames"] = {
        "train": train_ds.n_real_frames,
        "val":   val_ds.n_real_frames,
        "test":  test_ds.n_real_frames,
    }
    info["n_windows"] = {
        "train": train_ds.n_windows,
        "val":   val_ds.n_windows,
        "test":  test_ds.n_windows,
    }
    info["total_valid_timesteps"] = {
        "train": train_ds.total_valid_timesteps,
        "val":   val_ds.total_valid_timesteps,
        "test":  test_ds.total_valid_timesteps,
    }
    info["window_size"] = window_size
    info["boundary"] = boundary
    return train_loader, val_loader, test_loader, info


def compute_kassab_class_weights(sequences, num_classes):
    """
    Inverse-frequency weights from a list of Kassab sequence records (the train
    split). Counts each real frame exactly once — independent of the window
    structure (windows multi-count frames). Train-subset only, never the full
    pre-split set: prevents val/test label statistics from leaking into the
    training loss. Normalized so the minimum weight is 1.0.

    Returns:
        weights: list[float] of length num_classes
        counts:  list[int]   raw frame counts per class
    """
    counts = torch.zeros(num_classes)
    for s in sequences:
        c = int(s["class"])
        if 0 <= c < num_classes:
            counts[c] += int(s["n_frames"])
    safe = counts.clone()
    safe[safe == 0] = 1.0
    weights = counts.sum() / (num_classes * safe)
    weights = weights / weights.min()
    return weights.tolist(), [int(c.item()) for c in counts]
