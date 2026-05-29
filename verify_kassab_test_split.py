"""Replicate Kassab's extract_tackle_sequences_and_undersample + split_data_by_game
pipeline at 25 FPS over local TACDEC labels. Compare per-class frame counts in
the resulting train/val/test against Kassab's printed values in TempTAC.ipynb:

    train:  [bg=9085, live=4589, replay=8840]
    val:    [bg=1941, live=1398, replay=1861]
    test:   [bg=1474, live=1954, replay=1872]

If our 25-FPS replication matches, the kassab_concat 5-FPS pipeline divergence
is due to stride-5 subsampling alone. If not, the label parsing / split mapping
also differs.

Class IDs (Kassab convention): 0=bg, 1=live, 2=replay.
Kassab's raw 5-class y is remapped via {2->1, 3->2} so live+live-incomplete -> 1
and replay+replay-incomplete -> 2. We mirror that here.

Usage:
    uv run python verify_kassab_test_split.py \
        --labels-dir data/TACDEC/labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Kassab's class scheme (post-remap): 0=bg, 1=live, 2=replay.
KASSAB_BG = 0
KASSAB_LIVE = 1
KASSAB_REPLAY = 2

_LIVE_TYPES = {"tackle-live", "tackle-live-incomplete"}
_REPLAY_TYPES = {"tackle-replay", "tackle-replay-incomplete"}


def build_25fps_labels(label_path: Path) -> np.ndarray:
    """Per-frame labels at 25 FPS in Kassab's class scheme (0=bg, 1=live, 2=replay)."""
    with open(label_path) as f:
        data = json.load(f)
    frame_count = data["media_attributes"]["frame_count"]
    labels = np.full(frame_count, KASSAB_BG, dtype=np.int64)
    for ev in data["events"]:
        t = ev["type"]
        if t in _LIVE_TYPES:
            cls = KASSAB_LIVE
        elif t in _REPLAY_TYPES:
            cls = KASSAB_REPLAY
        else:
            continue
        s, e = ev["frame_start"], ev["frame_end"]
        labels[s : e + 1] = cls
    return labels


def split_sequences_np(indices: np.ndarray) -> list[np.ndarray]:
    """Split a sorted index array into runs of consecutive indices.
    Mirrors what Kassab calls split_sequences_np in TempTAC.ipynb.
    """
    if len(indices) == 0:
        return []
    splits = np.where(np.diff(indices) != 1)[0] + 1
    return np.split(indices, splits)


def extract_tackle_sequences_and_undersample(
    y_global: np.ndarray,
    frame_counts: np.ndarray,
    max_tackles: int = 500,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Faithful replication of TempTAC.ipynb cell 7.

    Returns
    -------
    all_seq : list of int arrays
        Kept sub-sequences as global index arrays. Mirrors Kassab's `all_seq`.
    new_y : np.ndarray
        Concatenated labels for kept sub-sequences (= what Kassab passes
        forward to split_data_by_game).
    new_y_global_idx : np.ndarray
        Global index for each row of new_y (same length as new_y).
    """
    np.random.seed(42)  # mirror Kassab's notebook RNG state at this point
    # NOTE: Kassab's notebook does NOT explicitly seed before cell 7; the global
    # RNG state at that point depends on prior cells. We seed 42 here so the
    # bg-slice randint draws are reproducible -- the SEQUENCE COUNTS we care
    # about don't depend on the slice positions, only on cap binding.

    all_seq: list[np.ndarray] = []
    new_y_pieces: list[np.ndarray] = []
    new_y_idx_pieces: list[np.ndarray] = []

    bg_kept = 0
    replay_kept = 0

    start_idx = 0
    for count in frame_counts:
        end_idx = start_idx + int(count)
        y_slice = y_global[start_idx:end_idx]

        # Foreground runs (non-zero labels), preserved in clip-relative order.
        fg_local = np.where(y_slice != KASSAB_BG)[0]
        fg_global = fg_local + start_idx
        fg_seqs = split_sequences_np(fg_global)

        # Background runs (zero labels).
        bg_local = np.where(y_slice == KASSAB_BG)[0]
        bg_global = bg_local + start_idx
        bg_seqs = split_sequences_np(bg_global)

        # Kassab iterates bg first, then fg, per clip.
        for bg_seq in bg_seqs:
            if len(bg_seq) < 70:
                continue
            # Kassab guards the bg cap on first-frame class == 0 (always true
            # for bg_seqs). When cap binds, the sequence is dropped entirely.
            if bg_kept >= max_tackles:
                continue
            bg_kept += 1
            start_random = int(np.random.randint(35, len(bg_seq) - 34))
            random_seq = bg_seq[start_random : start_random + 25]
            all_seq.append(random_seq)
            new_y_pieces.append(y_global[random_seq[0] : random_seq[-1] + 1])
            new_y_idx_pieces.append(np.arange(random_seq[0], random_seq[-1] + 1))

        for seq in fg_seqs:
            if seq.size == 0:
                continue
            first_cls = int(y_global[seq[0]])
            if first_cls == KASSAB_REPLAY:
                if replay_kept >= 280:
                    continue
                replay_kept += 1
            # Live is uncapped.
            all_seq.append(seq)
            new_y_pieces.append(y_global[seq[0] : seq[-1] + 1])
            new_y_idx_pieces.append(np.arange(seq[0], seq[-1] + 1))

        start_idx = end_idx

    new_y = np.concatenate(new_y_pieces) if new_y_pieces else np.array([], dtype=np.int64)
    new_y_idx = np.concatenate(new_y_idx_pieces) if new_y_idx_pieces else np.array([], dtype=np.int64)
    return all_seq, new_y, new_y_idx


def split_data_by_game(
    new_y: np.ndarray,
    new_y_global_idx: np.ndarray,
    frame_counts: np.ndarray,
    seed: int = 42,
    split_ratio: tuple[float, float, float] = (0.70, 0.15, 0.15),
    mode: str = "correct",
) -> dict[str, np.ndarray]:
    """Faithful replication of TempTAC.ipynb cell 10 (split_data_by_game).

    Parameters
    ----------
    mode : {"correct", "kassab_bug"}
        * "correct"     -- filter by actual game membership via
                            kept_indices (a real game-disjoint split).
        * "kassab_bug"  -- replicate Kassab's extract_data bug: slice the
                            undersampled `new_y` by ORIGINAL frame_counts
                            boundaries (which mostly index off the end of
                            new_y for games beyond clip ~52, and for games
                            within range the slice does NOT correspond to
                            that game's actual content -- it returns rows
                            of new_y at positions [boundaries[g-1],
                            boundaries[g]) regardless of origin clip).
    """
    if mode not in ("correct", "kassab_bug"):
        raise ValueError(f"unknown mode: {mode}")

    np.random.seed(seed)
    boundaries = np.cumsum(frame_counts)  # exclusive upper bounds per game
    game_indices = np.arange(len(frame_counts))
    np.random.shuffle(game_indices)

    total = len(game_indices)
    n_train = int(total * split_ratio[0])
    n_val = int(total * split_ratio[1])

    train_games = game_indices[:n_train]
    val_games = game_indices[n_train : n_train + n_val]
    test_games = game_indices[n_train + n_val :]

    def collect_correct(games: np.ndarray) -> np.ndarray:
        labels_out: list[np.ndarray] = []
        for g in games:
            lo = 0 if g == 0 else int(boundaries[g - 1])
            hi = int(boundaries[g])
            mask = (new_y_global_idx >= lo) & (new_y_global_idx < hi)
            labels_out.append(new_y[mask])
        return np.concatenate(labels_out) if labels_out else np.array([], dtype=np.int64)

    def collect_kassab_bug(games: np.ndarray) -> np.ndarray:
        n = len(new_y)
        labels_out: list[np.ndarray] = []
        for g in games:
            start = 0 if g == 0 else int(boundaries[g - 1])
            end = int(boundaries[g])
            # Kassab's bug: index new_y (length n) with original-frame-count
            # boundaries (which go up to 271,902). Python slicing clips to
            # [0, n], so for games past the first ~52 the slice is empty.
            if start >= n:
                continue
            labels_out.append(new_y[start : min(end, n)])
        return np.concatenate(labels_out) if labels_out else np.array([], dtype=np.int64)

    collect = collect_correct if mode == "correct" else collect_kassab_bug
    return {
        "train": collect(train_games),
        "val":   collect(val_games),
        "test":  collect(test_games),
        "train_games": train_games,
        "val_games":   val_games,
        "test_games":  test_games,
    }


def per_class_counts(y: np.ndarray) -> list[int]:
    """Returns [#bg, #live, #replay] in Kassab's class scheme."""
    return [int(np.sum(y == c)) for c in (KASSAB_BG, KASSAB_LIVE, KASSAB_REPLAY)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels-dir", required=True,
                    help="Directory of TACDEC label JSONs. Each <stem>.json "
                         "corresponds to one <stem>.mp4 in Kassab's videos dir.")
    args = ap.parse_args()

    label_dir = Path(args.labels_dir)
    stems = sorted(p.stem for p in label_dir.glob("*.json"))
    if len(stems) != 425:
        print(f"WARNING: found {len(stems)} label JSONs, expected 425.")

    # Build the global y array + frame_counts in sorted-stem order.
    y_pieces: list[np.ndarray] = []
    frame_counts: list[int] = []
    for stem in stems:
        labels = build_25fps_labels(label_dir / f"{stem}.json")
        y_pieces.append(labels)
        frame_counts.append(len(labels))
    y_global = np.concatenate(y_pieces)
    frame_counts_arr = np.array(frame_counts, dtype=np.int64)
    print(f"Loaded {len(stems)} clips, total frames at 25 FPS = {len(y_global):,}")

    # Step 1: extract + undersample
    all_seq, new_y, new_y_idx = extract_tackle_sequences_and_undersample(
        y_global, frame_counts_arr, max_tackles=500
    )
    pool_counts = per_class_counts(new_y)
    print(f"\nPool after undersampling (Kassab order [bg, live, replay]):")
    print(f"  ours    : {pool_counts}  total={sum(pool_counts):,}")
    print(f"  Kassab  : [12500, 7941, 12573]  total=33,014")

    expected = {
        "train": [9085, 4589, 8840],
        "val":   [1941, 1398, 1861],
        "test":  [1474, 1954, 1872],
    }

    for mode in ("correct", "kassab_bug"):
        splits = split_data_by_game(new_y, new_y_idx, frame_counts_arr,
                                      seed=42, split_ratio=(0.70, 0.15, 0.15),
                                      mode=mode)
        print(f"\nPer-class frame counts at 25 FPS, mode={mode!r}")
        print(f"  (Kassab order [bg, live, replay])")
        for split_name in ("train", "val", "test"):
            ours = per_class_counts(splits[split_name])
            exp = expected[split_name]
            match = "MATCH" if ours == exp else "DIFFER"
            print(f"  {split_name:5s}  ours={ours}  expected={exp}  -> {match}")



if __name__ == "__main__":
    main()
