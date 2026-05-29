"""Game-level split + per-split class balancing for the DINOv3 spatial probe.

Protocol:
1. Split games (NOT clips) 70/15/15 with a seeded RNG, so no game appears in
   more than one split.
2. Within each split, count tackle-live frames. Undersample background and
   tackle-replay frame pools down to that count with a separate seeded RNG.

Class scheme (matches src/data/tacdec_dataset.py:_remap_to_3_classes):
    0 = tackle-live      (tackle-live + tackle-live-incomplete)
    1 = tackle-replay    (tackle-replay + tackle-replay-incomplete)
    2 = background       (frames not covered by any annotated event)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Class scheme lives in data/labels.py; re-exported here so existing callers of
# ``from data.splits import TACKLE_LIVE, ...`` keep working.
from data.labels import (  # noqa: F401
    BACKGROUND,
    LIVE_TYPES as _LIVE_TYPES,
    REPLAY_TYPES as _REPLAY_TYPES,
    TACKLE_LIVE,
    TACKLE_REPLAY,
)


def build_frame_labels(label_path: str | Path) -> np.ndarray:
    """Build per-frame 3-class labels for one TACDEC clip.

    Returns an int64 array of shape (frame_count,) with values in {0, 1, 2}.
    Background is the default; event intervals overwrite (inclusive at both ends).
    """
    with open(label_path) as f:
        data = json.load(f)

    frame_count = data["media_attributes"]["frame_count"]
    labels = np.full(frame_count, BACKGROUND, dtype=np.int64)

    for event in data["events"]:
        event_type = event["type"]
        if event_type in _LIVE_TYPES:
            cls = TACKLE_LIVE
        elif event_type in _REPLAY_TYPES:
            cls = TACKLE_REPLAY
        else:
            continue

        start = event["frame_start"]
        end = event["frame_end"]
        labels[start : end + 1] = cls

    return labels


def _list_clips_by_game(label_dir: Path) -> Dict[int, List[str]]:
    """Group clip IDs by game_id. clip_id = JSON filename stem."""
    clips_by_game: Dict[int, List[str]] = {}
    for path in sorted(label_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        game_id = data["metadata"]["game_id"]
        clip_id = path.stem
        clips_by_game.setdefault(game_id, []).append(clip_id)
    return clips_by_game


def split_games_from_file(path: str | Path) -> Dict[str, List[str]]:
    """Load a pre-computed clip-ID partition from a JSON file.

    Expected schema:
        {"train": [clip_id, ...], "val": [clip_id, ...], "test": [clip_id, ...]}

    Used for reproducing external splits exactly -- e.g., Kassab TempTAC's
    video-level 70/15/15 partition (see scripts/dump_kassab_split.py).
    Clip IDs are sorted on load so the order is reproducible regardless of
    how the JSON was written. No frac/seed are consulted when this path is
    used; the override is bit-identical to the file's contents.
    """
    with open(path) as f:
        data = json.load(f)
    for key in ("train", "val", "test"):
        if key not in data or not isinstance(data[key], list):
            raise ValueError(
                f"split file {path!r} missing list field {key!r}"
            )
    return {
        "train": sorted(data["train"]),
        "val":   sorted(data["val"]),
        "test":  sorted(data["test"]),
    }


def split_games(
    label_dir: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Partition clips into train/val/test such that no game spans two splits.

    Games are shuffled with `seed`, then assigned to test, val, train in that
    order according to the requested fractions (computed on game counts, not
    clip counts — TACDEC has roughly one clip per game so the difference is small).
    """
    label_dir = Path(label_dir)
    clips_by_game = _list_clips_by_game(label_dir)

    game_ids = sorted(clips_by_game.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)

    n_games = len(game_ids)
    n_test = int(round(n_games * test_frac))
    n_val = int(round(n_games * val_frac))

    test_games = game_ids[:n_test]
    val_games = game_ids[n_test : n_test + n_val]
    train_games = game_ids[n_test + n_val :]

    def collect(games: List[int]) -> List[str]:
        return sorted(c for g in games for c in clips_by_game[g])

    return {
        "train": collect(train_games),
        "val": collect(val_games),
        "test": collect(test_games),
    }


def kfold_split_games(
    label_dir: str | Path,
    n_folds: int,
    fold_idx: int,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Game-level k-fold partition with the same {train, val, test} contract.

    Games are shuffled once with `seed`, then chunked into `n_folds` blocks via
    np.array_split. Block `fold_idx` becomes the test set. The remaining blocks
    are concatenated, and the last `round(val_frac * n_games)` games of that
    concatenation form the val set; the rest form the train set.

    Holding `seed` fixed while varying `fold_idx` guarantees the global game
    ordering — and therefore the train/val/test partition — is reproducible
    across folds.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if not (0 <= fold_idx < n_folds):
        raise ValueError(f"fold_idx must be in [0, {n_folds}), got {fold_idx}")

    label_dir = Path(label_dir)
    clips_by_game = _list_clips_by_game(label_dir)

    game_ids = sorted(clips_by_game.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(game_ids)

    n_games = len(game_ids)
    folds = [list(f) for f in np.array_split(game_ids, n_folds)]
    test_games = folds[fold_idx]
    remaining = [g for i, fold in enumerate(folds) if i != fold_idx for g in fold]

    n_val = max(1, round(val_frac * n_games))
    n_val = min(n_val, len(remaining) - 1)
    val_games = remaining[-n_val:]
    train_games = remaining[:-n_val]

    def collect(games: List[int]) -> List[str]:
        return sorted(c for g in games for c in clips_by_game[g])

    return {
        "train": collect(train_games),
        "val": collect(val_games),
        "test": collect(test_games),
    }


def balance_split(
    clip_ids: List[str],
    labels_by_clip: Dict[str, np.ndarray],
    seed: int = 0,
) -> List[Tuple[str, int, int]]:
    """Build a class-balanced frame pool for one split.

    Counts tackle-live frames across the given clips, then randomly undersamples
    background and tackle-replay pools to that count. All tackle-live frames are
    kept (the minority class is the binding constraint).

    Returns a flat list of (clip_id, frame_idx, class) tuples, shuffled with `seed`.
    """
    rng = np.random.default_rng(seed)

    pools: Dict[int, List[Tuple[str, int]]] = {
        TACKLE_LIVE: [],
        TACKLE_REPLAY: [],
        BACKGROUND: [],
    }
    for clip_id in clip_ids:
        labels = labels_by_clip[clip_id]
        for cls in (TACKLE_LIVE, TACKLE_REPLAY, BACKGROUND):
            idxs = np.flatnonzero(labels == cls)
            pools[cls].extend((clip_id, int(i)) for i in idxs)

    target = len(pools[TACKLE_LIVE])
    if target == 0:
        raise ValueError(
            "No tackle-live frames in this split; cannot balance. "
            "Check the split seed or game-level partitioning."
        )

    balanced: List[Tuple[str, int, int]] = []
    for cls in (TACKLE_LIVE, TACKLE_REPLAY, BACKGROUND):
        pool = pools[cls]
        if len(pool) > target:
            chosen_idx = rng.choice(len(pool), size=target, replace=False)
            pool = [pool[i] for i in chosen_idx]
        balanced.extend((clip_id, frame_idx, cls) for clip_id, frame_idx in pool)

    order = rng.permutation(len(balanced))
    return [balanced[i] for i in order]
