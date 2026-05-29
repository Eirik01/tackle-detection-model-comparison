"""Class-balanced window construction for the DINOv3 / V-JEPA 2 temporal probes.

Companion to ``splits.py``. Operates at 5 FPS with W=10 frame windows
(2 s of temporal context), matching both attentive-probe configurations.

Protocol (3 classes after the 5->3 merge in ``tacdec_dataset.py``):

1. Game-disjoint train/val/test split is delegated to
   ``splits.split_games`` for parity between the spatial and temporal
   evaluations.

2. Frame timeline is subsampled 25 FPS -> 5 FPS by stride-5 indexing:
   5-FPS frame ``k`` corresponds to 25-FPS frame ``5k``. The same indexing maps
   event intervals from 25 FPS to 5 FPS.

3. Balanced training set (per split): one W=10 window per sample.
     * tackle-live : every event whose clamped 5-FPS center frame still has
       the event's class label. Events at the first or last few frames of a
       clip whose window has to be clip-clamped past the event interval are
       dropped (typically ~5% of foreground events) so the "center-frame
       label = window label" rule holds strictly.
     * tackle-replay : random ``N = #live (post-filter)`` events sampled
       without replacement with a fixed seed; same center-label filter.
     * background : random ``N`` eligible background segments sampled without
       replacement, one window per segment with uniform-random center inside
       the eligible region.

4. Test-time evaluation: per-clip stride-1 sliding of W=10 at 5 FPS. No
   concatenation across clips, so the V-JEPA 2 attentive probe (which extracts
   features per clip) and the DINOv3 attentive probe see the same windows.

Window labelling uses the center-frame label. Required because live events at
5 FPS are typically shorter than W=10, so a majority-label rule would erase the
live class.

Eligibility for a background window center:
  * Distance from any real event >= EVENT_GAP_5FPS frames at 5 FPS.
  * Window fully inside the clip: center in ``[CENTER, n5 - W + CENTER]``.
  * Enclosing eligible run is at least ``MIN_BG_SEG_5FPS`` frames long.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from data.labels import (
    BACKGROUND,
    LIVE_TYPES as _LIVE_TYPES,
    REPLAY_TYPES as _REPLAY_TYPES,
    TACKLE_LIVE,
    TACKLE_REPLAY,
)
from data.splits import (
    build_frame_labels,
    split_games,           # re-export so callers have a single import surface
    split_games_from_file, # re-export for the --split-file override
)

__all__ = [
    "BACKGROUND",
    "TACKLE_LIVE",
    "TACKLE_REPLAY",
    "W",
    "CENTER",
    "STRIDE_S",
    "MIN_BG_SEG_5FPS",
    "EVENT_GAP_5FPS",
    "KASSAB_BG_MIN_LEN_5FPS",
    "KASSAB_BG_SLICE_LEN_5FPS",
    "KASSAB_BG_SLICE_MARGIN_5FPS",
    "split_games",
    "split_games_from_file",
    "labels_at_5fps",
    "extract_events_5fps",
    "build_balanced_windows",
    "build_kassab_concat_stream",
    "build_kassab_concat_windows",
    "build_kassab_concat_windows_from_manifest",
    "kassab_buggy_split",
]

W = 10                  # window length in 5-FPS frames (= 2.0 s)
CENTER = W // 2 - 1     # 4. Lower-middle center (even W -> no exact middle).
STRIDE_S = 5            # 25 FPS -> 5 FPS subsample stride
MIN_BG_SEG_5FPS = 20    # min eligible background segment length at 5 FPS
EVENT_GAP_5FPS = 5      # buffer between any event frame and a bg window center

# Kassab TempTAC's per-sequence retention rule scaled from 25 FPS to 5 FPS.
# Notebook uses min bg run length >= 70, slice length 25, margin 35/34 on
# either side. Divide by STRIDE_S = 5 (round to integer) for the 5-FPS twins.
KASSAB_BG_MIN_LEN_5FPS = 14   # 70 / 5
KASSAB_BG_SLICE_LEN_5FPS = 5  # 25 / 5
KASSAB_BG_SLICE_MARGIN_5FPS = 7  # ~35 / 5



def labels_at_5fps(labels_25fps: np.ndarray) -> np.ndarray:
    """Subsample a 25-FPS per-frame label array to 5 FPS via stride-5 indexing."""
    return labels_25fps[::STRIDE_S]


def _valid_center_range(n5: int) -> Tuple[int, int]:
    """Inclusive ``(min, max)`` window-center indices for a clip with ``n5`` 5-FPS frames."""
    return CENTER, n5 - W + CENTER  # window covers [c - CENTER, c + W - 1 - CENTER]


def extract_events_5fps(label_path: str | Path) -> List[Tuple[int, int, int]]:
    """Return ``[(class, start_5fps, end_5fps)]`` for the foreground events in one clip.

    Event 5-FPS extent is ``[ceil(start/5), floor(end/5)]`` when that range is
    non-empty; otherwise the event is collapsed to a single 5-FPS frame at the
    rounded midpoint so very short events still produce a valid center.
    """
    with open(label_path) as f:
        data = json.load(f)

    events: List[Tuple[int, int, int]] = []
    for ev in data["events"]:
        et = ev["type"]
        if et in _LIVE_TYPES:
            cls = TACKLE_LIVE
        elif et in _REPLAY_TYPES:
            cls = TACKLE_REPLAY
        else:
            continue
        s_25, e_25 = ev["frame_start"], ev["frame_end"]
        s_5 = (s_25 + STRIDE_S - 1) // STRIDE_S  # ceil
        e_5 = e_25 // STRIDE_S                   # floor
        if e_5 < s_5:
            mid = round((s_25 + e_25) / 2 / STRIDE_S)
            s_5 = e_5 = int(mid)
        events.append((cls, s_5, e_5))
    return events


def _eligible_background_segments(
    n5: int, events_5fps: List[Tuple[int, int, int]]
) -> List[Tuple[int, int]]:
    """Half-open ``(start, end)`` runs of valid background window-center positions.

    A 5-FPS index is eligible iff it (a) is a valid window center (window fits
    inside the clip), and (b) is at least ``EVENT_GAP_5FPS`` frames away from
    every event frame. Only runs of length ``>= MIN_BG_SEG_5FPS`` are returned.
    """
    blocked = np.zeros(n5, dtype=bool)
    for _, s_5, e_5 in events_5fps:
        lo = max(0, s_5 - EVENT_GAP_5FPS)
        hi = min(n5, e_5 + EVENT_GAP_5FPS + 1)
        blocked[lo:hi] = True

    min_c, max_c = _valid_center_range(n5)
    valid_center = np.zeros(n5, dtype=bool)
    if max_c >= min_c:
        valid_center[min_c : max_c + 1] = True

    eligible = (~blocked) & valid_center

    segs: List[Tuple[int, int]] = []
    i = 0
    while i < n5:
        if not eligible[i]:
            i += 1
            continue
        j = i
        while j < n5 and eligible[j]:
            j += 1
        if (j - i) >= MIN_BG_SEG_5FPS:
            segs.append((i, j))
        i = j
    return segs


def build_balanced_windows(
    clip_ids: List[str],
    label_dir: str | Path,
    seed: int = 0,
) -> List[Tuple[str, int, int]]:
    """Build a class-balanced training set of W=10 window centers at 5 FPS.

    Returns a shuffled list of ``(clip_id, center_frame_5fps, class)`` tuples.
    Sample count per class equals ``#tackle-live events`` in ``clip_ids``.
    Raises if any class cannot be filled (missing live events, too few replay
    events, or too few eligible background segments).
    """
    label_dir = Path(label_dir)
    rng = np.random.default_rng(seed)

    live_pool: List[Tuple[str, int]] = []
    replay_pool: List[Tuple[str, int]] = []
    bg_seg_pool: List[Tuple[str, int, int]] = []  # (clip_id, seg_start, seg_end)

    for clip_id in clip_ids:
        path = label_dir / f"{clip_id}.json"
        labels_25 = build_frame_labels(path)
        labels_5 = labels_at_5fps(labels_25)
        n5 = len(labels_5)
        events = extract_events_5fps(path)

        min_c, max_c = _valid_center_range(n5)
        if max_c < min_c:
            continue  # clip too short to host any window

        for cls, s_5, e_5 in events:
            center = (s_5 + e_5) // 2
            center = max(min_c, min(max_c, center))
            # Enforce the center-frame labelling rule strictly: if clip-boundary
            # clamping or subsample misalignment moves the center off the event,
            # drop it rather than mint a window whose label disagrees with the
            # 5-FPS ground truth used at evaluation time.
            if int(labels_5[center]) != cls:
                continue
            if cls == TACKLE_LIVE:
                live_pool.append((clip_id, center))
            elif cls == TACKLE_REPLAY:
                replay_pool.append((clip_id, center))

        for s, e in _eligible_background_segments(n5, events):
            bg_seg_pool.append((clip_id, s, e))

    n_target = len(live_pool)
    if n_target == 0:
        raise ValueError(
            "No tackle-live events in the given clips; cannot balance temporal "
            "windows. Check the split."
        )
    if len(replay_pool) < n_target:
        raise ValueError(
            f"Only {len(replay_pool)} tackle-replay events available, "
            f"need {n_target} to balance against tackle-live."
        )
    if len(bg_seg_pool) < n_target:
        raise ValueError(
            f"Only {len(bg_seg_pool)} eligible background segments "
            f"(>= {MIN_BG_SEG_5FPS} frames, >= {EVENT_GAP_5FPS}-frame gap from "
            f"events); need {n_target}."
        )

    replay_idx = rng.choice(len(replay_pool), size=n_target, replace=False)
    replay_sel = [replay_pool[i] for i in replay_idx]

    bg_seg_idx = rng.choice(len(bg_seg_pool), size=n_target, replace=False)
    bg_sel: List[Tuple[str, int]] = []
    for i in bg_seg_idx:
        clip_id, s, e = bg_seg_pool[i]
        center = int(rng.integers(s, e))  # uniform on [s, e)
        bg_sel.append((clip_id, center))

    balanced = (
        [(c, k, TACKLE_LIVE) for c, k in live_pool]
        + [(c, k, TACKLE_REPLAY) for c, k in replay_sel]
        + [(c, k, BACKGROUND) for c, k in bg_sel]
    )
    order = rng.permutation(len(balanced))
    return [balanced[i] for i in order]


# ---- Kassab TempTAC concat-and-slide (strict parity) -----------------------


def _runs_of_true(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open ``[start, end)`` runs of consecutive ``True`` values in ``mask``."""
    runs: List[Tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def build_kassab_concat_stream(
    clip_ids: List[str],
    label_dir: str | Path,
    seed: int = 0,
    replay_cap: int = 280,
    bg_cap: int = 500,
    bg_slice_len: int = KASSAB_BG_SLICE_LEN_5FPS,
    bg_min_len: int = KASSAB_BG_MIN_LEN_5FPS,
    bg_slice_margin: int = KASSAB_BG_SLICE_MARGIN_5FPS,
) -> List[Dict]:
    """Replicate Kassab's ``extract_tackle_sequences_and_undersample`` at 5 FPS.

    Iterates the clips in the order given. For each clip:

    1. Identify background runs (contiguous 5-FPS frames with class BACKGROUND)
       of length >= ``bg_min_len``. For each such run, if the running background
       count is below ``bg_cap``, sample a slice of ``bg_slice_len`` frames
       starting at a uniform random offset in
       ``[bg_slice_margin, run_len - bg_slice_margin)`` (matches Kassab's
       ``np.random.randint(35, len-34)`` after STRIDE_S=5 scaling). The slice
       is kept whole; the running bg counter increments by one.
    2. Identify foreground runs (contiguous 5-FPS frames with any non-background
       class). The run's "label" is the class of its first frame (matches
       Kassab's ``y[seq[0]]`` book-keeping). A run starting with TACKLE_REPLAY
       is dropped once the replay counter hits ``replay_cap``; otherwise the
       run is kept whole. TACKLE_LIVE-first runs are uncapped.

    Per-clip emission order is BG runs first, then FG runs (matches the
    notebook's nested loops). Cross-clip order is the iteration order of
    ``clip_ids``.

    Returns one dict per kept sub-sequence:
        {
            "clip_id":       str,
            "first_cls":     int (class of the first frame; for cap accounting),
            "local_indices": np.ndarray[int64], 5-FPS frame indices in the clip,
            "labels":        np.ndarray[int64], per-frame class labels,
        }
    The concatenation of all ``local_indices`` (in list order) defines the
    global frame stream that ``build_kassab_concat_windows`` slides over.
    """
    if bg_slice_len <= 0:
        raise ValueError(f"bg_slice_len must be > 0, got {bg_slice_len}")
    # Draw range matches Kassab's np.random.randint(35, len-34) after STRIDE_S
    # rescaling: start in [bg_slice_margin, run_len - bg_slice_margin] inclusive
    # (= [lo, hi) with hi = run_len - bg_slice_margin + 1). Need bg_min_len
    # >= 2 * bg_slice_margin for the range to be non-empty AND bg_slice_len
    # <= bg_slice_margin so the slice always fits inside the run.
    if bg_min_len < 2 * bg_slice_margin:
        raise ValueError(
            f"bg_min_len ({bg_min_len}) must be >= 2 * bg_slice_margin "
            f"({bg_slice_margin}); otherwise the random offset range is empty."
        )
    if bg_slice_len > bg_slice_margin:
        raise ValueError(
            f"bg_slice_len ({bg_slice_len}) must be <= bg_slice_margin "
            f"({bg_slice_margin}); otherwise the slice can run off the end "
            f"of the bg run."
        )
    label_dir = Path(label_dir)
    rng = np.random.default_rng(seed)

    sub_seqs: List[Dict] = []
    bg_kept = 0
    replay_kept = 0

    for clip_id in clip_ids:
        path = label_dir / f"{clip_id}.json"
        labels_5 = labels_at_5fps(build_frame_labels(path)).astype(np.int64)
        n5 = len(labels_5)
        if n5 == 0:
            continue

        # Background runs first (matches Kassab's order).
        bg_mask = (labels_5 == BACKGROUND)
        for s, e in _runs_of_true(bg_mask):
            run_len = e - s
            if run_len < bg_min_len:
                continue
            if bg_kept >= bg_cap:
                continue
            lo = bg_slice_margin
            hi = run_len - bg_slice_margin + 1  # exclusive (= len - margin inclusive)
            # bg_min_len >= 2*margin guarantees hi > lo so this draw is valid.
            start_offset = int(rng.integers(lo, hi))
            slice_start = s + start_offset
            slice_end = min(slice_start + bg_slice_len, e)
            local_indices = np.arange(slice_start, slice_end, dtype=np.int64)
            sub_seqs.append({
                "clip_id":       clip_id,
                "first_cls":     int(BACKGROUND),
                "local_indices": local_indices,
                "labels":        labels_5[local_indices],
            })
            bg_kept += 1

        # Foreground runs (live + replay; class taken from the first frame).
        fg_mask = (labels_5 != BACKGROUND)
        for s, e in _runs_of_true(fg_mask):
            first_cls = int(labels_5[s])
            if first_cls == TACKLE_REPLAY:
                if replay_kept >= replay_cap:
                    continue
                replay_kept += 1
            local_indices = np.arange(s, e, dtype=np.int64)
            sub_seqs.append({
                "clip_id":       clip_id,
                "first_cls":     first_cls,
                "local_indices": local_indices,
                "labels":        labels_5[local_indices],
            })

    return sub_seqs


def _manifest_from_sub_seqs(sub_seqs: List[Dict]) -> List[Tuple[str, int, int]]:
    """Flatten kept sub-sequences into a list of ``(clip_id, local_5fps_idx, label)``
    tuples in concat order. Each tuple is one frame in the global stream."""
    manifest: List[Tuple[str, int, int]] = []
    for sub in sub_seqs:
        clip_id = sub["clip_id"]
        local_indices = sub["local_indices"]
        labels = sub["labels"]
        for idx_pos in range(len(local_indices)):
            manifest.append((clip_id,
                             int(local_indices[idx_pos]),
                             int(labels[idx_pos])))
    return manifest


def build_kassab_concat_windows_from_manifest(
    manifest: List[Tuple[str, int, int]],
    window_size: int = W,
) -> List[Dict]:
    """Slide stride-1 windows of length ``window_size`` over a pre-built manifest
    of ``(clip_id, local_5fps_idx, label)`` tuples. See
    ``build_kassab_concat_windows`` for the per-window dict schema.
    """
    n = len(manifest)
    if n < window_size:
        return []

    windows: List[Dict] = []
    for start in range(n - window_size + 1):
        block = manifest[start : start + window_size]
        frames = [(c, i) for c, i, _ in block]
        labels = np.fromiter((l for _, _, l in block),
                              dtype=np.int64, count=window_size)
        center = block[CENTER]
        windows.append({
            "frames":        frames,
            "labels":        labels,
            "center_label":  int(center[2]),
            "center_frame":  (center[0], int(center[1])),
            "global_start":  start,
            "crosses_clip":  len({c for c, _ in frames}) > 1,
        })
    return windows


def build_kassab_concat_windows(
    sub_seqs: List[Dict],
    window_size: int = W,
) -> List[Dict]:
    """Slide stride-1 windows of length ``window_size`` over the global stream
    obtained by concatenating ``sub_seqs`` in list order. Cross-clip windows
    are kept (the whole point of Kassab's protocol).

    Each window dict contains:
        {
            "frames":        list[(clip_id:str, local_5fps_idx:int)],  length W
            "labels":        np.ndarray[int64], shape (W,) per-frame class labels
            "center_label":  int, the center-frame class (loss target),
            "center_frame":  (clip_id, local_5fps_idx) of the center,
            "global_start":  int, position in the concatenated stream,
            "crosses_clip":  bool, True iff the W frames span >= 2 distinct clips,
        }

    With ``N`` total frames in the concat stream, exactly ``max(0, N-W+1)``
    windows are produced. No padding, no wrap-around.
    """
    return build_kassab_concat_windows_from_manifest(
        _manifest_from_sub_seqs(sub_seqs), window_size=window_size,
    )


def kassab_buggy_split(
    manifest: List[Tuple[str, int, int]],
    frame_counts_5fps: np.ndarray,
    seed: int = 42,
    split_ratio: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Dict[str, List[Tuple[str, int, int]]]:
    """Faithfully replicate Kassab TempTAC's ``extract_data`` flat-index slice
    bug (TempTAC.ipynb cell 10) at 5 FPS.

    Background
    ----------
    Kassab's ``split_data_by_game`` (notebook lines 297-336) shuffles game
    indices, then for each game ``g`` calls ``extract_data(games, boundaries,
    new_X, ...)`` which does ``new_X[boundaries[g-1] : boundaries[g]]``. The
    bug: ``new_X`` is the **undersampled** concat tensor (length ~33 k @ 25 FPS,
    ~6.6 k @ 5 FPS), but ``boundaries = np.cumsum(frame_counts)`` is over
    **original** clip frame counts (max ~271 k @ 25 FPS / ~54 k @ 5 FPS).

    Consequences:
    1. For games whose ``boundaries[g-1]`` exceeds the manifest length, the
       slice is empty (most games past clip ~52 in alphabetical order).
    2. For games whose ``[boundaries[g-1], boundaries[g])`` lies within the
       manifest, the slice returns rows of the manifest at those positions
       -- but those rows belong to whatever clips happened to be processed
       in that index range, NOT to game ``g``.

    The resulting train/val/test pools are therefore NOT actually game-disjoint;
    they are positional slices of the global undersampled stream. We replicate
    this exactly so our evaluation pool matches the one Kassab's classification
    reports were computed on.

    Args
    ----
    manifest : list of ``(clip_id, local_5fps_idx, label)``
        The global flattened concat stream (output of
        ``_manifest_from_sub_seqs``).
    frame_counts_5fps : np.ndarray[int64]
        Per-clip 5-FPS row counts in the same clip-iteration order used to
        build the manifest (i.e. sorted clip IDs). The notebook's
        ``frame_counts.npy`` is at 25 FPS; pass ``frame_counts_25fps //
        STRIDE_S`` here for the 5 FPS twin.
    seed : int
        Game-shuffle seed. Default 42 (Kassab's notebook seed).
    split_ratio : tuple
        (train, val, test) fractions. ``int(N*ratio)`` rounding matches Kassab.

    Returns
    -------
    Dict mapping ``"train"``, ``"val"``, ``"test"`` to per-split manifest
    slices (lists of ``(clip_id, local_5fps_idx, label)``). Each slice may
    contain frames from any number of underlying clips because the bug
    silently mixes content.
    """
    np.random.seed(seed)
    boundaries = np.cumsum(frame_counts_5fps)
    n_games = len(frame_counts_5fps)
    game_indices = np.arange(n_games)
    np.random.shuffle(game_indices)

    n_train = int(n_games * split_ratio[0])
    n_val = int(n_games * split_ratio[1])

    train_games = game_indices[:n_train]
    val_games = game_indices[n_train : n_train + n_val]
    test_games = game_indices[n_train + n_val :]

    n_manifest = len(manifest)

    def collect(games: np.ndarray) -> List[Tuple[str, int, int]]:
        rows: List[Tuple[str, int, int]] = []
        for g in games:
            start = 0 if g == 0 else int(boundaries[g - 1])
            end = int(boundaries[g])
            if start >= n_manifest:
                continue
            rows.extend(manifest[start : min(end, n_manifest)])
        return rows

    return {
        "train": collect(train_games),
        "val":   collect(val_games),
        "test":  collect(test_games),
    }
