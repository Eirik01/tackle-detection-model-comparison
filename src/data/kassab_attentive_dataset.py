"""
Kassab-style window-list dataset for the patch-token attentive probe protocol.

This is a parallel pipeline to ``kassab_dataset.py`` (BiLSTM-style concat
stream + center-frame label) but tailored to the attentive probe comparison:

  - Each item is ONE training window: (video_id, anchor_row, label).
  - Features are loaded lazily per item: per-frame patch tokens for DINOv3
    (gather W consecutive feature rows -> [W*256, 1024]) or per-window dense
    tokens for V-JEPA2 (one cached entry -> [N_tokens, 1024]).
  - Window centers come from Kassab's subsampling rules (cell-7 of TempTAC.ipynb)
    expressed in target-FPS feature-row space, scaled from the 25 FPS originals:
        bg_chunk     :  25 source frames -> round(25 * target_fps / 25) rows
        bg_min_seg   :  70 source frames -> round(70 * target_fps / 25) rows
        replay_cap   :  280 (unchanged)
        live         :  uncapped
  - Game-disjoint 70/15/15 split via legacy np.random (matches cell-9, mirrors
    kassab_dataset.split_by_game).

Feature file conventions:
    DINOv3 dense  : ``{video_id}_dinov3_l_{fps}fps_dense_features.npy``  (fp16)
                    shape (T, num_patches, D), one row per frame at target_fps.
    V-JEPA2 dense : ``{video_id}_vjepa2_l_{fps}fps_dense_w{W}.npz``      (fp16)
                    shape (T_windows, N_tokens, D) where each row is one V-JEPA2
                    forward over W raw frames at the configured stride.

For both backbones, ``anchor_row`` is the row index in the target-FPS feature
stream. DINOv3 gathers [anchor_row - W//2 + 1 : anchor_row + W//2 + 1] (W rows)
and flattens to a token grid. V-JEPA2 reads the single row at anchor_row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.kassab_dataset import (
    CLASS_BACKGROUND,
    CLASS_NAMES,
    CLASS_TACKLE_LIVE,
    CLASS_TACKLE_REPLAY,
)


# ---- Label loading at target FPS --------------------------------------------

# 5 -> 3 class merge (matches TACDECSpottingDataset)
_LABEL_NAME_TO_CLASS = {
    "tackle-live": CLASS_TACKLE_LIVE,
    "tackle-replay": CLASS_TACKLE_REPLAY,
    "tackle-live-incomplete": CLASS_TACKLE_LIVE,
    "tackle-replay-incomplete": CLASS_TACKLE_REPLAY,
}


def _load_video_labels_at_target_fps(label_path: Path, target_fps: float):
    """
    Build a per-row label array at target_fps for one video.

    Returns
    -------
    labels : np.ndarray, shape (T_target,), dtype int64
        Per-row class label in {bg, tackle-live, tackle-replay}.
    game_id : int
    n_target : int
        Length of `labels` (== number of rows in the dense feature file at this fps).
    stride : int
        Sampling stride (source FPS / target FPS, integer-rounded).
    """
    with open(label_path, "r") as f:
        data = json.load(f)
    src_fps = float(data["media_attributes"]["frame_rate"])
    n_src = int(data["media_attributes"].get("frame_count", 0))
    stride = max(1, int(src_fps / target_fps))

    src_labels = np.full(max(n_src, 1), CLASS_BACKGROUND, dtype=np.int64)
    for event in data.get("events", []):
        cls = _LABEL_NAME_TO_CLASS.get(event["type"])
        if cls is None:
            continue
        s = max(0, int(event["frame_start"]))
        e = min(len(src_labels) - 1, int(event["frame_end"]))
        if e >= s:
            src_labels[s : e + 1] = cls

    # Sub-sample to target FPS by stride. Mirrors what the extractors do
    # (`base_extractor._compute_stride` -> int floor of source/target).
    target_labels = src_labels[::stride]
    n_target = len(target_labels)

    game_id = int(data.get("metadata", {}).get("game_id", -1))
    return target_labels, game_id, n_target, stride


# ---- Sequence extraction (target-FPS feature-row space) ---------------------


def extract_sequences_per_video(labels_dir: Path, target_fps: float):
    """
    Walk every label JSON in alphabetical order (matches the extractor's
    ``sorted(glob)`` traversal), emit one record per maximal contiguous run.
    All frame indices are in target-FPS feature-row space.
    """
    sequences = []
    for label_path in sorted(Path(labels_dir).glob("*.json")):
        labels, game_id, n_target, stride = _load_video_labels_at_target_fps(
            label_path, target_fps
        )
        if game_id == -1 or n_target == 0:
            continue
        run_start = 0
        for i in range(1, n_target + 1):
            if i == n_target or labels[i] != labels[run_start]:
                sequences.append({
                    "video_id":    label_path.stem,
                    "game_id":     game_id,
                    "class":       int(labels[run_start]),
                    "start_frame": run_start,        # in target_fps rows
                    "end_frame":   i,                # exclusive
                    "n_frames":    i - run_start,
                    "stride":      stride,
                    "n_target":    n_target,
                })
                run_start = i
    return sequences


# ---- Kassab subsampling rules in target-FPS space ---------------------------


def _scale_from_25fps(n: int, target_fps: float) -> int:
    """Scale a frame count specified at 25 FPS to target_fps. >= 1."""
    return max(1, int(round(n * target_fps / 25.0)))


def sample_background_chunks(sequences, target_fps,
                             target_count=500, chunk_frames=None,
                             min_segment_frames=None, seed=42):
    """
    Walk bg segments in data order, take the first ``target_count`` whose
    length is >= ``min_segment_frames``, pick a random start that places the
    chunk past a leading buffer scaled from Kassab's 35 source-frame buffer.

    Defaults match the 25 FPS Kassab protocol scaled to target_fps:
        chunk_frames       = round(25 * target_fps / 25) = round(target_fps)
        min_segment_frames = round(70 * target_fps / 25)
        leading_buffer     = round(35 * target_fps / 25)
    """
    if chunk_frames is None:
        chunk_frames = _scale_from_25fps(25, target_fps)
    if min_segment_frames is None:
        min_segment_frames = max(2 * chunk_frames + 2,
                                 _scale_from_25fps(70, target_fps))
    leading = _scale_from_25fps(35, target_fps)

    rng = np.random.default_rng(seed)
    chunks = []
    for s in sequences:
        if s["class"] != CLASS_BACKGROUND:
            continue
        if s["n_frames"] < min_segment_frames:
            continue
        if len(chunks) >= target_count:
            break
        # Mirror cell-7's randint(35, n - 34) => start in [leading, n - chunk_frames).
        max_start = s["n_frames"] - chunk_frames
        if max_start <= leading:
            start_in_seg = leading if leading < s["n_frames"] - chunk_frames else 0
        else:
            start_in_seg = int(rng.integers(leading, max_start))
        global_start = s["start_frame"] + start_in_seg
        chunks.append({
            "video_id":    s["video_id"],
            "game_id":     s["game_id"],
            "class":       CLASS_BACKGROUND,
            "start_frame": global_start,
            "end_frame":   global_start + chunk_frames,
            "n_frames":    chunk_frames,
            "stride":      s["stride"],
            "n_target":    s["n_target"],
        })
    if len(chunks) < target_count:
        raise RuntimeError(
            f"Only {len(chunks)} bg segments with n_frames >= {min_segment_frames} "
            f"found at target_fps={target_fps}; need {target_count}."
        )
    return chunks


def build_kassab_attentive_sequences(labels_dir, target_fps,
                                      bg_count=500, replay_cap=280,
                                      bg_chunk_frames=None,
                                      bg_min_segment_frames=None,
                                      seed=42):
    """
    Produce the Kassab subsampled sequence list at target_fps. Tackle-live is
    uncapped, tackle-replay capped at ``replay_cap``, background subsampled to
    ``bg_count`` chunks via the Kassab rules.
    """
    raw = extract_sequences_per_video(labels_dir, target_fps)

    out = []
    replay_kept = 0
    for s in raw:
        if s["class"] == CLASS_TACKLE_LIVE:
            out.append(s)
        elif s["class"] == CLASS_TACKLE_REPLAY:
            if replay_kept < replay_cap:
                out.append(s)
                replay_kept += 1

    out.extend(sample_background_chunks(
        raw,
        target_fps=target_fps,
        target_count=bg_count,
        chunk_frames=bg_chunk_frames,
        min_segment_frames=bg_min_segment_frames,
        seed=seed,
    ))
    return out


def split_by_game(sequences, train=0.70, val=0.15, seed=42):
    """
    Game-disjoint split with the legacy np.random RNG (matches cell-9 of
    TempTAC.ipynb and ``kassab_dataset.split_by_game`` byte-for-byte).
    """
    games = sorted({s["game_id"] for s in sequences})
    np.random.seed(seed)
    np.random.shuffle(games)
    n = len(games)
    n_train = int(n * train)
    n_val = int(n * val)
    train_games = set(games[:n_train])
    val_games = set(games[n_train:n_train + n_val])
    test_games = set(games[n_train + n_val:])

    splits = {
        "train": [s for s in sequences if s["game_id"] in train_games],
        "val":   [s for s in sequences if s["game_id"] in val_games],
        "test":  [s for s in sequences if s["game_id"] in test_games],
    }
    game_ids = {
        "train": sorted(train_games),
        "val":   sorted(val_games),
        "test":  sorted(test_games),
    }
    return splits, game_ids


# ---- Window list ------------------------------------------------------------


def build_window_list(sequences, window_size: int, anchor_stride=None,
                      intra_window_stride=None):
    """
    Convert sequence records into one window per record: anchor row = sequence
    midpoint. Windows are defined by (video_id, anchor_row, class). The
    feature loader resolves anchor_row to a [W, ...] feature slice at fetch
    time.

    Boundary policy: Kassab no-pad. Each sequence's anchor is clamped into
    the per-video ``valid_anchor_range``; sequences from videos too short to
    contain even one valid window are dropped (with a warning).

    Args:
        sequences: list of sequence records (must contain stride and n_target).
        window_size: target-FPS frames per window.
        anchor_stride: source-frame stride between adjacent target-FPS rows.
            If None, inferred from each sequence's `stride` field.
        intra_window_stride: source-frame stride between adjacent frames inside
            one window. If None, defaults to `anchor_stride` (matches the
            shared 5 FPS protocol).
    """
    from window_protocol import valid_anchor_range

    windows = []
    skipped_short = 0
    for s in sequences:
        center = (s["start_frame"] + s["end_frame"] - 1) // 2

        a_stride = anchor_stride if anchor_stride is not None else s["stride"]
        iw_stride = intra_window_stride if intra_window_stride is not None else a_stride

        # n_target is in target-FPS rows; valid_anchor_range expects video
        # length in source frames. Convert.
        n_source = s["n_target"] * a_stride
        valid_lo, valid_hi = valid_anchor_range(
            video_length=n_source,
            anchor_stride=a_stride,
            intra_window_stride=iw_stride,
            window_length=window_size,
        )
        if valid_hi < valid_lo:
            skipped_short += 1
            continue

        anchor = max(valid_lo, min(valid_hi, center))
        windows.append({
            "video_id":  s["video_id"],
            "game_id":   s["game_id"],
            "class":     s["class"],
            "anchor":    int(anchor),
            "n_target":  s["n_target"],
            "stride":    s["stride"],
        })
    if skipped_short:
        print(f"  build_window_list: skipped {skipped_short} sequences "
              f"from videos too short for W={window_size} at this stride.")
    return windows


# ---- Feature loaders --------------------------------------------------------


class _FeatureLoader:
    """
    Lazy per-video feature loader with a tiny LRU. Subclasses define
    ``_open_video`` (returning a numpy-array-like) and ``_window_tokens``
    (extracting the [N_tokens, D] feature for a given anchor row).
    """

    def __init__(self, features_dir: Path, fps: float, max_cached: int = 4):
        self.features_dir = Path(features_dir)
        self.fps = fps
        self.max_cached = max_cached
        self._cache: dict = {}
        self._order: list = []

    def get_feature(self, video_id: str, anchor: int) -> np.ndarray:
        if video_id in self._cache:
            arr = self._cache[video_id]
        else:
            arr = self._open_video(video_id)
            self._cache[video_id] = arr
            self._order.append(video_id)
            if len(self._order) > self.max_cached:
                evict = self._order.pop(0)
                self._cache.pop(evict, None)
        return self._window_tokens(arr, anchor)

    def _open_video(self, video_id: str):
        raise NotImplementedError

    def _window_tokens(self, arr, anchor: int) -> np.ndarray:
        raise NotImplementedError


class DINOv3DenseLoader(_FeatureLoader):
    """
    DINOv3 patch token cache: one .npy per video, shape (T_source, num_patches, D).
    Slice W rows around the anchor (stride source_fps/target_fps when source > target)
    and flatten to a token grid of shape (W * num_patches, D).

    Because DINOv3 is a per-frame image model, a single 25 FPS dense file can
    be read at any lower effective rate by stride-indexing -- there is no need
    to re-extract for each target FPS. ``source_fps`` is the FPS embedded in
    the on-disk filename; ``fps`` (target) is the effective sampling rate the
    dataset operates at. When they're equal, behavior matches the legacy
    contiguous-slice path.
    """

    def __init__(self, features_dir, fps, window_size, source_fps=None,
                 model_id="dinov3_l", max_cached: int = 4, dense_tag: str = ""):
        super().__init__(features_dir, fps, max_cached)
        self.window_size = int(window_size)
        self.model_id = model_id
        self.source_fps = float(source_fps) if source_fps is not None else float(fps)
        # Optional extraction-protocol tag inserted before "_dense_features" in
        # the on-disk filename. Mirrors what extract_features.py wrote for
        # non-default preprocessing (e.g. "reflect", "yolo_reflect").
        self.dense_tag = dense_tag.strip("_")
        # Stride between successive output rows in source-fps row space.
        # Mirrors what `BaseFeatureExtractor._compute_stride` does at extraction.
        self._src_stride = max(1, int(round(self.source_fps / float(fps))))
        self._half_left = window_size // 2 - 1
        self._half_right = window_size // 2 + 1

    def _open_video(self, video_id: str):
        # File path uses source FPS (matches what extract_features.py wrote).
        tag = f"_{self.dense_tag}" if self.dense_tag else ""
        path = self.features_dir / (
            f"{video_id}_{self.model_id}_{self.source_fps}fps{tag}_dense_features.npy"
        )
        return np.load(path, mmap_mode="r")

    def _window_tokens(self, arr, anchor: int) -> np.ndarray:
        # Use the shared protocol with 'clamp' as a safety net. In the no-pad
        # protocol, callers (build_window_list / eval mAP) only request anchors
        # in the valid range, so the clamp never actually triggers -- but if a
        # bug ever produces a boundary anchor, edge-replication is safer than
        # an out-of-range index error.
        from window_protocol import select_source_frames

        src_rows = select_source_frames(
            anchor_idx=anchor,
            video_length=arr.shape[0],
            anchor_stride=self._src_stride,
            intra_window_stride=self._src_stride,
            window_length=self.window_size,
            boundary="clamp",
        )
        # Fancy-index returns a contiguous copy. For stride==1 this is
        # equivalent to (and slightly slower than) arr[start:end] -- the
        # difference is sub-millisecond at our per-window dims.
        window = np.asarray(arr[src_rows], dtype=np.float32)
        return window.reshape(-1, window.shape[-1])

    def get_frame_tokens(self, video_id: str, frame_5fps: int) -> np.ndarray:
        """Return patch tokens for a single 5-FPS frame as ``[num_patches, D]``.

        Used by the kassab_concat cross-clip dataset, which assembles W=10
        windows by gathering single-frame token sets from possibly different
        clips and stacking them into ``[W * num_patches, D]``. The cache reuse
        path is identical to ``get_feature``; the difference is the returned
        slice has no temporal stack dimension.
        """
        if video_id in self._cache:
            arr = self._cache[video_id]
        else:
            arr = self._open_video(video_id)
            self._cache[video_id] = arr
            self._order.append(video_id)
            if len(self._order) > self.max_cached:
                evict = self._order.pop(0)
                self._cache.pop(evict, None)
        src_row = int(frame_5fps) * self._src_stride
        src_row = max(0, min(src_row, arr.shape[0] - 1))  # clamp (defensive)
        return np.asarray(arr[src_row], dtype=np.float32)


class VJEPA2DenseLoader(_FeatureLoader):
    """
    V-JEPA2 dense window cache. Supports two on-disk layouts:

      (a) Streaming format (preferred, used for long clips):
          {video_id}_vjepa2_l_{fps}fps[_tag]_dense_w{W}*.npy
          + sidecar {video_id}_..._dense_w{W}*.meta.npz
          The .npy is mmap'd; the sidecar carries valid_lo, valid_hi,
          anchor_stride, intra_window_stride, window_length, n_source_frames.

      (b) Legacy single-archive format (TACDEC backfill, short clips):
          {video_id}_vjepa2_l_{fps}fps[_tag]_dense_w{W}*.npz
          (dense + metadata bundled in one compressed archive)

    The anchor row IS one V-JEPA2 forward in both layouts.
    """

    def __init__(self, features_dir, fps, window_size, model_id="vjepa2_l",
                 max_cached: int = 4, dense_tag: str = ""):
        super().__init__(features_dir, fps, max_cached)
        self.window_size = int(window_size)
        self.model_id = model_id
        # Optional extraction-protocol tag inserted before "_dense_w{W}" in
        # the on-disk filename (e.g. "reflect" for reflective-padding runs).
        self.dense_tag = dense_tag.strip("_")

    def _candidate_paths(self, video_id: str):
        tag = f"_{self.dense_tag}" if self.dense_tag else ""
        stem_prefix = f"{video_id}_{self.model_id}_{self.fps}fps{tag}_dense_w{self.window_size}"
        # Prefer the streaming .npy layout; fall back to legacy .npz if no
        # .npy exists (TACDEC files were written before the format change).
        npy_matches = sorted(self.features_dir.glob(f"{stem_prefix}*.npy"))
        if npy_matches:
            return npy_matches
        return sorted(self.features_dir.glob(f"{stem_prefix}*.npz"))

    def _open_video(self, video_id: str):
        candidates = self._candidate_paths(video_id)
        if not candidates:
            tag = f"_{self.dense_tag}" if self.dense_tag else ""
            raise FileNotFoundError(
                f"No V-JEPA2 dense file for video_id={video_id} at fps={self.fps} "
                f"W={self.window_size} (dense_tag={self.dense_tag!r}); expected "
                f"{self.features_dir}/{video_id}_{self.model_id}_{self.fps}fps{tag}_dense_w{self.window_size}*.{{npy,npz}}"
            )
        # Prefer the most-tagged file (has the most underscores) so a new
        # protocol output supersedes a legacy one if both exist.
        path = max(candidates, key=lambda p: p.name.count("_"))

        if path.suffix == ".npy":
            # Streaming layout: dense bytes via mmap, metadata from sidecar.
            dense = np.load(path, mmap_mode="r")
            meta_path = path.parent / (path.stem + ".meta.npz")
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Streaming dense file {path.name} has no metadata sidecar "
                    f"{meta_path.name}; cannot determine valid_lo/valid_hi."
                )
            with np.load(meta_path) as meta:
                valid_lo = int(meta["valid_lo"])
                valid_hi = int(meta["valid_hi"])
        else:
            # Legacy bundled .npz.
            with np.load(path) as npz:
                dense = np.asarray(npz["dense"])
                # Self-describing files (post no-pad protocol) carry valid_lo so
                # the loader can map anchor -> row. Legacy files default to 0
                # (= anchor index == row index, the old behaviour).
                valid_lo = int(npz["valid_lo"]) if "valid_lo" in npz.files else 0
                valid_hi = (int(npz["valid_hi"]) if "valid_hi" in npz.files
                            else valid_lo + dense.shape[0] - 1)

        return {"dense": dense, "valid_lo": valid_lo, "valid_hi": valid_hi}

    def _window_tokens(self, arr, anchor: int) -> np.ndarray:
        # arr is the dict returned by _open_video. anchor is the global
        # target-FPS anchor index; row = anchor - valid_lo.
        dense = arr["dense"]
        valid_lo = arr["valid_lo"]
        valid_hi = arr.get("valid_hi", valid_lo + dense.shape[0] - 1)
        row = anchor - valid_lo
        n_valid = valid_hi - valid_lo + 1
        if row < 0 or row >= n_valid:
            raise IndexError(
                f"Anchor {anchor} maps to row {row}, outside [0, {n_valid}). "
                f"valid range=[{valid_lo}, {valid_hi}]; caller should restrict "
                "anchors to valid_anchor_range(n_target=video_length, ...)."
            )
        return np.asarray(dense[row], dtype=np.float32)


# ---- PyTorch Dataset --------------------------------------------------------


class KassabAttentiveDataset(Dataset):
    """
    PyTorch Dataset wrapping a window list + a feature loader.

    Each item:
        features:  Tensor [N_tokens, D]  (DINOv3: W*256;   V-JEPA2: ~1024)
        label:     int                   (center-frame class label)
        video_id:  str                   (audit)
        anchor:    int                   (audit)
    """

    def __init__(self, window_list, feature_loader: _FeatureLoader):
        self.windows = window_list
        self.loader = feature_loader

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        feats = self.loader.get_feature(w["video_id"], w["anchor"])
        return {
            "features":  torch.from_numpy(feats),
            "label":     int(w["class"]),
            "video_id":  w["video_id"],
            "anchor":    int(w["anchor"]),
        }


def attentive_collate(batch):
    return {
        "features":  torch.stack([b["features"] for b in batch], dim=0),
        "labels":    torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "video_ids": [b["video_id"] for b in batch],
        "anchors":   torch.tensor([b["anchor"] for b in batch], dtype=torch.long),
    }


# ---- One-call entrypoint ----------------------------------------------------


def get_kassab_attentive_dataloaders(
    labels_dir,
    features_dir,
    backbone: str,        # 'dinov3' or 'vjepa2'
    window_size: int,
    target_fps: float = 4.0,
    bg_count: int = 500,
    replay_cap: int = 280,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    feature_loader_cache: int = 4,
    source_fps: float | None = None,
    dense_tag: str = "",
):
    """
    Build train/val/test dataloaders for the Kassab patch-token attentive probe
    pipeline. Returns ``(train_loader, val_loader, test_loader, info)``.

    ``info`` carries:
        - frame_counts_per_split : per-class window counts per split
        - n_sequences            : per-split window-list length
        - game_ids               : literal game-IDs per split (audit)
        - target_fps, window_size, backbone, _splits (raw windows)
    """
    if backbone not in ("dinov3", "vjepa2"):
        raise ValueError(f"backbone must be 'dinov3' or 'vjepa2', got {backbone!r}")

    sequences = build_kassab_attentive_sequences(
        labels_dir=labels_dir,
        target_fps=target_fps,
        bg_count=bg_count,
        replay_cap=replay_cap,
        seed=seed,
    )
    splits, game_ids = split_by_game(sequences,
                                     train=train_frac, val=val_frac, seed=seed)

    # Build window lists per split.
    split_windows = {
        name: build_window_list(seqs, window_size=window_size)
        for name, seqs in splits.items()
    }

    # One loader instance per split (independent caches; per-worker friendly).
    # source_fps lets DINOv3 read a 25 FPS file at a lower effective target_fps
    # via stride-indexing -- no re-extraction needed. Defaults to target_fps
    # when not provided, preserving the legacy "extract per FPS" behavior.
    eff_source_fps = source_fps if source_fps is not None else target_fps

    def _make_loader():
        if backbone == "dinov3":
            return DINOv3DenseLoader(features_dir, target_fps, window_size,
                                      source_fps=eff_source_fps,
                                      max_cached=feature_loader_cache,
                                      dense_tag=dense_tag)
        # V-JEPA2 features are produced per-window by a separate forward at
        # extraction time, so the on-disk file IS at target_fps -- source_fps
        # has no meaning here and is ignored.
        return VJEPA2DenseLoader(features_dir, target_fps, window_size,
                                  max_cached=feature_loader_cache,
                                  dense_tag=dense_tag)

    train_ds = KassabAttentiveDataset(split_windows["train"], _make_loader())
    val_ds   = KassabAttentiveDataset(split_windows["val"],   _make_loader())
    test_ds  = KassabAttentiveDataset(split_windows["test"],  _make_loader())

    # persistent_workers=True keeps each worker's feature LRU warm across
    # epochs (massive win for small datasets where the LRU + workers can hold
    # most of the videos in RAM after epoch 1). PyTorch refuses the flag with
    # num_workers=0, so gate on that.
    dl_kwargs = dict(collate_fn=attentive_collate, num_workers=num_workers)
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **dl_kwargs)

    # Per-split per-class counts (for class weights and audit).
    frame_counts_per_split = {}
    for name, wins in split_windows.items():
        per_class = {c: 0 for c in CLASS_NAMES}
        for w in wins:
            per_class[w["class"]] += 1
        frame_counts_per_split[name] = per_class

    info = {
        "frame_counts_per_split": frame_counts_per_split,
        "n_sequences": {k: len(v) for k, v in split_windows.items()},
        "game_ids": game_ids,
        "target_fps": target_fps,
        "source_fps": eff_source_fps,
        "window_size": window_size,
        "backbone": backbone,
        # Underscore-prefixed: not JSON-serializable
        "_splits": split_windows,
    }
    return train_loader, val_loader, test_loader, info


def compute_class_weights(window_list, num_classes=3, normalization="min1"):
    """
    Inverse-frequency class weights over a window list. Train split only --
    no leakage from val/test.

    ``normalization`` controls the final scaling of the inverse-frequency
    vector ``w_c = N / (K * count_c)``:

    * ``"min1"`` (default) - divide by ``min(w)`` so the smallest class
      weight becomes 1.0. Backwards-compatible with the original probe runs.
    * ``"balanced"`` - leave ``w`` unscaled. Matches sklearn's
      ``compute_class_weight('balanced', ...)`` semantics exactly; required for
      apples-to-apples comparison against Kassab's TempTAC training (which
      uses sklearn's 'balanced' directly).
    """
    if normalization not in ("min1", "balanced"):
        raise ValueError(
            f"normalization must be 'min1' or 'balanced', got {normalization!r}"
        )
    counts = np.zeros(num_classes, dtype=np.float64)
    for w in window_list:
        c = int(w["class"])
        if 0 <= c < num_classes:
            counts[c] += 1
    safe = counts.copy()
    safe[safe == 0] = 1.0
    weights = counts.sum() / (num_classes * safe)
    if normalization == "min1":
        weights = weights / weights.min()
    return weights.tolist(), counts.astype(int).tolist()
