"""
Feature loaders + window dataset for the attentive-probe pipelines.

Everything the DINOv3 / V-JEPA 2 attentive probes need to turn cached dense
features into per-window training items:

  - ``_load_video_labels_at_target_fps`` : read a clip's JSON labels onto the
    target-FPS feature-row grid (5 -> 3 class merge).
  - ``DINOv3DenseLoader`` / ``VJEPA2DenseLoader`` : lazy per-video feature
    loaders with a small LRU. Given a clip + anchor row they return one window
    of tokens -- per-frame patch tokens for DINOv3 (gather W consecutive rows
    -> [W*256, 1024]) or one cached dense entry for V-JEPA 2 ([N_tokens, 1024]).
  - ``AttentiveWindowDataset`` + ``attentive_collate`` : wrap a window list
    (built by ``temporal_protocol`` / ``balanced_temporal_dataset``) and a
    loader into a PyTorch dataset / dataloader.
  - ``compute_class_weights`` : inverse-frequency class weights over a train
    window list.

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
from torch.utils.data import Dataset

from data.labels import (
    BACKGROUND as CLASS_BACKGROUND,
    CLASS_NAMES,
    TACKLE_LIVE as CLASS_TACKLE_LIVE,
    TACKLE_REPLAY as CLASS_TACKLE_REPLAY,
)

# Public surface of this module. CLASS_NAMES is re-exported from data.labels as
# a convenience so callers get the loaders and the class-name map in one import.
__all__ = [
    "CLASS_NAMES",
    "DINOv3DenseLoader",
    "VJEPA2DenseLoader",
    "AttentiveWindowDataset",
    "attentive_collate",
    "compute_class_weights",
    "_load_video_labels_at_target_fps",
]


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
        # non-default preprocessing (e.g. "reflect").
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
        # protocol, callers only request anchors in the valid range, so the
        # clamp never actually triggers -- but if a
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


class AttentiveWindowDataset(Dataset):
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
