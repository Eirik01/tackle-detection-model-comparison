"""Cross-clip concat-and-slide dataset for strict Kassab TempTAC parity.

Companion to ``balanced_temporal_dataset.py``. The other temporal datasets
build one window per (clip, center) pair, with the window's W underlying
frames always coming from a single clip. This dataset instead consumes the
*global* window list produced by ``temporal_protocol.build_kassab_concat_windows``,
where each window's W underlying frames may span two or more clips (matching
Kassab TempTAC's ``torch.cat``-then-slide pipeline at 5 FPS).

DINOv3 only -- V-JEPA 2's per-window dense features bake in single-clip
temporal context, so cross-clip windows can't be assembled from the existing
extraction. The wiring in ``train_temporal.py`` / ``eval_temporal.py``
raises if a V-JEPA 2 backbone is paired with ``--protocol kassab_concat``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.kassab_attentive_dataset import (
    CLASS_NAMES,
    DINOv3DenseLoader,
    attentive_collate,
)
from data.temporal_protocol import (
    STRIDE_S,
    W as PROTO_W,
    _manifest_from_sub_seqs,
    build_kassab_concat_stream,
    build_kassab_concat_windows,
    build_kassab_concat_windows_from_manifest,
    kassab_buggy_split,
    split_games,
    split_games_from_file,
)


__all__ = [
    "KassabConcatTemporalDataset",
    "get_kassab_concat_temporal_dataloaders",
]


class KassabConcatTemporalDataset(Dataset):
    """Wraps a list of cross-clip windows (from ``build_kassab_concat_windows``)
    and a ``DINOv3DenseLoader`` cache. Each item assembles the W per-frame
    patch token sets across (possibly different) clips into the same
    ``[W * num_patches, D]`` layout the existing single-clip loader produces,
    so the downstream collate / probe code paths are unchanged.

    The ``video_ids`` / ``anchors`` fields returned by the collate function
    are populated with the **center frame's** clip and 5-FPS index (for
    misclassification dumps and event metrics). The other W-1 underlying
    frames are not surfaced -- they're consumed only to build the feature
    tensor.
    """

    def __init__(self, windows: List[Dict], loader: DINOv3DenseLoader):
        self.windows = windows
        self.loader = loader

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict:
        w = self.windows[idx]
        frames = w["frames"]               # list of (clip_id, local_5fps_idx)
        # Per-frame patch tokens, shape [num_patches, D]. Stack into
        # [W, num_patches, D] then flatten to [W*num_patches, D] to match the
        # single-clip layout produced by DINOv3DenseLoader._window_tokens.
        per_frame = [self.loader.get_frame_tokens(c, i) for c, i in frames]
        tokens = np.stack(per_frame, axis=0)            # [W, P, D]
        tokens = tokens.reshape(-1, tokens.shape[-1])   # [W*P, D]
        center_clip, center_idx = w["center_frame"]
        return {
            "features": torch.from_numpy(tokens),
            "label":    int(w["center_label"]),
            "video_id": center_clip,
            "anchor":   int(center_idx),
        }


def _scan_n_target(clip_ids: List[str], label_dir: Path) -> Dict[str, int]:
    """Number of 5-FPS rows per clip (frame_count // STRIDE_S)."""
    out: Dict[str, int] = {}
    for clip_id in clip_ids:
        with open(label_dir / f"{clip_id}.json") as f:
            data = json.load(f)
        out[clip_id] = int(data["media_attributes"]["frame_count"]) // STRIDE_S
    return out


def get_kassab_concat_temporal_dataloaders(
    labels_dir,
    features_dir,
    backbone: str,
    window_size: int = PROTO_W,
    target_fps: float = 5.0,
    source_fps: float | None = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    feature_loader_cache: int = 4,
    dense_tag: str = "",
    replay_cap: int = 280,
    bg_count: int = 500,
    split_file: str | Path | None = None,
    split_mode: str = "kassab_bug",
):
    """Strict Kassab TempTAC concat-and-slide dataloaders at 5 FPS.

    Pipeline (caps applied GLOBALLY, partitioned afterwards):
      1. Sort the union of all train/val/test clip IDs (matches Kassab's
         sorted-.mp4 iteration order).
      2. Run ``build_kassab_concat_stream`` once over the union with a
         single seeded RNG, so the 500 bg / 280 replay caps bind first-come
         across the entire dataset. Same semantics as cell 7 of
         TempTAC.ipynb (``extract_tackle_sequences_and_undersample`` runs
         before ``split_data_by_game``).
      3. Partition the resulting flattened stream into train/val/test
         according to ``split_mode``:
           * ``"kassab_bug"`` (default) -- faithfully replicate Kassab's
             ``extract_data`` flat-index slice (slice the undersampled
             concat tensor by ORIGINAL-frame-count boundaries; cell 10 of
             TempTAC.ipynb). The resulting pools are NOT game-disjoint:
             they are positional slices of the global undersampled stream,
             matching the bias of Kassab's evaluated test pool exactly.
             Use this for direct numerical parity with Kassab's reported
             classification reports.
           * ``"correct"`` -- partition by actual clip-to-split membership
             (real game-disjoint). The pools are honest train/val/test
             but differ from Kassab's reported numbers because his split
             logic was buggy. Use this for our own headline evaluation.
      4. Per split: slide stride-1 W windows over the per-split stream.
      5. Wrap in ``KassabConcatTemporalDataset`` over a shared
         ``DINOv3DenseLoader`` cache.

    Returns ``(train_loader, val_loader, test_loader, info)``. ``info`` keys:
        * ``frame_counts_per_split`` (windows per class per split, by center-frame label)
        * ``n_sequences``            (windows per split)
        * ``n_sub_sequences``        (kept sub-sequences per split)
        * ``n_cross_clip_windows``   (windows whose W frames span >= 2 clips)
        * ``game_ids``               (sorted unique game ids per split clip-list)
        * ``protocol`` = ``"kassab_concat"``
    """
    if backbone != "dinov3":
        raise NotImplementedError(
            "kassab_concat is DINOv3-only. V-JEPA 2 dense features bake in "
            "single-clip temporal context (one feature row per pre-extracted "
            "tubelet), so cross-clip windows can't be assembled from the "
            "existing extraction. Use --protocol centered for V-JEPA 2."
        )
    if window_size != PROTO_W:
        raise ValueError(
            f"kassab_concat is wired for W={PROTO_W}; got window_size={window_size}."
        )
    if split_mode not in ("kassab_bug", "correct"):
        raise ValueError(
            f"split_mode must be 'kassab_bug' or 'correct', got {split_mode!r}"
        )

    label_dir = Path(labels_dir)
    if split_file is not None:
        splits = split_games_from_file(split_file)
    else:
        test_frac = max(0.0, 1.0 - train_frac - val_frac)
        splits = split_games(label_dir, val_frac=val_frac, test_frac=test_frac, seed=seed)

    eff_source_fps = source_fps if source_fps is not None else target_fps

    def _make_loader():
        return DINOv3DenseLoader(
            features_dir, target_fps, window_size,
            source_fps=eff_source_fps,
            max_cached=feature_loader_cache,
            dense_tag=dense_tag,
        )

    # Apply caps GLOBALLY across all clips, then partition. Mirrors Kassab's
    # notebook order: cell 7 (extract_tackle_sequences_and_undersample on the
    # whole X/y) runs before cell 10 (split_data_by_game).
    #
    # Clip-iteration order = sorted union of all clip IDs, matching the order
    # in which Kassab's notebook processes frame_counts (sorted .mp4 names).
    all_clips_sorted = sorted({c for ids in splits.values() for c in ids})
    global_sub_seqs = build_kassab_concat_stream(
        all_clips_sorted, label_dir, seed=seed,
        replay_cap=replay_cap, bg_cap=bg_count,
    )

    split_windows: Dict[str, List[Dict]] = {}
    split_subseqs: Dict[str, List[Dict]] = {name: [] for name in splits}

    if split_mode == "correct":
        # Real game-disjoint partition. Sub-sequence ordering within a split
        # is preserved from the global pass; per-split windows are sliding
        # over the split's contiguous slice of the global stream.
        clip_to_split: Dict[str, str] = {}
        for split_name, clip_ids in splits.items():
            for c in clip_ids:
                clip_to_split[c] = split_name
        for sub in global_sub_seqs:
            split_subseqs[clip_to_split[sub["clip_id"]]].append(sub)
        split_windows = {
            name: build_kassab_concat_windows(subs, window_size=window_size)
            for name, subs in split_subseqs.items()
        }
    else:  # "kassab_bug"
        # Replicate Kassab's extract_data bug: slice the undersampled stream
        # by ORIGINAL-frame-count boundaries (at 5 FPS = frame_counts // STRIDE_S).
        # The resulting pools are NOT game-disjoint -- they're positional
        # slices of the global manifest. Required for evaluating on the same
        # test pool Kassab reports.
        global_manifest = _manifest_from_sub_seqs(global_sub_seqs)
        frame_counts_5fps_list = []
        for clip_id in all_clips_sorted:
            with open(label_dir / f"{clip_id}.json") as f:
                data = json.load(f)
            frame_counts_5fps_list.append(
                int(data["media_attributes"]["frame_count"]) // STRIDE_S
            )
        frame_counts_5fps = np.array(frame_counts_5fps_list, dtype=np.int64)
        per_split_manifest = kassab_buggy_split(
            global_manifest, frame_counts_5fps, seed=seed,
        )
        split_windows = {
            name: build_kassab_concat_windows_from_manifest(m, window_size=window_size)
            for name, m in per_split_manifest.items()
        }
        # No per-split sub-sequence list in this mode (the slice cuts mid-
        # sequence). Leave split_subseqs as empty lists for info compat.

    # Datasets (one loader per split so the LRU is scoped per worker / split).
    train_ds = KassabConcatTemporalDataset(split_windows["train"], _make_loader())
    val_ds   = KassabConcatTemporalDataset(split_windows["val"],   _make_loader())
    test_ds  = KassabConcatTemporalDataset(split_windows["test"],  _make_loader())

    dl_kwargs = dict(collate_fn=attentive_collate, num_workers=num_workers)
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True

    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=train_gen, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **dl_kwargs)

    # Bookkeeping
    frame_counts_per_split: Dict[str, Dict[int, int]] = {}
    n_cross_clip: Dict[str, int] = {}
    for name, wins in split_windows.items():
        per_class = {c: 0 for c in CLASS_NAMES}
        cross = 0
        for w in wins:
            per_class[int(w["center_label"])] += 1
            if w["crosses_clip"]:
                cross += 1
        frame_counts_per_split[name] = per_class
        n_cross_clip[name] = cross

    all_clips = sorted({c for v in splits.values() for c in v})
    game_id_by_clip: Dict[str, int] = {}
    for clip_id in all_clips:
        with open(label_dir / f"{clip_id}.json") as f:
            data = json.load(f)
        game_id_by_clip[clip_id] = int(data["metadata"]["game_id"])
    game_ids = {
        name: sorted({game_id_by_clip[c] for c in clip_ids})
        for name, clip_ids in splits.items()
    }

    # Wrap split lists into the same per-window dict layout used by
    # KassabAttentiveDataset book-keeping (lets compute_class_weights and
    # the train_temporal.py "Train per-class counts" print reuse without
    # changes).
    info_splits = {
        name: [{"class": int(w["center_label"])} for w in wins]
        for name, wins in split_windows.items()
    }

    # Global pool stats (pre-partition). Lets the caller verify the caps
    # actually bound across the whole dataset, not within each split.
    global_pool_counts = {c: 0 for c in CLASS_NAMES}
    for sub in global_sub_seqs:
        for lbl in sub["labels"]:
            global_pool_counts[int(lbl)] += 1
    n_global_seqs_by_class = {c: 0 for c in CLASS_NAMES}
    for sub in global_sub_seqs:
        n_global_seqs_by_class[int(sub["first_cls"])] += 1

    info = {
        "frame_counts_per_split":  frame_counts_per_split,
        "n_sequences":             {k: len(v) for k, v in split_windows.items()},
        "n_sub_sequences":         {k: len(v) for k, v in split_subseqs.items()},
        "n_cross_clip_windows":    n_cross_clip,
        "game_ids":                game_ids,
        "target_fps":              target_fps,
        "source_fps":              eff_source_fps,
        "window_size":             window_size,
        "backbone":                backbone,
        "protocol":                "kassab_concat",
        "split_mode":              split_mode,
        "replay_cap":              replay_cap,
        "bg_count":                bg_count,
        "global_pool_frames":      global_pool_counts,
        "global_pool_n_seqs":      n_global_seqs_by_class,
        "global_pool_n_clips":     len(all_clips_sorted),
        "_splits":                 info_splits,
    }
    return train_loader, val_loader, test_loader, info
