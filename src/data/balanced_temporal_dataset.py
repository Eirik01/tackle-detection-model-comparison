"""Balanced-windows attentive-probe dataset (5 FPS / W=10).

Companion to ``temporal_loaders.py``. The Kassab dataset uses a
concat-and-slide rule against a no-pad stream; this one instead uses
``temporal_protocol.build_balanced_windows`` as the upstream sampler:
one window per event, class-balanced background, and a strict
"center-frame label == event class" filter.

Reuses the file loaders, PyTorch dataset wrapper, and collator from
``temporal_loaders.py`` verbatim. The temporal_protocol's
``c`` (5-FPS center index) is exactly ``anchor_idx`` in
``window_protocol``: both V-JEPA 2 (``dense[c - valid_lo]``) and DINOv3
(``select_source_frames(anchor_idx=c, ...)``) resolve to the same 10
underlying 25-FPS source frames.

Splits come from ``splits.split_games`` (re-exported by
``temporal_protocol``) so the train/val/test partition matches the
spatial probe game-disjoint split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from data.temporal_loaders import (
    DINOv3DenseLoader,
    AttentiveWindowDataset,
    VJEPA2DenseLoader,
    attentive_collate,
)
from data.labels import CLASS_NAMES
from data.temporal_protocol import (
    STRIDE_S,
    W as PROTO_W,
    build_balanced_windows,
    split_games,
    split_games_from_file,
)

__all__ = [
    "get_balanced_temporal_dataloaders",
]


def _scan_clip_metadata(
    clip_ids: List[str], label_dir: Path
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Read each clip's JSON once. Returns ``(game_id_by_clip, n_target_by_clip)``.

    ``n_target`` is the number of 5-FPS rows (= ``frame_count_25fps // STRIDE_S``),
    matching the convention used by ``temporal_protocol.labels_at_5fps``.
    """
    game_id_by_clip: Dict[str, int] = {}
    n_target_by_clip: Dict[str, int] = {}
    for clip_id in clip_ids:
        with open(label_dir / f"{clip_id}.json") as f:
            data = json.load(f)
        game_id_by_clip[clip_id] = int(data["metadata"]["game_id"])
        n_target_by_clip[clip_id] = int(data["media_attributes"]["frame_count"]) // STRIDE_S
    return game_id_by_clip, n_target_by_clip


def _windows_to_dicts(
    tuples: List[Tuple[str, int, int]],
    game_id_by_clip: Dict[str, int],
    n_target_by_clip: Dict[str, int],
) -> List[dict]:
    """Convert ``build_balanced_windows`` tuples to the dict format expected by
    ``AttentiveWindowDataset`` + ``DINOv3DenseLoader`` / ``VJEPA2DenseLoader``."""
    return [
        {
            "video_id": clip_id,
            "game_id":  game_id_by_clip[clip_id],
            "class":    int(cls),
            "anchor":   int(center),
            "n_target": n_target_by_clip[clip_id],
            "stride":   STRIDE_S,
        }
        for clip_id, center, cls in tuples
    ]


def _assemble_loaders(
    *,
    split_tuples: Dict[str, List[Tuple[str, int, int]]],
    splits: Dict[str, List[str]],
    label_dir: Path,
    features_dir,
    backbone: str,
    window_size: int,
    target_fps: float,
    eff_source_fps: float,
    batch_size: int,
    num_workers: int,
    feature_loader_cache: int,
    dense_tag: str,
    protocol_name: str,
    seed: int = 42,
):
    """Shared back end for the temporal dataloaders.

    Takes per-split ``(clip_id, center, class)`` tuples and builds the loaders +
    info dict. ``splits`` is the clip-id partition, used only for the game-ID
    audit in ``info``.
    """
    all_clips = sorted({c for v in splits.values() for c in v})
    game_id_by_clip, n_target_by_clip = _scan_clip_metadata(all_clips, label_dir)

    split_windows: Dict[str, List[dict]] = {
        name: _windows_to_dicts(tuples, game_id_by_clip, n_target_by_clip)
        for name, tuples in split_tuples.items()
    }

    def _make_loader():
        if backbone == "dinov3":
            return DINOv3DenseLoader(
                features_dir, target_fps, window_size,
                source_fps=eff_source_fps,
                max_cached=feature_loader_cache,
                dense_tag=dense_tag,
            )
        return VJEPA2DenseLoader(
            features_dir, target_fps, window_size,
            max_cached=feature_loader_cache,
            dense_tag=dense_tag,
        )

    train_ds = AttentiveWindowDataset(split_windows["train"], _make_loader())
    val_ds   = AttentiveWindowDataset(split_windows["val"],   _make_loader())
    test_ds  = AttentiveWindowDataset(split_windows["test"],  _make_loader())

    # persistent_workers=True keeps each worker's feature LRU warm across
    # epochs. PyTorch refuses the flag with num_workers=0, so gate on that.
    dl_kwargs = dict(collate_fn=attentive_collate, num_workers=num_workers)
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True

    # Explicit seeded generator so the train shuffle order is reproducible
    # independently of how many global-RNG draws happened before this point
    # (e.g. probe weight init), rather than relying on global RNG ordering.
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=train_gen, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **dl_kwargs)

    frame_counts_per_split = {}
    for name, wins in split_windows.items():
        per_class = {c: 0 for c in CLASS_NAMES}
        for w in wins:
            per_class[w["class"]] += 1
        frame_counts_per_split[name] = per_class

    game_ids = {
        name: sorted({game_id_by_clip[c] for c in clip_ids})
        for name, clip_ids in splits.items()
    }

    info = {
        "frame_counts_per_split": frame_counts_per_split,
        "n_sequences": {k: len(v) for k, v in split_windows.items()},
        "game_ids":    game_ids,
        "target_fps":  target_fps,
        "source_fps":  eff_source_fps,
        "window_size": window_size,
        "backbone":    backbone,
        "protocol":    protocol_name,
        "_splits":     split_windows,
    }
    return train_loader, val_loader, test_loader, info


def _validate_and_split(backbone, window_size, labels_dir, train_frac, val_frac,
                         seed, split_file=None):
    if backbone not in ("dinov3", "vjepa2"):
        raise ValueError(f"backbone must be 'dinov3' or 'vjepa2', got {backbone!r}")
    if window_size != PROTO_W:
        raise ValueError(
            f"temporal_protocol is wired for W={PROTO_W}; got window_size={window_size}. "
            "Update temporal_protocol's W if you need a different window length."
        )
    label_dir = Path(labels_dir)
    if split_file is not None:
        splits = split_games_from_file(split_file)
    else:
        test_frac = max(0.0, 1.0 - train_frac - val_frac)
        splits = split_games(label_dir, val_frac=val_frac, test_frac=test_frac, seed=seed)
    return label_dir, splits


def get_balanced_temporal_dataloaders(
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
    split_file: str | Path | None = None,
):
    """Build train/val/test loaders for the balanced-windows attentive probe.

    One W=10 window per event, class-balanced background (see
    ``temporal_protocol.build_balanced_windows``). Each item produced by the
    returned loaders matches the attentive window dataset's contract:
        features  : Tensor [N_tokens, D]
                      DINOv3 -> ``W * num_patches`` rows
                      V-JEPA 2 -> single dense window slice
        labels    : LongTensor [B]  (center-frame class label)
        video_ids : list[str]
        anchors   : LongTensor [B]  (5-FPS center index)

    Returns ``(train_loader, val_loader, test_loader, info)``.
    """
    label_dir, splits = _validate_and_split(
        backbone, window_size, labels_dir, train_frac, val_frac, seed,
        split_file=split_file,
    )
    split_tuples = {
        name: build_balanced_windows(clip_ids, label_dir, seed=seed)
        for name, clip_ids in splits.items()
    }
    eff_source_fps = source_fps if source_fps is not None else target_fps
    return _assemble_loaders(
        split_tuples=split_tuples,
        splits=splits,
        label_dir=label_dir,
        features_dir=features_dir,
        backbone=backbone,
        window_size=window_size,
        target_fps=target_fps,
        eff_source_fps=eff_source_fps,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_loader_cache=feature_loader_cache,
        dense_tag=dense_tag,
        protocol_name="balanced",
        seed=seed,
    )

