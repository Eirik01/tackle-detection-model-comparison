"""Dump Kassab TempTAC's train/val/test partition as a clip-ID JSON file.

Kassab's notebook (``tacdec-kassab-implementation/TempTAC.ipynb``,
``split_data_by_game``) splits 425 videos 70/15/15 with this exact recipe:

    np.random.seed(42)
    idx = np.arange(len(frame_counts))   # ordered by sorted .mp4 filename
    np.random.shuffle(idx)
    n_train = int(N * 0.70)              # 297
    n_val   = int(N * 0.15)              # 63
    train, val, test = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

``frame_counts.npy`` is indexed by ``sorted(os.listdir(path_to_videos))`` (see
``feature_extraction.ipynb``), so a Kassab index ``i`` corresponds to the
``i``-th .mp4 in lexicographic order. We replay that recipe over the sorted
TACDEC label-file stems (the JSON stems match the .mp4 stems 1:1) and write
out a ``{train, val, test}`` clip-ID JSON consumable by
``train_temporal.py --split-file`` / ``eval_temporal.py --split-file``.

The output is bit-identical to Kassab's notebook split. Pass it to both
training and eval to remove the split mismatch from the comparison.

Usage:
  uv run python dump_kassab_split.py \
      --labels-dir /path/to/TACDEC/labels \
      --output tackle-detection-model-comparison/data/kassab_split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def kassab_partition(clip_stems: list[str], seed: int = 42,
                     train_frac: float = 0.70, val_frac: float = 0.15
                     ) -> dict[str, list[str]]:
    """Replicate ``split_data_by_game`` from TempTAC.ipynb.

    Uses the legacy ``np.random.seed`` + ``np.random.shuffle`` RNG (not
    ``np.random.default_rng``), the same ``int()`` truncation for the
    fractional split sizes, and the same train/val/test slicing order. The
    permutation is therefore identical to Kassab's at seed 42.
    """
    n = len(clip_stems)
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return {
        "train": sorted(clip_stems[i] for i in train_idx),
        "val":   sorted(clip_stems[i] for i in val_idx),
        "test":  sorted(clip_stems[i] for i in test_idx),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels-dir", required=True,
                    help="Directory of TACDEC label JSONs. Each <stem>.json "
                         "corresponds to one <stem>.mp4 in Kassab's video dir, "
                         "so sorted JSON stems == Kassab's sorted .mp4 list.")
    ap.add_argument("--output", required=True,
                    help="Output JSON path (will be created). Schema: "
                         '{"train": [...], "val": [...], "test": [...]}')
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed (default 42, matching TempTAC.ipynb).")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac",   type=float, default=0.15)
    ap.add_argument("--kassab-frame-counts", default=None,
                    help="Optional path to Kassab's frame_counts.npy. If "
                         "given, the script asserts len(frame_counts) == "
                         "number of label files; otherwise the order would "
                         "not match Kassab's indexing.")
    args = ap.parse_args()

    label_dir = Path(args.labels_dir)
    if not label_dir.is_dir():
        raise SystemExit(f"labels-dir does not exist: {label_dir}")

    clip_stems = sorted(p.stem for p in label_dir.glob("*.json"))
    if not clip_stems:
        raise SystemExit(f"No .json label files under {label_dir}")

    if args.kassab_frame_counts is not None:
        fc = np.load(args.kassab_frame_counts)
        if len(fc) != len(clip_stems):
            raise SystemExit(
                f"frame_counts.npy has {len(fc)} entries but found "
                f"{len(clip_stems)} label JSONs in {label_dir}. Counts must "
                f"match for the Kassab index <-> clip-stem mapping to be "
                f"valid. Aborting."
            )

    splits = kassab_partition(
        clip_stems,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "_metadata": {
            "source": "Kassab TempTAC split_data_by_game (TempTAC.ipynb)",
            "seed": args.seed,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "n_total": len(clip_stems),
            "n_train": len(splits["train"]),
            "n_val":   len(splits["val"]),
            "n_test":  len(splits["test"]),
            "rng": "np.random.seed + np.random.shuffle (legacy)",
            "order_key": "lexicographic stem of label JSONs (== sorted .mp4 in Kassab's video dir)",
        },
        **splits,
    }
    out_path.write_text(json.dumps(metadata, indent=2))

    md = metadata["_metadata"]
    print(f"Wrote {out_path}")
    print(f"  n_total = {md['n_total']}  "
          f"train = {md['n_train']}  val = {md['n_val']}  test = {md['n_test']}")


if __name__ == "__main__":
    main()
