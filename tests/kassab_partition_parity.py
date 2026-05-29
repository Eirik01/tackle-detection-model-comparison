#!/usr/bin/env python3
"""
Kassab partition-parity smoke test.

Verifies that the data partition produced by run_kassab_spatial_comparison.py
is byte-identical to Kassab's spatial-approach.ipynb partition. Three checks,
each runs independently and skips if its prerequisites are missing:

  [1] Filename parity:
      Local TACDEC labels (data/tacdec/labels/*.json) match the file list
      inside tacdec-kassab-implementation/labels.zip.
  [2] Frame-count parity:
      First N videos in alphabetical order have the same OpenCV frame count
      as what run_kassab_spatial_comparison.py --inspect-only printed.
  [3] Shuffle determinism:
      kassab_split(...) on a fixed 425-video stub yields the documented
      train/val/test video index heads. Catches accidental changes to the
      split function (e.g. swapping legacy seed for default_rng).

Run:
    uv run python tests/kassab_partition_parity.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_kassab_spatial_comparison import kassab_split  # noqa: E402

LOCAL_LABELS_DIR = ROOT / "data" / "tacdec" / "labels"
LOCAL_VIDEOS_DIR = ROOT / "data" / "tacdec" / "videos"
KASSAB_LABELS_ZIP = ROOT / "tacdec-kassab-implementation" / "labels.zip"

# Reference values from --inspect-only run on 2026-05-05.
EXPECTED_FRAME_COUNTS_FIRST_10 = [600, 500, 650, 350, 350, 450, 550, 650, 650, 650]
EXPECTED_TRAIN_HEAD = [417, 75, 176, 30, 357, 347, 154, 153, 414, 157]
EXPECTED_VAL_HEAD = [254, 353, 4, 256, 381, 100, 226, 364, 213, 171]
EXPECTED_TEST_HEAD = [410, 49, 80, 205, 34, 263, 91, 339, 52, 345]
EXPECTED_NUM_VIDEOS = 425


class CheckResult:
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


def _print_result(name: str, status: str, detail: str = "") -> None:
    icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "·"}[status]
    print(f"  [{icon}] {status:4}  {name}" + (f"   ({detail})" if detail else ""))


def check_filename_parity() -> str:
    """[1] Same set of 425 video IDs in Kassab's labels.zip and our local labels dir."""
    name = "filename parity vs Kassab labels.zip"
    if not KASSAB_LABELS_ZIP.exists():
        _print_result(name, CheckResult.SKIP, f"missing {KASSAB_LABELS_ZIP.relative_to(ROOT)}")
        return CheckResult.SKIP
    if not LOCAL_LABELS_DIR.exists():
        _print_result(name, CheckResult.SKIP, f"missing {LOCAL_LABELS_DIR.relative_to(ROOT)}")
        return CheckResult.SKIP

    with zipfile.ZipFile(KASSAB_LABELS_ZIP) as zf:
        kassab_ids = sorted(
            Path(n).stem for n in zf.namelist()
            if n.startswith("labels/") and n.endswith(".json")
        )
    our_ids = sorted(p.stem for p in LOCAL_LABELS_DIR.glob("*.json"))

    if kassab_ids == our_ids and len(our_ids) == EXPECTED_NUM_VIDEOS:
        _print_result(name, CheckResult.PASS, f"{len(our_ids)} videos, identical IDs")
        return CheckResult.PASS

    only_kassab = sorted(set(kassab_ids) - set(our_ids))
    only_ours = sorted(set(our_ids) - set(kassab_ids))
    detail = f"kassab={len(kassab_ids)}, ours={len(our_ids)}"
    if only_kassab:
        detail += f", only-kassab[:3]={only_kassab[:3]}"
    if only_ours:
        detail += f", only-ours[:3]={only_ours[:3]}"
    _print_result(name, CheckResult.FAIL, detail)
    return CheckResult.FAIL


def check_frame_count_parity() -> str:
    """[2] First 10 alphabetically-ordered videos have the documented OpenCV frame counts."""
    name = "frame-count parity (first 10 videos)"
    if not LOCAL_VIDEOS_DIR.exists():
        _print_result(name, CheckResult.SKIP, f"missing {LOCAL_VIDEOS_DIR.relative_to(ROOT)}")
        return CheckResult.SKIP
    try:
        import cv2
    except ImportError:
        _print_result(name, CheckResult.SKIP, "cv2 not installed")
        return CheckResult.SKIP

    videos = sorted(LOCAL_VIDEOS_DIR.glob("*.mp4"))[:10]
    if len(videos) < 10:
        _print_result(name, CheckResult.SKIP, f"only {len(videos)} videos found, need 10")
        return CheckResult.SKIP

    counts = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()

    if counts == EXPECTED_FRAME_COUNTS_FIRST_10:
        _print_result(name, CheckResult.PASS, f"{counts}")
        return CheckResult.PASS
    _print_result(name, CheckResult.FAIL,
                  f"got {counts}, expected {EXPECTED_FRAME_COUNTS_FIRST_10}")
    return CheckResult.FAIL


def check_shuffle_determinism() -> str:
    """[3] kassab_split() produces the documented train/val/test video-index heads."""
    name = "shuffle determinism (kassab_split, seed=42)"
    # Use a stub frame_counts of length 425; the per-frame fan-out doesn't matter
    # for shuffle indices, so any positive int works.
    stub_counts = np.full(EXPECTED_NUM_VIDEOS, 1, dtype=np.int64)
    _, _, _, (train_v, val_v, test_v) = kassab_split(stub_counts, seed=42)

    train_head = train_v[:10].tolist()
    val_head = val_v[:10].tolist()
    test_head = test_v[:10].tolist()

    failures = []
    if train_head != EXPECTED_TRAIN_HEAD:
        failures.append(f"train_head={train_head}")
    if val_head != EXPECTED_VAL_HEAD:
        failures.append(f"val_head={val_head}")
    if test_head != EXPECTED_TEST_HEAD:
        failures.append(f"test_head={test_head}")

    if not failures:
        _print_result(name, CheckResult.PASS,
                      f"train={len(train_v)} val={len(val_v)} test={len(test_v)}")
        return CheckResult.PASS
    _print_result(name, CheckResult.FAIL, "; ".join(failures))
    return CheckResult.FAIL


def main() -> int:
    print("=" * 70)
    print("Kassab partition-parity smoke test")
    print("=" * 70)

    results = [
        check_filename_parity(),
        check_frame_count_parity(),
        check_shuffle_determinism(),
    ]

    print("=" * 70)
    n_pass = sum(r == CheckResult.PASS for r in results)
    n_fail = sum(r == CheckResult.FAIL for r in results)
    n_skip = sum(r == CheckResult.SKIP for r in results)
    print(f"Results: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
