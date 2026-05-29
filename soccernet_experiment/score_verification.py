"""Score firings against ground-truth tackles with explicit time tolerance.

Standard event-detection greedy matching, mirroring eval_temporal.py / the
SoccerNet-v2 action-spotting protocol:

  - Sort firings by confidence, descending.
  - For each firing, find the nearest unmatched truth within +/- TOLERANCE
    seconds. If found -> TP, mark truth used. Otherwise -> FP.
  - Truths never matched count as FN for that pipeline.

We compute matching independently per pipeline (each pipeline competes only
against itself for the truths). Per pipeline we report:

  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)
  F1        = 2 * P * R / (P + R)

split into two operating definitions:
  - tackle-vs-not-tackle:  collapsed binary task. A firing counts as TP if
                           there is ANY real tackle within the tolerance,
                           regardless of class. Answers "when the model
                           predicts tackle, is it actually a tackle?".
  - class-aware:           tighter. Predicted class (tackle-live vs
                           tackle-replay) must match the truth class.

Default tolerance: 2.0 s (matches the W=10 @ 5 FPS = 2 s training window).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

VALID_CLASSES = {"tackle-live", "tackle-replay"}


def parse_timestamp(s: str) -> float:
    """Accept seconds (e.g. '642.4') or mm:ss[.ss] (e.g. '10:42.40').
    Also tolerates mm:ss:00 where the user typed a stray ':' instead of '.'."""
    s = s.strip()
    if ":" not in s:
        return float(s)
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    if len(parts) == 3:
        # mm:ss:00 (typo: stray ':' instead of '.')  OR  hh:mm:ss (unlikely here)
        m, sec, frac = parts
        if int(m) < 60 and frac == "00":
            return int(m) * 60 + int(sec)
        # genuine hh:mm:ss fallback
        return int(m) * 3600 + int(sec) * 60 + float(frac)
    raise ValueError(f"unparseable timestamp: {s!r}")


def load_truths(path: Path):
    truths = []
    with open(path) as f:
        for r in csv.DictReader(f):
            # Accept either the new `timestamp` header or the legacy
            # `timestamp_sec` header.
            ts = (r.get("timestamp") or r.get("timestamp_sec") or "").strip()
            cls = r.get("class_name", "").strip()
            if not ts and not cls:
                continue  # blank row
            if not ts or not cls:
                raise SystemExit(
                    f"truth_tackles.csv: incomplete row {r}. "
                    "timestamp and class_name are required.")
            if cls not in VALID_CLASSES:
                raise SystemExit(
                    f"truth_tackles.csv: unknown class_name '{cls}'. "
                    f"Must be one of {sorted(VALID_CLASSES)}.")
            try:
                ts_sec = parse_timestamp(ts)
            except ValueError as e:
                raise SystemExit(f"truth_tackles.csv: {e}")
            truths.append({"timestamp_sec": ts_sec, "class_name": cls})
    return truths


def load_firings(path: Path):
    firings = []
    with open(path) as f:
        for r in csv.DictReader(f):
            firings.append({
                "pipeline": r["pipeline"],
                "timestamp_sec": float(r["timestamp_sec"]),
                "predicted_class": r["predicted_class"],
                "confidence": float(r["confidence"]),
            })
    return firings


def greedy_match(firings: list[dict], truths: list[dict],
                 tol: float, class_aware: bool):
    """Return (tp, fp, n_truth_matched). Mutates a local copy of truth flags."""
    used = [False] * len(truths)
    fs = sorted(firings, key=lambda f: -f["confidence"])
    tp = fp = 0
    for f in fs:
        best_i, best_dt = -1, tol + 1e-9
        for i, t in enumerate(truths):
            if used[i]:
                continue
            if class_aware and t["class_name"] != f["predicted_class"]:
                continue
            dt = abs(t["timestamp_sec"] - f["timestamp_sec"])
            if dt <= tol and dt < best_dt:
                best_i, best_dt = i, dt
        if best_i >= 0:
            tp += 1
            used[best_i] = True
        else:
            fp += 1
    matched = sum(used)
    return tp, fp, matched


def fmt_prf(tp: int, fp: int, fn: int) -> str:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return f"P={p*100:5.1f}% R={r*100:5.1f}% F1={f1*100:5.1f}% (TP={tp} FP={fp} FN={fn})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dir", type=Path,
                    default=Path(__file__).parent / "verification")
    ap.add_argument("--tolerance-sec", type=float, default=2.0,
                    help="prediction-to-truth time tolerance (default 2.0 s, "
                         "matching the W=10 @ 5 FPS training window).")
    args = ap.parse_args()

    truths = load_truths(args.verify_dir / "truth_tackles.csv")
    firings = load_firings(args.verify_dir / "firings.csv")

    n_live = sum(1 for t in truths if t["class_name"] == "tackle-live")
    n_repl = sum(1 for t in truths if t["class_name"] == "tackle-replay")
    print(f"Truth tackles in window : {len(truths)}  "
          f"(live={n_live}, replay={n_repl})")
    print(f"Tolerance              : +/- {args.tolerance_sec} s")
    print()

    by_pipe = defaultdict(list)
    for fr in firings:
        by_pipe[fr["pipeline"]].append(fr)

    print(f"== tackle-vs-not-tackle  (class ignored: 'when the model says tackle, is it a tackle?') ==")
    for pipe in sorted(by_pipe):
        fs = by_pipe[pipe]
        tp, fp, matched = greedy_match(fs, truths, args.tolerance_sec,
                                       class_aware=False)
        fn = len(truths) - matched
        print(f"  {pipe:<12}  {fmt_prf(tp, fp, fn)}")

    print()
    print(f"== class-aware  (predicted live/replay must match truth class) ==")
    for pipe in sorted(by_pipe):
        fs = by_pipe[pipe]
        tp, fp, matched = greedy_match(fs, truths, args.tolerance_sec,
                                       class_aware=True)
        fn = len(truths) - matched
        print(f"  {pipe:<12}  {fmt_prf(tp, fp, fn)}")

    if not truths:
        print()
        print("WARNING: truth_tackles.csv is empty. Add real tackle rows "
              "before the recall numbers mean anything.")


if __name__ == "__main__":
    main()
