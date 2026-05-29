"""Build the inputs for a ground-truth-driven precision/recall audit
of the Obj4 broadcast experiment.

Outputs two CSVs in soccernet_experiment/verification/:

  firings.csv         every fired event from every pipeline in the chosen
                      window at the chosen threshold. Auto-generated, do
                      not edit.

  truth_tackles.csv   header-only template (created only if missing). You
                      fill it by watching the window once and writing one
                      row per real tackle:
                          timestamp, class_name
                      timestamp accepts either seconds (642.4) or
                      mm:ss[.ss] (10:42.40).
                      class_name must be 'tackle-live' or 'tackle-replay'.

After labelling, run score_verification.py.

Defaults match the chosen subset: 10:00-12:00, confidence threshold 0.7.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PIPELINES = {
    "dinov3_attn": "2_720p_events_all.csv",
    "dinov3_lin":  "2_720p_spatial_events_all.csv",
    "vjepa2_attn": "2_720p_vjepa2_events_all.csv",
}


def mmss(sec: float) -> str:
    m = int(sec // 60)
    s = sec - 60 * m
    return f"{m:02d}:{s:05.2f}"


def load_firings(pred_dir: Path, threshold: float, t0: float, t1: float):
    out = []
    for pipeline, fname in PIPELINES.items():
        with open(pred_dir / fname) as f:
            for r in csv.DictReader(f):
                ts = float(r["timestamp_sec"])
                conf = float(r["confidence"])
                if conf < threshold or not (t0 <= ts < t1):
                    continue
                out.append({
                    "pipeline": pipeline,
                    "timestamp_sec": ts,
                    "predicted_class": r["class_name"],
                    "confidence": conf,
                })
    out.sort(key=lambda d: (d["timestamp_sec"], d["pipeline"]))
    return out


def write_firings(path: Path, firings: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipeline", "timestamp_sec", "mm_ss",
                    "predicted_class", "confidence"])
        for fr in firings:
            w.writerow([
                fr["pipeline"],
                f"{fr['timestamp_sec']:.2f}",
                mmss(fr["timestamp_sec"]),
                fr["predicted_class"],
                f"{fr['confidence']:.4f}",
            ])


def ensure_truth_template(path: Path) -> bool:
    """Create a header-only truth_tackles.csv if it doesn't exist.
    Returns True if we created it, False if we left an existing file alone."""
    if path.exists():
        return False
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "class_name"])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path,
                    default=Path(__file__).parent / "predictions")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "verification")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--t-start", type=float, default=600.0,
                    help="window start in seconds (default 600 = 10:00)")
    ap.add_argument("--t-end",   type=float, default=720.0,
                    help="window end in seconds (default 720 = 12:00)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    firings = load_firings(args.pred_dir, args.threshold, args.t_start, args.t_end)

    per_pipe = {p: 0 for p in PIPELINES}
    for fr in firings:
        per_pipe[fr["pipeline"]] += 1

    print(f"Window     : {mmss(args.t_start)} - {mmss(args.t_end)}  "
          f"({args.t_end - args.t_start:.0f} s)")
    print(f"Threshold  : {args.threshold}")
    print(f"Firings    : total={len(firings)}  " +
          "  ".join(f"{p}={n}" for p, n in per_pipe.items()))

    firings_path = args.out_dir / "firings.csv"
    write_firings(firings_path, firings)
    print(f"  wrote {firings_path}")

    truth_path = args.out_dir / "truth_tackles.csv"
    created = ensure_truth_template(truth_path)
    if created:
        print(f"  wrote {truth_path}  (header-only template)")
        print()
        print("Next: open the broadcast .mkv, scrub through "
              f"{mmss(args.t_start)}-{mmss(args.t_end)} once, "
              "and add one row per real tackle to truth_tackles.csv:")
        print("  timestamp, class_name")
        print("    timestamp accepts seconds (642.4) or mm:ss[.ss] (10:42.40)")
        print("    class_name must be 'tackle-live' or 'tackle-replay'")
        print("Then run: python3 soccernet_experiment/score_verification.py")
    else:
        print(f"  kept   {truth_path}  (already exists, not overwritten)")


if __name__ == "__main__":
    main()
