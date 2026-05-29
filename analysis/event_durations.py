"""Print average segment duration for the three target classes on TACDEC.

Classes:
  Live    -- merged tackle-live / tackle-live-incomplete events
  Replay  -- merged tackle-replay / tackle-replay-incomplete events
  Background -- contiguous runs of frames inside a clip not covered by any event

Durations are reported in seconds, using each clip's own frame rate.
A segment spanning frames [a, b] (inclusive) lasts (b - a + 1) / fps seconds.
"""

import glob
import json
import os
import statistics as st
from collections import defaultdict

LABELS = os.path.join(os.path.dirname(__file__), "..", "data", "TACDEC", "labels", "*.json")


def merge_intervals(intervals):
    """Merge overlapping/adjacent [start, end] inclusive intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1:          # overlapping or touching
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def main():
    durations = defaultdict(list)  # class -> list of durations in seconds

    for path in sorted(glob.glob(LABELS)):
        d = json.load(open(path))
        fps = d.get("media_attributes", {}).get("frame_rate", 25)
        n_frames = d.get("media_attributes", {}).get("frame_count")
        events = d.get("events", [])
        if n_frames is None:
            continue

        covered = []
        for e in events:
            if "frame_start" not in e or "frame_end" not in e:
                continue
            s, end = e["frame_start"], e["frame_end"]
            covered.append((s, end))
            cls = "tackle-replay" if "replay" in e.get("type", "") else "tackle-live"
            durations[cls].append((end - s + 1) / fps)

        # Background = complement of merged event spans within [0, n_frames - 1]
        cursor = 0
        for s, end in merge_intervals(covered):
            if s > cursor:                              # gap before this event
                durations["background"].append((s - cursor) / fps)
            cursor = max(cursor, end + 1)
        if cursor < n_frames:                           # trailing gap after last event
            durations["background"].append((n_frames - cursor) / fps)

    print(f"{'class':14s} {'count':>5} {'mean_s':>8} {'median_s':>9} {'std_s':>7} {'min_s':>6} {'max_s':>6}")
    for cls in ("tackle-live", "tackle-replay", "background"):
        v = durations[cls]
        print(f"{cls:14s} {len(v):5d} {st.mean(v):8.2f} {st.median(v):9.2f} "
              f"{st.pstdev(v):7.2f} {min(v):6.2f} {max(v):6.2f}")


if __name__ == "__main__":
    main()
