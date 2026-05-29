"""
Per-class event coverage under the W=10 @ 5 FPS temporal window protocol.

An event is "completely dropped" when none of its 5-FPS frames fall inside the
valid window-centre range [CENTER, n5 - W + CENTER] of its clip, i.e. it sits
wholly within the clip-boundary margin where no centred 10-frame window fits
(no padding). Counts are over all TACDEC label clips, with the incomplete
event types merged into their parent class (matches src/config.py).

Columns: Class | Total | Completely dropped | Total left
"""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.splits import build_frame_labels
from data.temporal_protocol import (
    labels_at_5fps,
    extract_events_5fps,
    _valid_center_range,
    TACKLE_LIVE,
    TACKLE_REPLAY,
)

LABELS = Path(__file__).resolve().parent.parent / "data" / "TACDEC" / "labels"
CLASS_NAME = {TACKLE_LIVE: "tackle-live", TACKLE_REPLAY: "tackle-replay"}


def compute():
    total = Counter()
    dropped = Counter()
    for path in sorted(LABELS.glob("*.json")):
        n5 = len(labels_at_5fps(build_frame_labels(path)))
        min_c, max_c = _valid_center_range(n5)
        for cls, s5, e5 in extract_events_5fps(path):
            total[cls] += 1
            lo, hi = max(s5, min_c), min(e5, max_c)
            if max(0, hi - lo + 1) == 0:
                dropped[cls] += 1
    return total, dropped


def main():
    total, dropped = compute()
    rows = [(CLASS_NAME[c], total[c], dropped[c], total[c] - dropped[c])
            for c in (TACKLE_LIVE, TACKLE_REPLAY)]
    rows.append(("total", sum(total.values()), sum(dropped.values()),
                 sum(total.values()) - sum(dropped.values())))

    w = 16
    print(f"{'Class':<{w}}{'Total':>10}{'Completely dropped':>22}{'Total left':>14}")
    print("-" * (w + 10 + 22 + 14))
    for name, tot, drp, left in rows:
        print(f"{name:<{w}}{tot:>10}{drp:>22}{left:>14}")


if __name__ == "__main__":
    main()
