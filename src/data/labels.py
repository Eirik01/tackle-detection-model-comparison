"""Canonical 3-class scheme for TACDEC tackle detection (after the 5 -> 3 merge).

Single source of truth shared by the spatial and temporal pipelines. Both naming
conventions used in the codebase resolve here: the bare ``TACKLE_LIVE`` /
``TACKLE_REPLAY`` / ``BACKGROUND`` (spatial side) and the ``CLASS_*``-prefixed
aliases imported by the temporal loaders.
"""

from __future__ import annotations


TACKLE_LIVE = 0
TACKLE_REPLAY = 1
BACKGROUND = 2

CLASS_ORDER = [TACKLE_LIVE, TACKLE_REPLAY, BACKGROUND]

CLASS_NAMES = {
    TACKLE_LIVE: "tackle-live",
    TACKLE_REPLAY: "tackle-replay",
    BACKGROUND: "background",
}

# Class names in CLASS_ORDER (sklearn target_names / confusion-matrix headers).
CLASS_NAMES_ORDER = [CLASS_NAMES[c] for c in CLASS_ORDER]

# 5 -> 3 merge: raw TACDEC event-type strings -> 3-class index.
LIVE_TYPES = {"tackle-live", "tackle-live-incomplete"}
REPLAY_TYPES = {"tackle-replay", "tackle-replay-incomplete"}
