"""
Shared window/frame-selection protocol for the patch-token attentive probe
comparison.

ONE source-of-truth function defines, for a given target-FPS anchor index,
the source-frame indices fed to the backbone. Both the V-JEPA2 dense
extractor and the DINOv3 stride-indexed loader import it. Any divergence
breaks the comparison: the same anchor must produce byte-identical input
frames across backbones.

Default protocol (Claude Desktop spec, 5 FPS / W=10):

    anchor_stride       =  5 source frames   (25 / 5 = 5, exact integer)
    intra_window_stride =  5 source frames   (= anchor_stride; one frame per output tick)
    window_length       = 10 frames           (10 / 5 = 2.0 s window duration)
    => source-frame span per window = (W - 1) * intra_stride = 45 source frames = 1.8 s

Anchor i -> centred on source frame i * anchor_stride. Indices for the W
frames of that window are produced by ``select_source_frames``. Boundary
indices are clamped to ``[0, video_length - 1]`` (edge replication).

Storage at the default protocol (V-JEPA2 dense):
    tokens per window = (W / 2) * 16 * 16 = 1280   (tubelet 2x16x16, 256x256 input)
    fp16 bytes        = 1280 * 1024 * 2 = ~2.6 MB / window
    full TACDEC       = ~165 GB across 425 videos.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- Default 5 FPS protocol --------------------------------------------------

DEFAULT_SOURCE_FPS = 25
DEFAULT_TARGET_FPS = 5
DEFAULT_ANCHOR_STRIDE = 5
DEFAULT_INTRA_WINDOW_STRIDE = 5
DEFAULT_WINDOW_LENGTH = 10


@dataclass(frozen=True)
class WindowProtocol:
    """Captures the window-selection rules used at extraction and at training time."""
    source_fps: int = DEFAULT_SOURCE_FPS
    target_fps: int = DEFAULT_TARGET_FPS
    anchor_stride: int = DEFAULT_ANCHOR_STRIDE
    intra_window_stride: int = DEFAULT_INTRA_WINDOW_STRIDE
    window_length: int = DEFAULT_WINDOW_LENGTH

    @classmethod
    def from_target_fps(cls, target_fps: int, source_fps: int = DEFAULT_SOURCE_FPS,
                        window_length: int = DEFAULT_WINDOW_LENGTH):
        """Build a protocol where anchor_stride == intra_window_stride == source/target."""
        if source_fps % target_fps != 0:
            raise ValueError(
                f"source_fps={source_fps} not divisible by target_fps={target_fps}; "
                "use exact integer FPS ratios (e.g. 25/5=5)."
            )
        if window_length % 2 != 0:
            raise ValueError(
                f"window_length must be even (V-JEPA2 tubelet=2 requires even T); "
                f"got {window_length}."
            )
        stride = source_fps // target_fps
        return cls(
            source_fps=source_fps,
            target_fps=target_fps,
            anchor_stride=stride,
            intra_window_stride=stride,
            window_length=window_length,
        )

    @property
    def window_duration_sec(self) -> float:
        """Real-time span of one window (first to last frame, in seconds)."""
        return (self.window_length - 1) * self.intra_window_stride / self.source_fps

    @property
    def output_rate_hz(self) -> float:
        """Anchors per second of source video."""
        return self.source_fps / self.anchor_stride


def select_source_frames(
    anchor_idx: int,
    video_length: int,
    *,
    anchor_stride: int = DEFAULT_ANCHOR_STRIDE,
    intra_window_stride: int = DEFAULT_INTRA_WINDOW_STRIDE,
    window_length: int = DEFAULT_WINDOW_LENGTH,
    boundary: str = "clamp",
) -> list[int]:
    """
    Return the W source-frame indices for the window anchored at ``anchor_idx``.

    Convention (lower-middle centre, matches the rest of the codebase):
        offsets = [(k - W//2 + 1) * intra_window_stride for k in range(W)]
        for W=10 stride=5 -> offsets = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
        anchor i centred at source frame i * anchor_stride
        index k -> centre + offsets[k]

    Parameters
    ----------
    anchor_idx : int
        Output-tick index, i.e. the i-th window in target-FPS space.
    video_length : int
        Number of source frames in the video (used for boundary clamping).
    anchor_stride, intra_window_stride, window_length : int
        Protocol parameters. Defaults to the shared 5 FPS protocol.
    boundary : {'clamp', 'zero'}
        How to handle frames whose index falls outside ``[0, video_length)``:
        - 'clamp' (default): clamp to ``[0, video_length - 1]`` (edge replication).
        - 'zero':  return raw indices; caller is responsible for treating
                   out-of-range indices as zero-pad. Used by the DINOv3 loader
                   so the existing zero-pad path keeps working without behaviour
                   change for stride==1 callers.

    Returns
    -------
    list[int] of length ``window_length``.
    """
    centre = anchor_idx * anchor_stride
    offsets = [(k - window_length // 2 + 1) * intra_window_stride
               for k in range(window_length)]
    indices = [centre + off for off in offsets]
    if boundary == "clamp":
        indices = [max(0, min(video_length - 1, idx)) for idx in indices]
    elif boundary == "zero":
        # leave indices as-is so the caller can detect out-of-range ones
        pass
    else:
        raise ValueError(f"boundary must be 'clamp' or 'zero', got {boundary!r}")
    return indices


def valid_anchor_range(
    video_length: int,
    *,
    anchor_stride: int = DEFAULT_ANCHOR_STRIDE,
    intra_window_stride: int = DEFAULT_INTRA_WINDOW_STRIDE,
    window_length: int = DEFAULT_WINDOW_LENGTH,
) -> tuple[int, int]:
    """
    Inclusive range ``[min_anchor, max_anchor]`` of anchors whose window fits
    entirely inside the video without clipping. Useful for callers that want
    to skip boundary-clipped windows (eval mAP) rather than zero/edge-pad.
    """
    half_left = window_length // 2 - 1
    half_right = window_length // 2 + 1
    # Earliest anchor i such that centre - half_left * intra_stride >= 0.
    lo = (half_left * intra_window_stride + anchor_stride - 1) // anchor_stride
    # Latest anchor i such that centre + (half_right - 1) * intra_stride <= V - 1.
    hi = (video_length - 1 - (half_right - 1) * intra_window_stride) // anchor_stride
    return max(0, lo), max(-1, hi)
