"""
Post-processing for frame-level predictions → event-level detections.

Converts per-frame class probabilities into discrete event detections
with timestamps and confidence scores. Supports both peak-based (for anchor-trained
models) and segment-based (for interval-trained models) strategies.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing import Optional


def predictions_to_events_peak(
    probs: np.ndarray,
    mask: np.ndarray,
    fps: float = 5.0,
    action_classes: Optional[list[int]] = None,
    sigma: float = 1.0,
    min_confidence: float = 0.3,
    min_distance_sec: float = 1.5,
) -> list[dict]:
    """
    Peak-based event detection. Best for anchor-trained models.
    
    Finds local maxima in per-class probability curves. Good when the model 
    produces peaked outputs centered on event anchors.
    
    Args:
        probs: Softmax probabilities [Seq_Len, num_classes]
        mask: Valid frame mask [Seq_Len] (1.0 = valid, 0.0 = padding)
        fps: Extraction FPS for timestamp conversion
        action_classes: Class indices to detect (default: all except last = background)
        sigma: Gaussian smoothing sigma in frames (0 = no smoothing)
        min_confidence: Minimum peak probability to count as detection
        min_distance_sec: Minimum distance between peaks in seconds
        
    Returns:
        List of detected events, each a dict with:
            - 'class': int, class index
            - 'frame': int, frame index of peak
            - 'timestamp_sec': float, time in seconds
            - 'confidence': float, probability at peak
    """
    seq_len = int(mask.sum())
    num_classes = probs.shape[1]
    
    if action_classes is None:
        action_classes = list(range(num_classes - 1))  # All except background
    
    min_distance_frames = max(1, int(min_distance_sec * fps))
    detections = []
    
    for cls_idx in action_classes:
        # Get probability curve for this class (valid frames only)
        cls_probs = probs[:seq_len, cls_idx].copy()
        
        # Apply Gaussian smoothing to reduce noise
        if sigma > 0:
            cls_probs = gaussian_filter1d(cls_probs, sigma=sigma)
        
        # Find local maxima
        peaks, properties = find_peaks(
            cls_probs,
            height=min_confidence,
            distance=min_distance_frames,
            prominence=0.05,  # Minimum prominence to filter flat regions
        )
        
        for peak_idx in peaks:
            detections.append({
                'class': cls_idx,
                'frame': int(peak_idx),
                'timestamp_sec': peak_idx / fps,
                'confidence': float(cls_probs[peak_idx]),
            })
    
    # Sort by confidence (descending) for NMS-style processing
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    
    return detections


def predictions_to_events_segment(
    preds: np.ndarray,
    probs: np.ndarray,
    mask: np.ndarray,
    fps: float = 5.0,
    action_classes: Optional[list[int]] = None,
    min_segment_frames: int = 2,
    min_confidence: float = 0.3,
) -> list[dict]:
    """
    Segment-based event detection. Best for interval-trained models.
    
    Groups consecutive frames with the same action prediction into segments,
    then returns the center of each segment as the event timestamp.
    
    Args:
        preds: Argmax predictions [Seq_Len]
        probs: Softmax probabilities [Seq_Len, num_classes]
        mask: Valid frame mask [Seq_Len]
        fps: Extraction FPS
        action_classes: Class indices to detect (default: all except last)
        min_segment_frames: Minimum frames in a segment to count as event
        min_confidence: Minimum mean probability to count as event
        
    Returns:
        List of detected events (same format as peak-based)
    """
    seq_len = int(mask.sum())
    num_classes = probs.shape[1]
    
    if action_classes is None:
        action_classes = list(range(num_classes - 1))
    
    detections = []
    
    for cls_idx in action_classes:
        # Find contiguous segments where this class is predicted
        cls_mask = (preds[:seq_len] == cls_idx).astype(np.int32)
        
        # Detect segment boundaries using diff
        # Pad with 0 on both sides to detect segments at edges
        padded = np.concatenate([[0], cls_mask, [0]])
        diff = np.diff(padded)
        
        starts = np.where(diff == 1)[0]   # Segment start indices
        ends = np.where(diff == -1)[0]     # Segment end indices (exclusive)
        
        for seg_start, seg_end in zip(starts, ends):
            seg_len = seg_end - seg_start
            
            if seg_len < min_segment_frames:
                continue
            
            # Compute center of segment
            center_frame = (seg_start + seg_end) // 2
            
            # Compute confidence as max probability over the segment.
            # Using max (not mean) because edge frames have lower probability
            # as the model transitions, which would penalize longer (better) segments.
            # This is also consistent with peak-based detection where confidence = peak value.
            seg_confidence = float(probs[seg_start:seg_end, cls_idx].max())
            
            if seg_confidence < min_confidence:
                continue
            
            detections.append({
                'class': cls_idx,
                'frame': int(center_frame),
                'timestamp_sec': center_frame / fps,
                'confidence': seg_confidence,
            })
    
    # Sort by confidence (descending)
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    
    return detections


def cross_class_nms(
    detections: list[dict],
    min_distance_sec: float = 1.5,
) -> list[dict]:
    """
    Cross-class Non-Maximum Suppression.
    
    If two detections of different classes are within min_distance_sec,
    keep the higher-confidence one. This prevents double-detecting the
    same event as both tackle-live and tackle-replay.
    
    Args:
        detections: List of detected events (sorted by confidence desc)
        min_distance_sec: Minimum temporal distance between detections
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Already sorted by confidence (descending)
    kept = []
    suppressed = set()
    
    for i, det in enumerate(detections):
        if i in suppressed:
            continue
        
        kept.append(det)
        
        # Suppress nearby lower-confidence detections
        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue
            
            time_diff = abs(det['timestamp_sec'] - detections[j]['timestamp_sec'])
            if time_diff < min_distance_sec:
                suppressed.add(j)
    
    return kept


def postprocess_clip(
    logits: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
    fps: float = 5.0,
    method: str = 'auto',
    labeling_mode: str = 'anchor',
    sigma: float = 1.0,
    min_confidence: float = 0.0,
    min_distance_sec: float = 0.5,
    min_segment_frames: int = 2,
    nms: bool = False,
) -> list[dict]:
    """
    Full post-processing pipeline for a single clip.
    
    Args:
        logits: Raw model output [Seq_Len, num_classes]
        mask: Valid frame mask [Seq_Len]
        num_classes: Number of classes
        fps: Extraction FPS
        method: Detection method — 'peak', 'segment', or 'auto' (picks based on labeling_mode)
        labeling_mode: Training labeling strategy ('anchor' or 'interval')
        sigma: Gaussian smoothing sigma (peak method only)
        min_confidence: Minimum detection confidence. Default 0.0 so the
            downstream SoccerNet evaluator (which sweeps 200 confidence
            thresholds in [0, 1]) sees the full peak list — pre-filtering here
            truncates the right end of the per-class PR curve and lowers AP.
            Noise is still suppressed by `prominence` and `distance` in
            find_peaks. Set higher than 0.0 only if reporting non-canonical
            metrics or if peak lists are consumed downstream of the eval.
        min_distance_sec: Per-class minimum distance between consecutive
            detections (passed to find_peaks(distance=...) as the per-class
            NMS step). Default 0.5s. Upper-bounded by the minimum same-class
            inter-event gap observed in TACDEC (0.64s for tackle-live, 0.88s
            for tackle-replay, both center-to-center); going larger silently
            merges genuine rapid-succession event pairs into a single peak.
        min_segment_frames: Minimum segment length (segment method only)
        nms: Whether to apply cross-class non-maximum suppression in addition
            to the per-class NMS already enforced by find_peaks(distance=...).
            Default False to match standard SoccerNet evaluation, where NMS is
            per-class only.

    Returns:
        List of detected events (per-class peaks; cross-class NMS optional).
    """
    # Resolve auto method
    if method == 'auto':
        method = 'peak' if labeling_mode == 'anchor' else 'segment'
    
    # Compute softmax probabilities
    # Numerically stable softmax
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    action_classes = list(range(num_classes - 1))  # All except background
    
    if method == 'peak':
        detections = predictions_to_events_peak(
            probs=probs,
            mask=mask,
            fps=fps,
            action_classes=action_classes,
            sigma=sigma,
            min_confidence=min_confidence,
            min_distance_sec=min_distance_sec,
        )
    elif method == 'segment':
        preds = np.argmax(probs, axis=1)
        detections = predictions_to_events_segment(
            preds=preds,
            probs=probs,
            mask=mask,
            fps=fps,
            action_classes=action_classes,
            min_segment_frames=min_segment_frames,
            min_confidence=min_confidence,
        )
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'peak', 'segment', or 'auto'.")
    
    # Apply cross-class NMS
    if nms:
        detections = cross_class_nms(detections, min_distance_sec=min_distance_sec)
    
    return detections


def extract_ground_truth_events(
    labels: np.ndarray,
    mask: np.ndarray,
    fps: float = 5.0,
    num_classes: int = 3,
) -> list[dict]:
    """
    Extract ground-truth events from frame-level label sequences.
    
    Finds contiguous segments of non-background labels and returns
    the center of each segment as the ground-truth event timestamp.
    
    Args:
        labels: Frame-level labels [Seq_Len]
        mask: Valid frame mask [Seq_Len]
        fps: Extraction FPS
        num_classes: Number of classes
        
    Returns:
        List of ground-truth events with:
            - 'class': int, class index
            - 'frame': int, center frame index
            - 'timestamp_sec': float, center time in seconds
    """
    seq_len = int(mask.sum())
    background_class = num_classes - 1
    action_classes = list(range(background_class))
    
    events = []
    
    for cls_idx in action_classes:
        cls_mask = (labels[:seq_len] == cls_idx).astype(np.int32)
        
        padded = np.concatenate([[0], cls_mask, [0]])
        diff = np.diff(padded)
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for seg_start, seg_end in zip(starts, ends):
            center_frame = (seg_start + seg_end) // 2
            events.append({
                'class': cls_idx,
                'frame': int(center_frame),
                'timestamp_sec': center_frame / fps,
            })
    
    return events
