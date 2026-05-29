"""
SoccerNet-canonical Average-mAP evaluation for TACDEC action spotting.

Wraps SoccerNet.Evaluation.ActionSpotting.{average_mAP,delta_curve} so the
metric exactly matches what published SoccerNet papers report. The adapter
converts our per-clip detection/ground-truth dict lists into the
(target, detection, closest) numpy vector triples the canonical evaluator
expects, then calls it in-memory (no JSON file I/O).

Detection format expected by this module (matches src/postprocess.py output):
    list of {'class': int, 'frame': int, 'timestamp_sec': float, 'confidence': float}

Ground-truth format expected (matches src/validate.py construction):
    list of {'class': int, 'frame': int, 'timestamp_sec': float}

Conventions to be aware of when comparing to published SoccerNet numbers:
  - The 'tight' grid is delta in [1, 2, 3, 4, 5] seconds.
  - SoccerNet's match window is +/- delta/2, not +/- delta. So 'tight' delta=5s
    means match within +/- 2.5s. Same convention used by all SoccerNet papers.
  - SoccerNet's per-class AP uses 11-point VOC interpolation.
  - Average-mAP across deltas is the trapezoidal AUC normalized by the number
    of intervals (len(deltas) - 1).
  - SoccerNet eval applies no NMS itself. NMS is the user's responsibility
    upstream in src/postprocess.py.
"""

from __future__ import annotations

import numpy as np

from SoccerNet.Evaluation.ActionSpotting import delta_curve


_TIGHT_DELTAS = np.arange(5) + 1               # [1, 2, 3, 4, 5] seconds
_LOOSE_DELTAS = np.arange(12) * 5 + 5          # [5, 10, ..., 60] seconds


def _build_clip_vectors(
    detections: list[dict],
    ground_truths: list[dict],
    num_action_classes: int,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert one clip's detection + GT lists into SoccerNet-format vectors.

    Returns
    -------
    target : (seq_len, num_action_classes) ndarray
        1.0 at frames where a GT event of that class is centered, 0 elsewhere.
    detection : (seq_len, num_action_classes) ndarray
        Confidence at frames with a predicted peak, -1 elsewhere (placeholder
        for "no detection" expected by SoccerNet's compute_class_scores).
    closest : (seq_len, num_action_classes) ndarray
        Voronoi assignment per class: each frame holds the value of the nearest
        GT event in that class. Used by SoccerNet for visibility flag matching.
    """
    target = np.zeros((seq_len, num_action_classes), dtype=np.float32)
    detection = np.full((seq_len, num_action_classes), -1.0, dtype=np.float32)
    closest = np.zeros((seq_len, num_action_classes), dtype=np.float32) - 1.0

    for gt in ground_truths:
        cls = int(gt['class'])
        if cls < 0 or cls >= num_action_classes:
            continue
        frame = int(gt['frame'])
        if 0 <= frame < seq_len:
            target[frame, cls] = 1.0

    for det in detections:
        cls = int(det['class'])
        if cls < 0 or cls >= num_action_classes:
            continue
        frame = int(det['frame'])
        if 0 <= frame < seq_len:
            # If multiple peaks land on the same frame, keep the higher score.
            detection[frame, cls] = max(float(detection[frame, cls]), float(det['confidence']))

    # Voronoi closest-GT vector per class, mirroring the construction in
    # SoccerNet.Evaluation.ActionSpotting.evaluate.
    for c in range(num_action_classes):
        gt_idx = np.where(target[:, c] != 0)[0].tolist()
        if not gt_idx:
            continue
        gt_idx.insert(0, -gt_idx[0])
        gt_idx.append(2 * seq_len)
        for i in range(1, len(gt_idx) - 1):
            start = max(0, (gt_idx[i - 1] + gt_idx[i]) // 2)
            stop = min(seq_len, (gt_idx[i] + gt_idx[i + 1]) // 2)
            closest[start:stop, c] = target[gt_idx[i], c]

    return target, detection, closest


def evaluate_average_map(
    all_detections: list[list[dict]],
    all_ground_truths: list[list[dict]],
    all_seq_lens: list[int],
    num_classes: int = 3,
    fps: float = 25.0,
    metric: str = 'tight',
    class_names: list[str] | None = None,
    verbose: bool = True,
) -> dict:
    """
    SoccerNet-canonical Average-mAP across multiple temporal tolerances.

    Parameters
    ----------
    all_detections : list of per-clip detection lists.
    all_ground_truths : list of per-clip ground-truth lists.
    all_seq_lens : per-clip valid frame counts (e.g. int(mask.sum())).
    num_classes : total class count including background; background is the last
        index and is excluded from Average-mAP.
    fps : extraction FPS that the detection/GT 'frame' fields are indexed at.
        Passed to SoccerNet as 'framerate'.
    metric : 'tight' (delta in {1..5}s, the SoccerNet-v3 tight metric),
        'loose' (delta in {5,10,..,60}s, the SoccerNet-v1/v2 loose metric),
        or 'at1'..'at5' for a single-tolerance metric.
    class_names : optional list of action-class names for verbose output.
    verbose : if True, print a per-tolerance per-class breakdown.

    Returns
    -------
    dict with keys:
        'average_mAP' : float, headline metric.
        'per_class_avg_ap' : dict[class_idx -> float], AP averaged across deltas.
        'per_tolerance' : list of dicts, one per delta, each containing
            'tolerance_sec', 'mAP', 'per_class_ap'.
        'tolerances' : list[float], the delta grid in seconds.
        'metric' : str, the requested metric name.
    """
    if not (len(all_detections) == len(all_ground_truths) == len(all_seq_lens)):
        raise ValueError(
            f"Mismatched lengths: {len(all_detections)} detections, "
            f"{len(all_ground_truths)} ground-truths, {len(all_seq_lens)} seq_lens"
        )

    num_action_classes = num_classes - 1
    if num_action_classes <= 0:
        raise ValueError(f"num_classes must be > 1 (got {num_classes})")

    if metric == 'tight':
        deltas = _TIGHT_DELTAS
    elif metric == 'loose':
        deltas = _LOOSE_DELTAS
    elif metric.startswith('at'):
        deltas = np.array([int(metric[2:])])
    else:
        raise ValueError(f"Unknown metric '{metric}'. Use 'tight', 'loose', 'at1'..'at5'.")

    targets, detections, closests = [], [], []
    for clip_dets, clip_gts, seq_len in zip(all_detections, all_ground_truths, all_seq_lens):
        t, d, c = _build_clip_vectors(clip_dets, clip_gts, num_action_classes, int(seq_len))
        targets.append(t)
        detections.append(d)
        closests.append(c)

    # delta_curve returns the per-delta breakdown we want for the per-tolerance
    # table; average_mAP would re-run delta_curve internally, so we compute it
    # here once and aggregate ourselves to match SoccerNet's trapezoidal rule.
    mAP_list, mAP_per_class_list, _, _, _, _ = delta_curve(
        targets, closests, detections, framerate=fps, deltas=deltas
    )

    if len(mAP_list) == 1:
        a_mAP = float(mAP_list[0])
        a_mAP_per_class = np.array(mAP_per_class_list[0], dtype=np.float64)
    else:
        # Trapezoidal AUC normalised by the number of intervals — same formula
        # as SoccerNet's average_mAP.
        integral = sum((mAP_list[i] + mAP_list[i + 1]) / 2.0 for i in range(len(mAP_list) - 1))
        a_mAP = float(integral / (len(mAP_list) - 1))
        per_class_stack = np.stack(mAP_per_class_list)  # (num_deltas, num_classes)
        per_class_integral = np.sum(
            (per_class_stack[:-1] + per_class_stack[1:]) / 2.0, axis=0
        )
        a_mAP_per_class = per_class_integral / (len(mAP_list) - 1)

    per_tolerance = []
    for delta_sec, mAP_val, mAP_per_class in zip(deltas, mAP_list, mAP_per_class_list):
        per_tolerance.append({
            'tolerance_sec': float(delta_sec),
            'mAP': float(mAP_val),
            'per_class_ap': {i: float(mAP_per_class[i]) for i in range(num_action_classes)},
        })

    per_class_avg = {i: float(a_mAP_per_class[i]) for i in range(num_action_classes)}

    if verbose:
        names = class_names if class_names else [f"Class-{i}" for i in range(num_action_classes)]
        names = list(names)[:num_action_classes]
        print("\n" + "=" * 70)
        print(f"SOCCERNET-CANONICAL EVALUATION (metric='{metric}', framerate={fps})")
        print("=" * 70)
        header = f"{'Tolerance':>10s}"
        for name in names:
            header += f" | {name:>14s}"
        header += f" | {'mAP':>8s}"
        print(header)
        print("-" * len(header))
        for row in per_tolerance:
            line = f"{'±' + f'{row['tolerance_sec']:.0f}s':>10s}"
            for i in range(num_action_classes):
                line += f" | {row['per_class_ap'][i] * 100:>13.2f}%"
            line += f" | {row['mAP'] * 100:>7.2f}%"
            print(line)
        print("-" * len(header))
        avg_line = f"{'Average':>10s}"
        for i in range(num_action_classes):
            avg_line += f" | {per_class_avg[i] * 100:>13.2f}%"
        avg_line += f" | {a_mAP * 100:>7.2f}%"
        print(avg_line)
        print("=" * 70)
        # Greppable per-tolerance breakdown for downstream aggregators.
        for row in per_tolerance:
            print(f"mAP @ ±{row['tolerance_sec']:.0f}s: {row['mAP'] * 100:.2f}%")
        # Headline keeps the legacy ">>> Average-mAP: X%" format so existing
        # regex-based metric collectors continue to parse correctly.
        print(f"\n  >>> Average-mAP: {a_mAP * 100:.2f}% <<<  (SoccerNet-canonical, metric={metric})\n")

    return {
        'average_mAP': a_mAP,
        'per_class_avg_ap': per_class_avg,
        'per_tolerance': per_tolerance,
        'tolerances': deltas.astype(float).tolist(),
        'metric': metric,
    }
