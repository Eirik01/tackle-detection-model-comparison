"""
Verify that our adapter (soccernet_eval.evaluate_average_map) produces the same
numbers as calling SoccerNet's canonical average_mAP directly.

Expected outcome: every comparison passes. SoccerNet's average_mAP and our
adapter both use trapezoidal AUC for the headline AND for per-class
aggregation across tolerances; per-tolerance values come from delta_curve
either way.

Run with: uv run python tests/canonical_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import numpy as np

from SoccerNet.Evaluation.ActionSpotting import average_mAP as soccernet_average_mAP
from SoccerNet.Evaluation.ActionSpotting import delta_curve as soccernet_delta_curve

from soccernet_eval import _build_clip_vectors, evaluate_average_map


def make_synthetic_set(
    n_clips: int = 50,
    seq_len: int = 600,
    n_events_per_clip: int = 2,
    fps: float = 25.0,
    detection_jitter_frames: int = 5,
    detection_recall: float = 0.85,
    n_false_positives_per_clip: int = 3,
    seed: int = 0,
) -> tuple[list, list, list]:
    """Build a synthetic detections/ground-truths set for the parity test."""
    rng = np.random.default_rng(seed)
    all_dets, all_gts, all_seq_lens = [], [], []

    for _ in range(n_clips):
        gt_frames = rng.choice(
            np.arange(50, seq_len - 50), size=n_events_per_clip, replace=False
        )
        gts = []
        for i, f in enumerate(sorted(gt_frames)):
            cls = i % 2
            gts.append({'class': int(cls), 'frame': int(f), 'timestamp_sec': float(f / fps)})

        dets = []
        for g in gts:
            if rng.random() < detection_recall:
                jitter = int(rng.integers(-detection_jitter_frames, detection_jitter_frames + 1))
                det_frame = max(0, min(seq_len - 1, g['frame'] + jitter))
                dets.append({
                    'class': g['class'],
                    'frame': det_frame,
                    'timestamp_sec': float(det_frame / fps),
                    'confidence': float(rng.uniform(0.7, 0.95)),
                })

        for _ in range(n_false_positives_per_clip):
            fp_cls = int(rng.integers(0, 2))
            fp_frame = int(rng.integers(0, seq_len))
            dets.append({
                'class': fp_cls,
                'frame': fp_frame,
                'timestamp_sec': float(fp_frame / fps),
                'confidence': float(rng.uniform(0.2, 0.6)),
            })

        all_dets.append(dets)
        all_gts.append(gts)
        all_seq_lens.append(seq_len)

    return all_dets, all_gts, all_seq_lens


def main():
    print("=" * 70)
    print("CANONICAL PARITY TEST (adapter vs SoccerNet average_mAP)")
    print("=" * 70)

    all_dets, all_gts, all_seq_lens = make_synthetic_set(seed=0)
    fps = 25.0
    num_action_classes = 2
    deltas = np.arange(5) + 1

    # Build the SoccerNet-format vectors via our helper.
    targets, detections, closests = [], [], []
    for clip_dets, clip_gts, seq_len in zip(all_dets, all_gts, all_seq_lens):
        t, d, c = _build_clip_vectors(clip_dets, clip_gts, num_action_classes, int(seq_len))
        targets.append(t)
        detections.append(d)
        closests.append(c)

    # Call SoccerNet's average_mAP directly on the same vectors.
    print("\nCalling SoccerNet.Evaluation.ActionSpotting.average_mAP directly...")
    a_mAP_sn, a_mAP_per_class_sn, *_ = soccernet_average_mAP(
        targets, detections, closests, framerate=fps, deltas=deltas
    )

    # Call SoccerNet's delta_curve directly for per-tolerance reference.
    print("Calling SoccerNet.Evaluation.ActionSpotting.delta_curve directly...")
    mAP_per_delta_sn, mAP_per_class_per_delta_sn, *_ = soccernet_delta_curve(
        targets, closests, detections, framerate=fps, deltas=deltas
    )

    # Call our adapter on the same input.
    print("Calling our soccernet_eval.evaluate_average_map adapter...\n")
    ours = evaluate_average_map(
        all_detections=all_dets,
        all_ground_truths=all_gts,
        all_seq_lens=all_seq_lens,
        num_classes=3,
        fps=fps,
        metric='tight',
        class_names=['Tackle-Live', 'Tackle-Replay'],
        verbose=False,
    )

    # ─── Compare ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print("\n[Headline Average-mAP]")
    print(f"  SoccerNet average_mAP : {a_mAP_sn * 100:>8.4f}%")
    print(f"  Our adapter           : {ours['average_mAP'] * 100:>8.4f}%")
    diff = (ours['average_mAP'] - a_mAP_sn) * 100
    flag = "PASS" if abs(diff) < 1e-6 else "FAIL"
    print(f"  Δ                     : {diff:>+8.4f} pts  [{flag}]")

    print("\n[Per-tolerance mAP]")
    print(f"  {'δ (s)':>6s} | {'SoccerNet delta_curve':>22s} | {'Our adapter':>14s} | {'Δ':>10s}")
    print("  " + "-" * 60)
    for delta_sec, sn_val, our_row in zip(deltas, mAP_per_delta_sn, ours['per_tolerance']):
        d = (our_row['mAP'] - sn_val) * 100
        flag = "PASS" if abs(d) < 1e-6 else "FAIL"
        print(f"  {delta_sec:>6d} | {sn_val * 100:>21.4f}% | {our_row['mAP'] * 100:>13.4f}% | {d:>+8.4f}  [{flag}]")

    print("\n[Per-class Average-AP across tolerances]")
    print("  (Both use trapezoidal AUC; numbers should match exactly.)")
    print(f"  {'class':>14s} | {'SoccerNet':>12s} | {'Ours':>12s} | {'Δ':>10s}")
    print("  " + "-" * 56)
    class_names = ['Tackle-Live', 'Tackle-Replay']
    for c in range(num_action_classes):
        sn_val = a_mAP_per_class_sn[c] * 100
        our_val = ours['per_class_avg_ap'][c] * 100
        d = our_val - sn_val
        flag = "PASS" if abs(d) < 1e-6 else "DIFF"
        print(f"  {class_names[c]:>14s} | {sn_val:>11.4f}% | {our_val:>11.4f}% | {d:>+8.4f}  [{flag}]")

    print("\n" + "=" * 70)
    print("All three (headline, per-tolerance, per-class) must match exactly.")
    print("Any FAIL here means the adapter has drifted from canonical SoccerNet.")
    print("=" * 70)


if __name__ == "__main__":
    main()
