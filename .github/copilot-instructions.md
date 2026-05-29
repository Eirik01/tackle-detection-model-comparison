# GitHub Copilot Instructions - TACDEC Action Spotting

## Project Context
Master's thesis investigating how spatial and spatio-temporal foundation models compare as backbones for tackle event detection in broadcast footage.
- **Goal:** Compare spatial (DINOv3) vs spatio-temporal (V-JEPA2) encoders with identical temporal heads to isolate representational quality differences.
- **Critical:** Frame-level metrics AND event-level metrics (Average-mAP) are mandatory.
- Prevent overfitting on small data (340 clips) via strict model constraints.

## Research Question & Objectives

**Research Question:** How do spatial and spatio-temporal foundation models compare as backbones for tackle event detection in broadcast footage?

**Objectives:**

1. **Objective 1: Efficacy and Efficiency**
   - Evaluate the comparative baseline performance of spatial and spatio-temporal foundation models for tackle detection.
   - Analyze trade-off between detection performance and computational efficiency for feasibility in live broadcast pipelines.

2. **Objective 2: The Role of Post-Hoc Temporal Aggregation**
   - Assess the role of post-hoc temporal aggregation by comparing tackle detection performance using a linear probe versus an attentive probe.
   - Compare performance on both spatial and spatio-temporal foundation model backbones.

3. **Objective 3: Robustness to Broadcast Distortion**
   - Investigate robustness of spatial and spatio-temporal representations under broadcast distortion.
   - Quantitatively analyze detection performance on replay clips featuring spatial occlusions, scale shifts, and temporal scaling.

4. **Objective 4: Discriminative Capacity**
   - Evaluate models' discriminative capacity on untrimmed broadcast footage.
   - Assess foundation models' resilience to false positives for background events.

## Coding Standards
- **Style:** Python 3.11+. Use Type Hints for all function arguments/returns.
- **Docs:**: Do not create docs, unless explicitly requested
- **Imports:** Never use `src.` prefix within `src/` files (e.g., `from config import...`, NOT `from src.config`).

## Dataset Structure (TACDEC)
- **FPS:** 8.0.
- **Classes (5-class mode):**
  0: tackle-live | 1: tackle-replay | 2: live-inc | 3: replay-inc | 4: background
- **3-class Merged Mode:**
  - Live-Inc (2) → Tackle-Live (0)
  - Replay-Inc (3) → Tackle-Replay (1)
  - Background (4) → Background (2)
- **Labeling Strategy (Ablation Study):**
  - `anchor` mode: Center ± tolerance → peaked targets (better for peak-based spotting)
  - `interval` mode: Full event range ± tolerance → plateau targets (uses annotation boundaries)
  - Run BOTH modes and compare event-level mAP to determine which is better for TACDEC.

## Architecture Pipeline
- **Feature Extraction:**
  - DINOv3: Per-frame (no temporal context in features)
  - V-JEPA2: Sliding 16-frame windows (stride 3, has temporal context)
- **Heads (the three thesis pipelines):**
    - DINOv3 + linear probe: per-frame linear classifier on the CLS token. Spatial-only baseline, no temporal modeling. Evaluated with 5-fold game-disjoint CV.
    - DINOv3 + attentive probe: temporal reasoning lives in the head. Single 70/15/15 split.
    - V-JEPA2 + attentive probe: temporal reasoning lives in the backbone. Single 70/15/15 split.
  - **Class Weights:** Inverse frequency normalized to min=1.0. **Never** use raw inverse counts.
- **Output:** Frame-level predictions (one class per frame)

## Evaluation Pipeline
1. **Frame-Level (secondary):** Per-class precision/recall/F1, macro F1. Verifies feature discrimination.
2. **Event-Level (primary — thesis standard):**
   - **Post-processing:** `postprocess.py` converts frame predictions → event detections.
     - `auto` method: Picks `peak` for anchor-trained models, `segment` for interval-trained.
     - `peak`: Gaussian smoothing → local maxima → cross-class NMS.
     - `segment`: Contiguous segments → center extraction → cross-class NMS.
   - **Metric:** `event_metrics.py` computes **Average-mAP** over tolerances δ ∈ {1, 2, 3, 4, 5}s.
   - Both are automatically run by `validate.py`.
   
## Research Question in Context
The primary research question aligns with the 4 objectives above:
- **RQ Focus:** Can spatial encoders (DINOv3) with learned temporal aggregation match spatio-temporal encoders (V-JEPA2) on tackle detection tasks?
- **Evaluation Scope:** This is tested across efficiency constraints, temporal modeling approaches, broadcast distortions, and on untrimmed footage with background clutter.
