"""Cluster the DENSE features the attentive probes actually consume.

Companion to ``feature_clustering_figure.py`` (which clusters the CLS features
that the DINOv3 + Linear Probe consumes). For each backbone this script loads
the same dense-token cache used by the attentive probe at training/eval time,
mean-pools tokens per window to a single 1024-dim feature, and computes the
same silhouette / Calinski-Harabasz / cosine-similarity metrics used by the
CLS script. Two t-SNE panels are written to a single PDF so the live/replay
separation can be compared visually.

Run on fox (where the dense caches live):

    cd <repo>/thesis_code
    python visualization/feature_clustering_dense.py 2>&1 | tee /tmp/feature_clustering_dense.log

Outputs:
    figures/feature_clustering_dense.pdf
    figures/feature_clustering_dense_metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import TACDEC_FEATURES, TACDEC_LABELS  # noqa: E402
from data.balanced_temporal_dataset import (  # noqa: E402
    get_balanced_temporal_dataloaders,
)

CLASS_NAMES = ["Background", "Tackle-Live", "Tackle-Replay"]
CLASS_COLOURS = {0: "#666666", 1: "#d24545", 2: "#e89126"}

# Mirrors the run_train_eval_temporal_{dinov3,vjepa2}.sh `--protocol centered`
# default: balanced W=10 @ 5 FPS windows, reflect padding, 70/15/15 split.
TARGET_FPS = 5.0
WINDOW_SIZE = 10
PADDING_MODE = "reflect"


def collect_test_features(backbone: str, features_dir: Path,
                          source_fps: float, seed: int = 42):
    """Iterate the balanced test loader for one backbone, mean-pool tokens per
    window, and return (features [N, D], labels [N])."""
    _train, _val, test_loader, info = get_balanced_temporal_dataloaders(
        labels_dir=TACDEC_LABELS,
        features_dir=features_dir,
        backbone=backbone,
        window_size=WINDOW_SIZE,
        target_fps=TARGET_FPS,
        source_fps=source_fps,
        seed=seed,
        batch_size=16,
        num_workers=0,
        feature_loader_cache=4,
        dense_tag=PADDING_MODE,
    )
    label = f"{backbone}-L"
    print(f"  {label} split: test_games={len(info['game_ids']['test'])} "
          f"test_windows={info['n_sequences']['test']}")

    feats, labels = [], []
    for batch in test_loader:
        pooled = batch["features"].mean(dim=1).cpu().numpy().astype(np.float32)
        feats.append(pooled)
        labels.append(batch["labels"].cpu().numpy())
    F = np.concatenate(feats, axis=0)
    L = np.concatenate(labels, axis=0).astype(np.int64)
    print(f"  {label} pooled to dim {F.shape[1]}; "
          f"class counts {np.bincount(L, minlength=3).tolist()}")
    return F, L


def compute_metrics(features: np.ndarray, labels: np.ndarray,
                    model_name: str) -> dict:
    """Silhouette + Calinski-Harabasz + per-class intra-similarity + per-pair
    inter-similarity (cosine). Mirrors compute_clustering_metrics in the CLS
    script but stores everything per-class/per-pair so we can read the
    live-vs-replay number out of the JSON."""
    print(f"\n=== Clustering metrics: {model_name} ===")

    silh = float(silhouette_score(features, labels))
    cal = float(calinski_harabasz_score(features, labels))
    print(f"Silhouette Score: {silh:.4f}")
    print(f"Calinski-Harabasz Index: {cal:.2f}")

    intra = {}
    print("Intra-class cosine similarity:")
    for c in np.unique(labels):
        m = labels == c
        if m.sum() <= 1:
            continue
        sim = cosine_similarity(features[m])
        avg = float(np.mean(sim[np.triu_indices_from(sim, k=1)]))
        intra[CLASS_NAMES[int(c)]] = avg
        print(f"  {CLASS_NAMES[int(c)]:14s}: {avg:.4f}")

    inter = {}
    print("Inter-class cosine similarity:")
    unique = np.unique(labels)
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            ci, cj = int(unique[i]), int(unique[j])
            sim = cosine_similarity(features[labels == ci], features[labels == cj])
            avg = float(np.mean(sim))
            key = f"{CLASS_NAMES[ci]} vs {CLASS_NAMES[cj]}"
            inter[key] = avg
            print(f"  {key:34s}: {avg:.4f}")

    intra_avg = float(np.mean(list(intra.values())))
    inter_avg = float(np.mean(list(inter.values())))
    sep = intra_avg / inter_avg if inter_avg > 0 else float("inf")
    print(f"Avg intra: {intra_avg:.4f}   Avg inter: {inter_avg:.4f}   Ratio: {sep:.4f}")

    return {
        "silhouette": silh,
        "calinski_harabasz": cal,
        "intra_class": intra,
        "inter_class": inter,
        "intra_avg": intra_avg,
        "inter_avg": inter_avg,
        "separation_ratio": sep,
        "n_samples": int(features.shape[0]),
    }


def plot_tsne_panel(ax, features: np.ndarray, labels: np.ndarray, title: str):
    emb = TSNE(n_components=2, perplexity=30.0, learning_rate="auto",
               init="pca", random_state=42).fit_transform(features)
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7,
                   c=CLASS_COLOURS[int(c)], label=CLASS_NAMES[int(c)],
                   edgecolors="none")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "figures",
                    help="output directory (default: <repo>/figures)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DINOv3 dense features (balanced centred test split)...")
    F_d, L_d = collect_test_features(
        backbone="dinov3",
        features_dir=TACDEC_FEATURES / "dinov3_large",
        source_fps=25.0, seed=SEED,
    )

    print("\nLoading V-JEPA 2 dense features (balanced centred test split)...")
    F_v, L_v = collect_test_features(
        backbone="vjepa2",
        features_dir=TACDEC_FEATURES / "vjepa2_large",
        source_fps=5.0, seed=SEED,
    )

    metrics = {
        "config": {
            "target_fps": TARGET_FPS, "window_size": WINDOW_SIZE,
            "padding_mode": PADDING_MODE, "protocol": "centered (balanced)",
            "seed": SEED, "pooling": "mean-over-tokens",
        },
        "DINOv3-Large_dense": compute_metrics(F_d, L_d, "DINOv3-Large (dense, mean-pooled)"),
        "VJEPA2-Large_dense": compute_metrics(F_v, L_v, "V-JEPA 2-Large (dense, mean-pooled)"),
    }

    # Headline comparison line — what we actually want to read
    print("\n" + "=" * 64)
    print("HEADLINE: live-vs-replay inter-class cosine similarity")
    print("=" * 64)
    key = "Tackle-Live vs Tackle-Replay"
    print(f"  DINOv3-Large  (dense, attentive input): "
          f"{metrics['DINOv3-Large_dense']['inter_class'][key]:.4f}")
    print(f"  V-JEPA 2-Large (dense, attentive input): "
          f"{metrics['VJEPA2-Large_dense']['inter_class'][key]:.4f}")
    print("(Higher = more collapsed; lower = better separated.)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    plot_tsne_panel(axes[0], F_d, L_d,
                    "DINOv3-Large dense (attentive probe input)")
    plot_tsne_panel(axes[1], F_v, L_v,
                    "V-JEPA 2-Large dense (attentive probe input)")
    handles = [mpatches.Patch(color=CLASS_COLOURS[c], label=CLASS_NAMES[c]) for c in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("t-SNE of dense window features on the TACDEC test split "
                 f"(mean-pooled, W={WINDOW_SIZE} @ {TARGET_FPS:g} FPS, {PADDING_MODE})")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    pdf_path = args.out_dir / "feature_clustering_dense.pdf"
    json_path = args.out_dir / "feature_clustering_dense_metrics.json"
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote: {pdf_path}")
    print(f"Wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
