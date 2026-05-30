"""Shared plotting helpers for the spatial pipeline (train + eval)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.labels import CLASS_NAMES, CLASS_ORDER


def save_confusion_matrix(cm: np.ndarray, output_path: Path, title: str | None = None) -> None:
    """Save a confusion-matrix heatmap as a PNG. `title` is optional (training
    omits it; evaluation passes a distribution label)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels([CLASS_NAMES[c] for c in CLASS_ORDER], rotation=30, ha="right")
    ax.set_yticklabels([CLASS_NAMES[c] for c in CLASS_ORDER])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
