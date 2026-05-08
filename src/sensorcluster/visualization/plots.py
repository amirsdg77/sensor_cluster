"""UMAP plot used by the report generator.

Renders all points coloured by HDBSCAN cluster, overlays the labeled rows
as black-edged markers per class, and rings ``UNKNOWN_*`` clusters in red.
Training-time only; requires the ``train`` extras (umap-learn + matplotlib).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sensorcluster.features.dimreduce import UMAPVisualizer
from sensorcluster.models.label_map import ClusterLabelMap


def save_cluster_umap(
    *,
    X_embedded: np.ndarray,
    cluster_labels: np.ndarray,
    y: pd.Series,
    label_map: ClusterLabelMap,
    out_path: Path,
    seed: int = 42,
) -> Path:
    """Embed `X_embedded` to 2-D via UMAP, render the cluster plot, and save.

    The input is expected to be the PCA projection used for clustering, so
    the embedding visualizes the same space the model sees. Returns the path
    to the saved PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coords = UMAPVisualizer(random_state=seed).fit_transform(X_embedded)

    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    unique = sorted(set(cluster_labels.tolist()))

    # Color noise grey, others from a colormap.
    cmap = plt.get_cmap("tab20")
    colors: dict[int, tuple[float, ...]] = {}
    palette_idx = 0
    for cid in unique:
        if cid == -1:
            colors[cid] = (0.6, 0.6, 0.6, 0.5)
        else:
            colors[cid] = cmap(palette_idx % 20)
            palette_idx += 1

    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    for cid in unique:
        mask = cluster_labels == cid
        name = label_map.name_for(cid)
        is_unknown = (cid != -1) and (not label_map.is_known(cid))
        edge = "red" if is_unknown else "none"
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18 if cid != -1 else 10,
            c=[colors[cid]],
            edgecolors=edge,
            linewidths=0.6 if is_unknown else 0.0,
            label=f"{name} (n={int(mask.sum())})",
            alpha=0.85,
        )

    # Overlay labeled points as black-edged markers per class. Markers are
    # cycled deterministically so adding a new class never silently reuses
    # an existing one (which would make two classes indistinguishable).
    y_arr = y.to_numpy()
    label_classes = sorted({float(v) for v in y_arr if not pd.isna(v)})
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "*", "p"]
    for i, c in enumerate(label_classes):
        mask = y_arr == c
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            marker=marker_cycle[i % len(marker_cycle)],
            label=f"label CLASS_{int(c)}",
        )

    ax.set_title("UMAP projection of PCA-reduced sensor space")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=7, framealpha=0.85, ncols=2)

    # Custom legend entry explaining the red ring.
    extra = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="red",
            markersize=10,
            linewidth=0,
            label="UNKNOWN cluster (no label)",
        ),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=[*handles, *extra],
        labels=[*labels, "UNKNOWN cluster (no label)"],
        loc="best",
        fontsize=7,
        framealpha=0.85,
        ncols=2,
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
