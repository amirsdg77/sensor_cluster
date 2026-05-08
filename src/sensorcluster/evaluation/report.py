"""Generate `evaluation_report.md` and `model_card.md` after a training run."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from sensorcluster.evaluation.cv import CVResult
from sensorcluster.evaluation.metrics import InternalMetrics
from sensorcluster.models.label_map import ClusterLabelMap


def _fmt(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def generate_report(
    *,
    out_dir: Path,
    label_map: ClusterLabelMap,
    internal: InternalMetrics,
    cv: CVResult | None,
    stability: dict[str, float] | None,
    extra: dict[str, Any] | None = None,
    umap_plot: Path | None = None,
) -> Path:
    """Write the human-readable training report and matching model card.

    Outputs:
        - ``out_dir/evaluation_report.md`` — full run summary with quality
          flags, per-cluster purity, internal metrics, CV/stability tables,
          and (when supplied) the embedded UMAP plot.
        - ``out_dir/model_card.md`` — Google-style model card driven by the
          same inputs.

    Returns the path to ``evaluation_report.md``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "evaluation_report.md"
    extra = extra or {}

    n_known = sum(1 for e in label_map.entries.values() if e.is_known)
    n_unknown = sum(1 for e in label_map.entries.values() if (not e.is_known) and (not e.is_noise))
    n_noise = sum(1 for e in label_map.entries.values() if e.is_noise)

    # Quality flags appear at the top of the report so degenerate fits are
    # impossible to miss when scanning the artifact.
    quality_flags: list[str] = []
    if internal.n_clusters == 0:
        quality_flags.append(
            "🔴 **No clusters discovered.** All points were assigned to noise. "
            "The data has insufficient density structure for the current "
            "HDBSCAN settings. Either lower `min_cluster_size`/`min_samples`, "
            "engineer additional features, or accept that the inputs are "
            "near-uniform and report that finding to the customer."
        )
    elif internal.noise_fraction > 0.7:
        quality_flags.append(
            f"🟡 **High noise fraction ({internal.noise_fraction:.0%}).** "
            "Most points are not assigned to any cluster. Treat the cluster map "
            "as exploratory; consider tightening `min_cluster_size`."
        )
    if cv is not None and cv.mean_ari < 0.1:
        quality_flags.append(
            f"🔴 **Cross-validated ARI is {cv.mean_ari:.2f}** — the discovered "
            "clusters do not align with the expert labels. The labels may be "
            "noisy, or the sensor features may not contain the signal that "
            "discriminates failure modes. Recommend an EDA pass with the "
            "domain expert before deploying."
        )

    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(UTC).isoformat()}_")
    lines.append("")
    if quality_flags:
        lines.append("## ⚠️ Quality flags")
        lines.append("")
        for f in quality_flags:
            lines.append(f"- {f}")
        lines.append("")
    lines.append("## Cluster summary")
    lines.append("")
    lines.append(f"- Total clusters discovered: **{internal.n_clusters}**")
    lines.append(f"- Of those, mapped to a known label: **{n_known}**")
    lines.append(f"- **`UNKNOWN_*` clusters (candidate undiscovered failure modes): {n_unknown}**")
    lines.append(
        f"- Noise group present: **{'yes' if n_noise else 'no'}** "
        f"({_fmt(internal.noise_fraction, 3)} of all points)"
    )
    lines.append(f"- Total points scored: {internal.n_points}")
    lines.append("")

    lines.append("## Per-cluster detail")
    lines.append("")
    lines.append(
        "| cluster_id | name | is_known | n_total | n_labeled | purity | label distribution |"
    )
    lines.append("|---:|:---|:---:|---:|---:|---:|:---|")
    for entry in sorted(label_map.entries.values(), key=lambda e: (e.is_noise, -e.n_total)):
        dist = ", ".join(f"{k}: {v}" for k, v in entry.label_distribution.items()) or "-"
        lines.append(
            f"| {entry.cluster_id} | {entry.name} | "
            f"{'✅' if entry.is_known else '⚠️'} | "
            f"{entry.n_total} | {entry.n_labeled} | {_fmt(entry.purity)} | {dist} |"
        )
    lines.append("")

    warnings = label_map.warnings()
    if warnings:
        lines.append("## ⚠️ Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Internal metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|:---|---:|")
    lines.append(f"| Silhouette (non-noise) | {_fmt(internal.silhouette)} |")
    lines.append(f"| Davies–Bouldin (non-noise) | {_fmt(internal.davies_bouldin)} |")
    lines.append(f"| Noise fraction | {_fmt(internal.noise_fraction)} |")
    lines.append("")

    if cv is not None:
        lines.append("## Cross-validation on labeled subset")
        lines.append("")
        lines.append(f"- {cv.n_folds}-fold stratified CV over the labeled points")
        lines.append(f"- Mean ARI: **{_fmt(cv.mean_ari)}** ± {_fmt(cv.std_ari)}")
        lines.append("- Per-fold ARI: " + ", ".join(_fmt(a) for a in cv.fold_aris))
        lines.append("")

    if stability is not None:
        lines.append("## Stability (bootstrap)")
        lines.append("")
        lines.append(
            f"- Mean pairwise ARI across {stability.get('n_pairs', 0)} resampled fits: "
            f"**{_fmt(stability.get('stability_mean_ari', float('nan')))}** "
            f"± {_fmt(stability.get('stability_std_ari', float('nan')))}"
        )
        lines.append("")

    if umap_plot is not None and Path(umap_plot).exists():
        rel = Path(umap_plot).name
        lines.append("## UMAP projection")
        lines.append("")
        lines.append(f"![UMAP projection]({rel})")
        lines.append("")

    if extra:
        lines.append("## Run metadata")
        lines.append("")
        for k, v in extra.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    _write_model_card(out_dir, label_map, internal, cv, extra)
    return report_path


def _write_model_card(
    out_dir: Path,
    label_map: ClusterLabelMap,
    internal: InternalMetrics,
    cv: CVResult | None,
    extra: dict[str, Any],
) -> None:
    """Write a Google-style model card driven by the same training summary
    as the evaluation report."""
    n_unknown = sum(1 for e in label_map.entries.values() if (not e.is_known) and (not e.is_noise))
    card = f"""# Model Card — sensorcluster

## Intended use
Triage and clustering of multivariate sensor readings around industrial machine
breakdowns. Outputs a cluster assignment, a confidence score, and a novelty score
for each input. Designed for predictive-maintenance workflows where most
historical breakdowns are unlabeled and the value of "we found a cluster of
events that doesn't look like any known failure" is high.

## Training data
- Multivariate sensor readings (numeric, normalized to ~[-1, 1]).
- A small labeled subset (3 known failure modes) plus a much larger unlabeled
  bulk. Labels treated as ground truth where available.

## Model
- StandardScaler → PCA (auto-`k` for {extra.get("pca_variance_target", 0.95)} variance, kept {extra.get("pca_n_components", "?")} components)
  → HDBSCAN (`min_cluster_size`={extra.get("min_cluster_size", "?")}, `min_samples`={extra.get("min_samples", "?")}).
- Cluster names derived by weighted majority vote over labeled members;
  empty clusters labeled `UNKNOWN_<id>` and surfaced as candidate novel modes.

## Metrics
- Discovered clusters: {internal.n_clusters} (of which UNKNOWN: {n_unknown})
- Silhouette: {_fmt(internal.silhouette)}; Davies–Bouldin: {_fmt(internal.davies_bouldin)}
- Noise fraction: {_fmt(internal.noise_fraction)}
- CV mean ARI: {_fmt(cv.mean_ari) if cv else "n/a"}

## Limitations
- ARI is computed against a tiny labeled set; high CV-ARI is necessary but not
  sufficient. Inspect per-cluster purity and the UMAP plot before deploying.
- HDBSCAN is sensitive to `min_cluster_size`; sweep it before going to production.
- Inputs assumed iid (no temporal structure); time-series extensions are roadmap.

## Ethical & operational considerations
- A miscalled "known cluster" can mask a genuine novel failure mode. The novelty
  score is exposed in the API so downstream consumers can re-threshold.
- Labels reflect expert judgment at training time; expert disagreement is not
  modeled.
"""
    (out_dir / "model_card.md").write_text(card, encoding="utf-8")
