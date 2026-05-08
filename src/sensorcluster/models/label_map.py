"""Cluster-id-to-label data structures persisted alongside the model.

These types are pure data artifacts that ride with the trained model. The
factory that *builds* them from training labels lives in
:mod:`sensorcluster.pipeline.label_propagation`; the data structures live
here so that downstream consumers (the report generator, the inference
pipeline, the API) can depend on them without pulling in the orchestration
layer.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Bumping this string is how we signal incompatible changes to the on-disk
# layout to old loaders. Loaders are expected to refuse versions they don't
# recognise rather than silently misinterpret fields.
LABEL_MAP_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class ClusterLabelEntry:
    """One row of the cluster->label map.

    Attributes:
        cluster_id: HDBSCAN cluster id (-1 for the noise group).
        name: Human-readable label (``CLASS_n`` / ``UNKNOWN_<id>`` / ``NOISE``).
        is_known: True when the cluster mapped to one of the known classes.
        is_noise: True when ``cluster_id`` is -1.
        purity: Largest class share among labeled members of the cluster, or
            NaN when the cluster has no labeled members.
        n_labeled: Number of labeled rows that fell into the cluster.
        n_total: Total number of rows (labeled + unlabeled) in the cluster.
        label_distribution: Stringified counts of each labeled class present
            in the cluster, e.g. ``{"1.0": 4, "2.0": 1}``.
    """

    cluster_id: int
    name: str
    is_known: bool
    is_noise: bool
    purity: float
    n_labeled: int
    n_total: int
    label_distribution: dict[str, int]


@dataclass
class ClusterLabelMap:
    """Map from HDBSCAN cluster id to its ``ClusterLabelEntry``."""

    entries: dict[int, ClusterLabelEntry] = field(default_factory=dict)
    purity_warning: float = 0.6

    def name_for(self, cluster_id: int) -> str:
        """Return the label name for `cluster_id`, falling back to ``NOISE``
        for ids not seen during training."""
        entry = self.entries.get(int(cluster_id))
        if entry is None:
            return "NOISE"
        return entry.name

    def is_known(self, cluster_id: int) -> bool:
        entry = self.entries.get(int(cluster_id))
        return bool(entry and entry.is_known)

    def warnings(self) -> list[str]:
        msgs = []
        for entry in self.entries.values():
            if entry.is_known and entry.purity < self.purity_warning:
                msgs.append(
                    f"Cluster {entry.cluster_id} ({entry.name}) low purity "
                    f"{entry.purity:.2f} < {self.purity_warning:.2f}"
                )
        return msgs

    # ---- persistence ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict representation.

        NaN purity values are emitted as JSON ``null`` rather than the
        non-standard ``NaN`` literal, so the file stays parseable by every
        spec-conformant JSON reader (browsers, jq, most JS clients).
        """
        return {
            "schema_version": LABEL_MAP_SCHEMA_VERSION,
            "purity_warning": self.purity_warning,
            "entries": {str(cid): _entry_to_jsonable(e) for cid, e in self.entries.items()},
        }

    def save(self, path: Path | str) -> None:
        # ``allow_nan=False`` would *raise* on NaN; we instead replace NaN
        # ourselves in :func:`_entry_to_jsonable` and keep the default writer
        # so any future stray NaN surfaces loudly rather than producing
        # technically-invalid JSON.
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> ClusterLabelMap:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("schema_version", "0.0.0")
        if version != LABEL_MAP_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cluster_label_map schema_version {version!r}; "
                f"this build expects {LABEL_MAP_SCHEMA_VERSION!r}."
            )
        entries = {int(cid): _entry_from_jsonable(entry) for cid, entry in data["entries"].items()}
        return cls(entries=entries, purity_warning=float(data.get("purity_warning", 0.6)))


def _entry_to_jsonable(entry: ClusterLabelEntry) -> dict[str, Any]:
    """Convert a `ClusterLabelEntry` to a dict that ``json.dumps`` can handle.

    The only non-JSON-safe value the dataclass can carry is a NaN purity, so
    we map that to ``None`` (JSON ``null``) and pass everything else through.
    """
    payload = asdict(entry)
    purity = payload["purity"]
    if isinstance(purity, float) and math.isnan(purity):
        payload["purity"] = None
    return payload


def _entry_from_jsonable(payload: dict[str, Any]) -> ClusterLabelEntry:
    """Inverse of :func:`_entry_to_jsonable`: hydrate purity back to float-NaN."""
    if payload.get("purity") is None:
        payload = {**payload, "purity": float("nan")}
    return ClusterLabelEntry(**payload)
