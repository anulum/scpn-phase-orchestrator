# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio morphogenetic-field review panel

"""Morphogenetic-field review panel builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ._shared import (
    _non_negative_float,
    _non_negative_int,
    _positive_int,
    _require_non_empty_text,
    _unit_interval_number,
)


def build_morphogenetic_field_studio_panel(
    svg_artifact: Mapping[str, object],
) -> dict[str, object]:
    """Return a Studio panel payload for morphogenetic field SVG artefacts.

    The helper renders already-computed topology-field evidence only. It
    validates the dependency-free SVG artefact, preserves the snapshot
    statistics and strongest off-diagonal field edges, and keeps actuation
    disabled for operator review.

    Parameters
    ----------
    svg_artifact : Mapping[str, object]
        The morphogenetic field SVG artefact.

    Returns
    -------
    dict[str, object]
        A Studio panel payload for morphogenetic field SVG artefacts.
    """
    record = _normalise_morphogenetic_field_svg_artifact(svg_artifact)
    snapshot = cast("dict[str, object]", record["snapshot"])
    top_edges = cast("tuple[dict[str, object], ...]", snapshot["top_edges"])
    strongest_edge = top_edges[0] if top_edges else {}
    return {
        "panel_kind": "studio_morphogenetic_field_panel",
        "renderer": "morphogenetic_field_svg",
        "format": "svg",
        "width": record["width"],
        "height": record["height"],
        "shape": snapshot["shape"],
        "snapshot": snapshot,
        "top_edge_count": len(top_edges),
        "strongest_edge": strongest_edge,
        "field_energy": {
            "mean": snapshot["mean"],
            "minimum": snapshot["minimum"],
            "maximum": snapshot["maximum"],
            "l2_norm": snapshot["l2_norm"],
        },
        "svg": record["svg"],
        "actuation_permitted": False,
        "operator_action": (
            "render as passive topology-field evidence; review strongest "
            "off-diagonal edges before any downstream policy action"
        ),
    }


def _normalise_morphogenetic_field_svg_artifact(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(artifact, Mapping):
        raise ValueError("morphogenetic SVG artifact must be a mapping")
    if artifact.get("format") != "svg":
        raise ValueError("format must be svg")
    width = _positive_int(artifact.get("width"), "width", minimum=1)
    height = _positive_int(artifact.get("height"), "height", minimum=1)
    svg = _require_review_svg(artifact.get("svg"))
    snapshot_raw = artifact.get("snapshot")
    if not isinstance(snapshot_raw, Mapping):
        raise ValueError("snapshot must be a mapping")
    snapshot = _normalise_morphogenetic_field_snapshot(snapshot_raw)
    return {
        "format": "svg",
        "width": width,
        "height": height,
        "snapshot": snapshot,
        "svg": svg,
    }


def _normalise_morphogenetic_field_snapshot(
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    shape = _normalise_matrix_shape(snapshot.get("shape"))
    mean = _unit_interval_number(snapshot.get("mean"), "mean")
    minimum = _unit_interval_number(snapshot.get("minimum"), "minimum")
    maximum = _unit_interval_number(snapshot.get("maximum"), "maximum")
    if minimum > mean + 1e-12 or mean > maximum + 1e-12:
        raise ValueError("snapshot statistics must satisfy minimum <= mean <= maximum")
    heatmap_rows = _normalise_heatmap_rows(
        snapshot.get("heatmap_rows"),
        expected_rows=shape[0],
        expected_columns=shape[1],
    )
    top_edges = _normalise_morphogenetic_top_edges(
        snapshot.get("top_edges"),
        shape=shape,
    )
    return {
        "shape": list(shape),
        "mean": mean,
        "minimum": minimum,
        "maximum": maximum,
        "l2_norm": _non_negative_float(snapshot.get("l2_norm"), "l2_norm"),
        "heatmap_rows": list(heatmap_rows),
        "top_edges": top_edges,
    }


def _normalise_matrix_shape(value: object) -> tuple[int, int]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("shape must be a two-item integer sequence")
    if len(value) != 2:
        raise ValueError("shape must be a two-item integer sequence")
    rows = _positive_int(value[0], "shape", minimum=1)
    columns = _positive_int(value[1], "shape", minimum=1)
    if rows != columns:
        raise ValueError("shape must be square")
    return (rows, columns)


def _normalise_heatmap_rows(
    value: object,
    *,
    expected_rows: int,
    expected_columns: int,
) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("heatmap_rows must be a sequence of strings")
    if len(value) != expected_rows:
        raise ValueError("heatmap_rows length must match shape")
    rows: list[str] = []
    for row in value:
        if not isinstance(row, str) or row == "":
            raise ValueError("heatmap_rows must contain non-empty strings")
        row_text = row
        if len(row_text) != expected_columns:
            raise ValueError("heatmap row width must match shape")
        rows.append(row_text)
    return tuple(rows)


def _normalise_morphogenetic_top_edges(
    value: object,
    *,
    shape: tuple[int, int],
) -> tuple[dict[str, object], ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("top_edges must be a sequence")
    edges: list[dict[str, object]] = []
    previous_weight: float | None = None
    for edge in value:
        if not isinstance(edge, Mapping):
            raise ValueError("top_edges entries must be mappings")
        source = _non_negative_int(edge.get("source"), "top_edges source")
        target = _non_negative_int(edge.get("target"), "top_edges target")
        if source >= shape[0] or target >= shape[1] or source == target:
            raise ValueError("top_edges must reference off-diagonal field edges")
        weight = _unit_interval_number(edge.get("weight"), "top_edges weight")
        if previous_weight is not None and weight > previous_weight + 1e-12:
            raise ValueError("top_edges must be sorted by descending weight")
        previous_weight = weight
        edges.append({"source": source, "target": target, "weight": weight})
    return tuple(edges)


def _require_review_svg(value: object) -> str:
    svg = _require_non_empty_text(value, "svg").strip()
    if not svg.startswith("<svg ") or not svg.endswith("</svg>"):
        raise ValueError("svg must be a complete SVG document")
    if "<script" in svg.lower():
        raise ValueError("svg must not contain script content")
    return svg
