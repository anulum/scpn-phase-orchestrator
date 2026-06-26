# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio morphogenetic panel tests

"""Studio facade contract tests for the morphogenetic-field review panel."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.studio as studio
from scpn_phase_orchestrator.supervisor import (
    MorphogeneticFieldState,
    render_morphogenetic_field_svg,
)


def _artifact() -> dict[str, object]:
    """Return a production morphogenetic field SVG audit artifact."""
    field = np.array(
        [
            [0.0, 0.9, 0.2],
            [0.4, 0.0, 0.7],
            [0.1, 0.3, 0.0],
        ],
        dtype=np.float64,
    )
    return render_morphogenetic_field_svg(
        MorphogeneticFieldState(field),
        top_k=3,
        cell_size=16,
        title="Studio field review",
    ).to_audit_record()


def _copy_mapping(payload: dict[str, object]) -> dict[str, object]:
    """Return a mutable JSON-like mapping copy."""
    return cast("dict[str, object]", deepcopy(payload))


def _snapshot(artifact: dict[str, object]) -> dict[str, object]:
    """Return the nested snapshot mapping with strict test-time typing."""
    return cast("dict[str, object]", artifact["snapshot"])


def test_morphogenetic_panel_renders_svg_snapshot_evidence() -> None:
    """The public Studio facade renders passive morphogenetic-field evidence."""
    artifact = _artifact()

    panel = studio.build_morphogenetic_field_studio_panel(artifact)

    assert panel["panel_kind"] == "studio_morphogenetic_field_panel"
    assert panel["renderer"] == "morphogenetic_field_svg"
    assert panel["format"] == "svg"
    assert panel["actuation_permitted"] is False
    assert panel["shape"] == [3, 3]
    assert panel["top_edge_count"] == 3
    assert panel["strongest_edge"]["source"] == 0
    assert panel["strongest_edge"]["target"] == 1
    assert panel["field_energy"]["maximum"] == pytest.approx(0.9)
    assert "actions_to_apply" not in panel
    assert "control_actions" not in panel
    decoded_panel = json.loads(json.dumps(panel, allow_nan=False))
    assert decoded_panel["panel_kind"] == panel["panel_kind"]


def test_morphogenetic_panel_renders_empty_top_edge_review_state() -> None:
    """Top-edge evidence is optional and renders as an empty review state."""
    artifact = _copy_mapping(_artifact())
    snapshot = _snapshot(artifact)
    snapshot["top_edges"] = None

    panel = studio.build_morphogenetic_field_studio_panel(artifact)

    assert panel["top_edge_count"] == 0
    assert panel["strongest_edge"] == {}
    assert panel["snapshot"]["top_edges"] == ()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (42, "artifact must be a mapping"),
        ({"format": "png"}, "format must be svg"),
        ({"snapshot": "bad"}, "snapshot must be a mapping"),
    ],
)
def test_morphogenetic_panel_rejects_malformed_artifact_shape(
    mutation: object,
    match: str,
) -> None:
    """Top-level artifact validation rejects malformed review payloads."""
    artifact = _copy_mapping(_artifact())
    if isinstance(mutation, dict):
        artifact.update(mutation)
    else:
        artifact = cast("dict[str, object]", mutation)

    with pytest.raises(ValueError, match=match):
        studio.build_morphogenetic_field_studio_panel(artifact)


def test_morphogenetic_panel_rejects_svg_script_content() -> None:
    """Complete SVG documents with script content fail closed."""
    artifact = _copy_mapping(_artifact())
    artifact["svg"] = '<svg viewBox="0 0 1 1"><script /></svg>'

    with pytest.raises(ValueError, match="script"):
        studio.build_morphogenetic_field_studio_panel(artifact)


def test_morphogenetic_panel_rejects_incomplete_svg_document() -> None:
    """SVG review artifacts must be complete documents."""
    artifact = _copy_mapping(_artifact())
    artifact["svg"] = "<svg></svg>"

    with pytest.raises(ValueError, match="complete SVG"):
        studio.build_morphogenetic_field_studio_panel(artifact)


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("minimum", 0.8, "minimum <= mean <= maximum"),
        ("shape", "bad", "two-item integer sequence"),
        ("shape", [3], "two-item integer sequence"),
        ("shape", [2, 3], "shape must be square"),
        ("heatmap_rows", "bad", "sequence of strings"),
        ("heatmap_rows", ["abc"], "length must match shape"),
        ("heatmap_rows", ["abc", "", "abc"], "non-empty strings"),
        ("heatmap_rows", ["ab", "ab", "ab"], "row width"),
        ("top_edges", "bad", "top_edges must be a sequence"),
    ],
)
def test_morphogenetic_panel_rejects_malformed_snapshot_fields(
    field_name: str,
    bad_value: object,
    match: str,
) -> None:
    """Snapshot shape, statistics, heatmap, and edge validation fail closed."""
    artifact = _copy_mapping(_artifact())
    snapshot = _snapshot(artifact)
    snapshot[field_name] = bad_value

    with pytest.raises(ValueError, match=match):
        studio.build_morphogenetic_field_studio_panel(artifact)


@pytest.mark.parametrize(
    ("top_edges", "match"),
    [
        ([42], "entries must be mappings"),
        ([{"source": 0, "target": 0, "weight": 0.8}], "off-diagonal"),
        (
            [
                {"source": 0, "target": 1, "weight": 0.2},
                {"source": 1, "target": 2, "weight": 0.8},
            ],
            "sorted",
        ),
    ],
)
def test_morphogenetic_panel_rejects_malformed_top_edge_rows(
    top_edges: object,
    match: str,
) -> None:
    """Top-edge row validation rejects malformed or unsorted evidence."""
    artifact = _copy_mapping(_artifact())
    snapshot = _snapshot(artifact)
    snapshot["top_edges"] = top_edges

    with pytest.raises(ValueError, match=match):
        studio.build_morphogenetic_field_studio_panel(artifact)
