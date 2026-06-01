# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Visualization API reference documentation tests

from __future__ import annotations

from pathlib import Path

VISUALIZATION_REFERENCE = Path("docs/reference/api/visualization.md")


def test_visualization_api_reference_meets_depth_baseline() -> None:
    doc = VISUALIZATION_REFERENCE.read_text(encoding="utf-8")

    assert len(doc.splitlines()) >= 567


def test_visualization_api_reference_documents_presentation_contracts() -> None:
    doc = VISUALIZATION_REFERENCE.read_text(encoding="utf-8")
    required_phrases = (
        "VisualizerStreamer",
        "network_graph_json",
        "coupling_heatmap_json",
        "torus_points_json",
        "phase_wheel_json",
        "presentation-only",
        "JSON encoders",
        "finite JSON payload boundaries",
        "boolean aliases",
        "complex aliases",
        "graph threshold exclusivity",
        "torus coordinate equations",
        "phase-wheel unit-circle equations",
        "outbound-only streamer semantics",
        "UPDEEngine",
        "typed-array annotation contracts",
    )

    for phrase in required_phrases:
        assert phrase in doc
