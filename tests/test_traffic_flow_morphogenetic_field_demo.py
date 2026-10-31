# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Traffic-flow morphogenetic field demo tests

from __future__ import annotations

import json

import numpy as np

from domainpacks.traffic_flow.morphogenetic_field_demo import (
    run_demo,
    traffic_spillback_demo_phases,
)


def _mapping(value: object) -> dict[str, object]:
    """Narrow a JSON-safe payload node to a string-keyed mapping."""
    assert isinstance(value, dict)
    return value


def _sequence(value: object) -> list[object]:
    """Narrow a JSON-safe payload node to a list."""
    assert isinstance(value, list)
    return value


def _int(value: object) -> int:
    """Narrow a JSON-safe payload node to an integer."""
    assert isinstance(value, int)
    return int(value)


def _real(value: object) -> float:
    """Narrow a JSON-safe payload node to a real number."""
    assert isinstance(value, (int, float))
    return float(value)


def _edge_deltas(audit: dict[str, object], key: str) -> dict[tuple[int, int], float]:
    return {
        (_int(edge["source"]), _int(edge["target"])): _real(edge["delta"])
        for edge in map(_mapping, _sequence(audit[key]))
    }


def test_traffic_flow_morphogenetic_demo_emits_spillback_audit() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "traffic_flow"
    assert payload["scenario"] == "corridor_equity_aligned_spillback_stress"
    assert payload["actuating"] is False
    assert _mapping(payload["policy"])["coherence_target"] == 0.80

    audit = _mapping(payload["audit"])
    grown = _edge_deltas(audit, "grown_edges")
    shrunk = _edge_deltas(audit, "shrunk_edges")

    assert _real(audit["global_coherence"]) < _real(
        _mapping(payload["policy"])["coherence_target"]
    )
    assert _real(audit["delta_norm"]) > 0.0
    assert _mapping(audit["field"])["shape"] == [6, 6]
    assert grown[(1, 2)] > 0.0
    assert grown[(1, 5)] > 0.0
    assert grown[(3, 4)] > 0.0
    assert shrunk[(0, 1)] < 0.0
    assert shrunk[(2, 4)] < 0.0

    snapshot = _mapping(payload["snapshot"])
    assert snapshot["shape"] == [6, 6]
    assert len(_sequence(snapshot["heatmap_rows"])) == 6
    assert len(_sequence(snapshot["top_edges"])) == 8
    top_edges = [_mapping(edge) for edge in _sequence(snapshot["top_edges"])]
    assert _real(top_edges[0]["weight"]) >= _real(top_edges[-1]["weight"])
    assert json.loads(json.dumps(payload, sort_keys=True))["domainpack"] == (
        "traffic_flow"
    )


def test_traffic_spillback_phases_encode_aligned_corridor_and_stress() -> None:
    phases = traffic_spillback_demo_phases()

    assert phases.shape == (6,)
    assert np.all(np.isfinite(phases))
    assert phases[2] - phases[1] < 0.1
    assert phases[5] - phases[1] < 0.05
    assert phases[3] - phases[0] < 0.2
    assert phases[4] - phases[3] < 0.3
    assert phases[0] - phases[1] > 2.0
