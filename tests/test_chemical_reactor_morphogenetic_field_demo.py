# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chemical Reactor morphogenetic field demo tests

from __future__ import annotations

import json
from numbers import Integral, Real

import numpy as np

from domainpacks.chemical_reactor.morphogenetic_field_demo import (
    chemical_reactor_stability_stress_phases,
    run_demo,
)


def _json_object(value: object) -> dict[str, object]:
    """Return a demo payload block narrowed to a JSON object."""
    assert isinstance(value, dict)
    return value


def _json_list(value: object) -> list[object]:
    """Return a demo payload block narrowed to a JSON list."""
    assert isinstance(value, list)
    return value


def _json_objects(value: object) -> list[dict[str, object]]:
    """Return a demo payload list narrowed to a list of JSON objects."""
    return [_json_object(item) for item in _json_list(value)]


def _integer(value: object) -> int:
    """Return a demo payload index narrowed to an integer."""
    assert isinstance(value, Integral) and not isinstance(value, bool)
    return int(value)


def _real(value: object) -> float:
    """Return a demo payload metric narrowed to a real number."""
    assert isinstance(value, Real) and not isinstance(value, bool)
    return float(value)


def _edge_deltas(audit: dict[str, object], key: str) -> dict[tuple[int, int], float]:
    return {
        (_integer(edge["source"]), _integer(edge["target"])): _real(edge["delta"])
        for edge in _json_objects(audit[key])
    }


def _assert_top_edges_desc(snapshot: dict[str, object]) -> None:
    top_edges = _json_objects(snapshot["top_edges"])
    assert all(
        _real(top_edges[idx]["weight"]) >= _real(top_edges[idx + 1]["weight"])
        for idx in range(len(top_edges) - 1)
    )


def test_chemical_reactor_morphogenetic_demo_emits_stability_audit() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "chemical_reactor"
    assert payload["scenario"] == "thermal_stability_stress_with_recovery_replay"
    assert payload["actuating"] is False
    policy = _json_object(payload["policy"])
    assert policy["coherence_target"] == 0.78

    audit = _json_object(payload["audit"])
    grown = _edge_deltas(audit, "grown_edges")
    shrunk = _edge_deltas(audit, "shrunk_edges")

    assert _real(audit["global_coherence"]) < _real(policy["coherence_target"])
    assert _real(audit["delta_norm"]) > 0.0
    assert _json_object(audit["field"])["shape"] == [4, 4]
    assert grown[(0, 1)] > 0.0
    assert grown[(2, 3)] > 0.0
    assert shrunk[(0, 2)] < 0.0
    assert shrunk[(1, 3)] < 0.0

    snapshot = _json_object(payload["snapshot"])
    assert snapshot["shape"] == [4, 4]
    assert len(_json_list(snapshot["heatmap_rows"])) == 4
    top_edges = _json_objects(snapshot["top_edges"])
    assert len(top_edges) == 6
    assert top_edges[0]["source"] == 0
    assert top_edges[0]["target"] == 1
    _assert_top_edges_desc(snapshot)
    assert _real(top_edges[0]["weight"]) >= _real(top_edges[-1]["weight"])
    assert json.loads(json.dumps(payload, sort_keys=True))["domainpack"] == (
        "chemical_reactor"
    )


def test_chemical_reactor_stability_stress_phases_are_deterministic() -> None:
    phases = chemical_reactor_stability_stress_phases()

    assert phases.shape == (4,)
    assert np.all(np.isfinite(phases))
    assert phases[1] - phases[0] < 0.1
    assert phases[2] - phases[0] > 2.0
    assert phases[3] - phases[2] < 0.1
    assert phases[2] - phases[1] > 2.0
    assert phases[3] - phases[1] > 2.0
