# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chemical Reactor morphogenetic field demo tests

from __future__ import annotations

import json

import numpy as np

from domainpacks.chemical_reactor.morphogenetic_field_demo import (
    chemical_reactor_stability_stress_phases,
    run_demo,
)


def _edge_deltas(audit: dict[str, object], key: str) -> dict[tuple[int, int], float]:
    return {
        (int(edge["source"]), int(edge["target"])): float(edge["delta"])
        for edge in audit[key]
    }


def _assert_top_edges_desc(snapshot: dict[str, object]) -> None:
    top_edges = snapshot["top_edges"]
    assert all(
        top_edges[idx]["weight"] >= top_edges[idx + 1]["weight"]
        for idx in range(len(top_edges) - 1)
    )


def test_chemical_reactor_morphogenetic_demo_emits_stability_audit() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "chemical_reactor"
    assert payload["scenario"] == "thermal_stability_stress_with_recovery_replay"
    assert payload["actuating"] is False
    assert payload["policy"]["coherence_target"] == 0.78

    audit = payload["audit"]
    grown = _edge_deltas(audit, "grown_edges")
    shrunk = _edge_deltas(audit, "shrunk_edges")

    assert audit["global_coherence"] < payload["policy"]["coherence_target"]
    assert audit["delta_norm"] > 0.0
    assert audit["field"]["shape"] == [4, 4]
    assert grown[(0, 1)] > 0.0
    assert grown[(2, 3)] > 0.0
    assert shrunk[(0, 2)] < 0.0
    assert shrunk[(1, 3)] < 0.0

    snapshot = payload["snapshot"]
    assert snapshot["shape"] == [4, 4]
    assert len(snapshot["heatmap_rows"]) == 4
    assert len(snapshot["top_edges"]) == 6
    assert snapshot["top_edges"][0]["source"] == 0
    assert snapshot["top_edges"][0]["target"] == 1
    _assert_top_edges_desc(snapshot)
    assert snapshot["top_edges"][0]["weight"] >= snapshot["top_edges"][-1]["weight"]
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
