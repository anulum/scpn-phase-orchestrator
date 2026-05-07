# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plasma-control morphogenetic field demo tests

from __future__ import annotations

import json

import numpy as np

from domainpacks.plasma_control.morphogenetic_field_demo import (
    plasma_edge_localised_demo_phases,
    run_demo,
)


def _edge_deltas(audit: dict[str, object], key: str) -> dict[tuple[int, int], float]:
    return {
        (int(edge["source"]), int(edge["target"])): float(edge["delta"])
        for edge in audit[key]
    }


def test_plasma_morphogenetic_demo_emits_edge_localised_audit() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "plasma_control"
    assert payload["scenario"] == "edge_localised_transport_barrier_stress"
    assert payload["actuating"] is False
    assert payload["policy"]["coherence_target"] == 0.78

    audit = payload["audit"]
    grown = _edge_deltas(audit, "grown_edges")
    shrunk = _edge_deltas(audit, "shrunk_edges")

    assert audit["global_coherence"] < payload["policy"]["coherence_target"]
    assert audit["delta_norm"] > 0.0
    assert audit["field"]["shape"] == [8, 8]
    assert grown[(4, 5)] > 0.0
    assert grown[(5, 6)] > 0.0
    assert grown[(2, 3)] > 0.0
    assert shrunk[(0, 1)] < 0.0
    assert shrunk[(2, 4)] < 0.0

    snapshot = payload["snapshot"]
    assert snapshot["shape"] == [8, 8]
    assert len(snapshot["heatmap_rows"]) == 8
    assert len(snapshot["top_edges"]) == 10
    assert snapshot["top_edges"][0]["weight"] >= snapshot["top_edges"][-1]["weight"]
    assert json.loads(json.dumps(payload, sort_keys=True))["domainpack"] == (
        "plasma_control"
    )


def test_plasma_edge_localised_phases_encode_transport_barrier_stress() -> None:
    phases = plasma_edge_localised_demo_phases()

    assert phases.shape == (8,)
    assert np.all(np.isfinite(phases))
    assert phases[5] - phases[4] < 0.04
    assert phases[6] - phases[5] < 0.04
    assert phases[3] - phases[2] < 0.3
    assert phases[0] - phases[1] > 2.0
    assert phases[2] - phases[4] > 2.0
