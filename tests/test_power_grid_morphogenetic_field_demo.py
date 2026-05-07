# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power-grid morphogenetic field demo tests

from __future__ import annotations

import json

import numpy as np

from domainpacks.power_grid.morphogenetic_field_demo import (
    grid_stress_demo_phases,
    run_demo,
)


def _edge_deltas(audit: dict[str, object], key: str) -> dict[tuple[int, int], float]:
    return {
        (int(edge["source"]), int(edge["target"])): float(edge["delta"])
        for edge in audit[key]
    }


def test_power_grid_morphogenetic_demo_emits_non_actuating_audit() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "power_grid"
    assert payload["scenario"] == "generation_area_stable_load_renewable_stress"
    assert payload["actuating"] is False
    assert payload["policy"]["coherence_target"] == 0.82

    audit = payload["audit"]
    grown = _edge_deltas(audit, "grown_edges")
    shrunk = _edge_deltas(audit, "shrunk_edges")

    assert audit["global_coherence"] < payload["policy"]["coherence_target"]
    assert audit["delta_norm"] > 0.0
    assert audit["field"]["shape"] == [5, 5]
    assert grown[(0, 1)] > 0.0
    assert grown[(1, 0)] > 0.0
    assert shrunk[(0, 3)] < 0.0
    assert shrunk[(1, 4)] < 0.0

    snapshot = payload["snapshot"]
    assert snapshot["shape"] == [5, 5]
    assert len(snapshot["heatmap_rows"]) == 5
    assert len(snapshot["top_edges"]) == 6
    assert snapshot["top_edges"][0]["weight"] >= snapshot["top_edges"][-1]["weight"]
    assert json.loads(json.dumps(payload, sort_keys=True))["domainpack"] == (
        "power_grid"
    )


def test_grid_stress_demo_phases_encode_stable_generation_and_stressed_loads() -> None:
    phases = grid_stress_demo_phases()

    assert phases.shape == (5,)
    assert np.all(np.isfinite(phases))
    assert phases[1] - phases[0] < 0.03
    assert phases[2] - phases[1] > 1.0
    assert phases[3] - phases[2] > 1.0
    assert phases[4] - phases[3] < 0.2
