# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network-security morphogenetic field demo tests

from __future__ import annotations

import json

import numpy as np

from domainpacks.network_security.morphogenetic_field_demo import (
    network_security_morphogenetic_phases,
    run_demo,
)


def _edge_deltas(audit: dict[str, object], key: str) -> dict[tuple[int, int], float]:
    return {
        (int(edge["source"]), int(edge["target"])): float(edge["delta"])
        for edge in audit[key]
    }


def test_network_security_morphogenetic_demo_emits_non_actuating_audit() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "network_security"
    assert payload["scenario"] == "lateral_movement_defence_field_replay"
    assert payload["actuating"] is False
    assert payload["policy"]["coherence_target"] == 0.76

    audit = payload["audit"]
    grown = _edge_deltas(audit, "grown_edges")
    shrunk = _edge_deltas(audit, "shrunk_edges")

    assert audit["global_coherence"] < payload["policy"]["coherence_target"]
    assert audit["delta_norm"] > 0.0
    assert audit["field"]["shape"] == [3, 3]
    assert grown[(0, 2)] > 0.0
    assert grown[(2, 0)] > 0.0
    assert shrunk[(0, 1)] < 0.0
    assert shrunk[(1, 0)] < 0.0
    assert shrunk[(1, 2)] < 0.0
    assert shrunk[(2, 1)] < 0.0

    snapshot = payload["snapshot"]
    assert snapshot["shape"] == [3, 3]
    assert len(snapshot["heatmap_rows"]) == 3
    assert len(snapshot["top_edges"]) == 6
    assert snapshot["top_edges"][0]["weight"] >= snapshot["top_edges"][-1]["weight"]
    assert json.loads(json.dumps(payload, sort_keys=True))["domainpack"] == (
        "network_security"
    )


def test_network_security_morphogenetic_phases_encode_attack_misalignment() -> None:
    phases = network_security_morphogenetic_phases()

    assert phases.shape == (3,)
    assert np.all(np.isfinite(phases))
    assert phases[2] - phases[0] < 0.1
    assert phases[1] - phases[0] > 2.0
    assert phases[1] - phases[2] > 2.0
