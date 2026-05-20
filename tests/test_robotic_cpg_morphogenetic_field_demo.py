# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Robotic CPG morphogenetic field demo tests

from __future__ import annotations

import json

import numpy as np
import pytest

from domainpacks.robotic_cpg.morphogenetic_field_demo import (
    robotic_cpg_morphogenetic_demo_phases,
    run_demo,
)


def _edge_deltas(audit: dict[str, object], key: str) -> dict[tuple[int, int], float]:
    return {
        (int(edge["source"]), int(edge["target"])): float(edge["delta"])
        for edge in audit[key]
    }


@pytest.fixture
def phase_fixture() -> np.ndarray:
    return robotic_cpg_morphogenetic_demo_phases()


def test_robotic_cpg_morphogenetic_demo_emits_non_actuating_audit() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "robotic_cpg"
    assert payload["scenario"] == "quadrupedal_gait_phase_field_replay"
    assert payload["actuating"] is False
    assert payload["policy"]["coherence_target"] == 0.80

    audit = payload["audit"]
    grown = _edge_deltas(audit, "grown_edges")
    shrunk = _edge_deltas(audit, "shrunk_edges")

    assert audit["global_coherence"] < payload["policy"]["coherence_target"]
    assert audit["delta_norm"] > 0.0
    assert audit["field"]["shape"] == [4, 4]
    assert grown[(0, 2)] > 0.0
    assert grown[(2, 0)] > 0.0
    assert shrunk[(0, 3)] < 0.0
    assert shrunk[(3, 0)] < 0.0

    snapshot = payload["snapshot"]
    assert snapshot["shape"] == [4, 4]
    assert len(snapshot["heatmap_rows"]) == 4
    assert len(snapshot["top_edges"]) == 6
    assert snapshot["top_edges"][0]["weight"] >= snapshot["top_edges"][-1]["weight"]
    restored = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    assert restored["domainpack"] == "robotic_cpg"


def test_robotic_cpg_morphogenetic_demo_phase_fixture_has_finite_gait_offsets(
    phase_fixture: np.ndarray,
) -> None:
    assert phase_fixture.shape == (4,)
    assert np.all(np.isfinite(phase_fixture))
    assert phase_fixture[0] == phase_fixture[1]
    assert phase_fixture[1] == phase_fixture[2]
    assert phase_fixture[3] != phase_fixture[0]
