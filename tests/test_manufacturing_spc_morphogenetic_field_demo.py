# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Manufacturing SPC morphogenetic field demo tests

from __future__ import annotations

import json

import numpy as np
import pytest

from domainpacks.manufacturing_spc.morphogenetic_field_demo import (
    manufacturing_stress_demo_phases,
    run_demo,
)


@pytest.fixture
def phases() -> np.ndarray:
    return manufacturing_stress_demo_phases()


def _edge_deltas(audit: dict[str, object], key: str) -> dict[tuple[int, int], float]:
    return {
        (int(edge["source"]), int(edge["target"])): float(edge["delta"])
        for edge in audit[key]
    }


def _sorted_snapshot_edges(
    snapshot: list[dict[str, object]],
) -> list[tuple[float, int, int]]:
    return [
        (float(edge["weight"]), int(edge["source"]), int(edge["target"]))
        for edge in sorted(
            snapshot,
            key=lambda edge: (
                -float(edge["weight"]),
                int(edge["source"]),
                int(edge["target"]),
            ),
        )
    ]


def _snapshot_tuples(snapshot: list[dict[str, object]]) -> list[tuple[float, int, int]]:
    return [
        (float(edge["weight"]), int(edge["source"]), int(edge["target"]))
        for edge in snapshot
    ]


def test_manufacturing_morphogenetic_demo_uses_deterministic_phases(
    phases: np.ndarray,
) -> None:
    np.testing.assert_allclose(phases, np.array([0.0, 0.0, np.pi], dtype=np.float64))
    assert phases.shape == (3,)
    assert np.all(np.isfinite(phases))


def test_manufacturing_morphogenetic_demo_emits_non_actuating_audit(
    phases: np.ndarray,
) -> None:
    payload = run_demo()

    assert payload["domainpack"] == "manufacturing_spc"
    assert payload["scenario"] == "tool_wear_pressure_spike_recovery"
    assert payload["actuating"] is False

    policy = payload["policy"]
    assert policy["coherence_target"] == 0.74
    assert policy["max_delta"] > 0.0

    audit = payload["audit"]
    grown = _edge_deltas(audit, "grown_edges")
    shrunk = _edge_deltas(audit, "shrunk_edges")

    assert audit["global_coherence"] < policy["coherence_target"]
    assert audit["delta_norm"] > 0.0
    assert audit["field"]["shape"] == [3, 3]
    assert grown[(0, 1)] > 0.0
    assert grown[(1, 0)] > 0.0
    assert shrunk[(0, 2)] < 0.0
    assert shrunk[(2, 0)] < 0.0

    snapshot = payload["snapshot"]
    assert snapshot["shape"] == [3, 3]
    assert len(snapshot["top_edges"]) == 6
    assert len(snapshot["heatmap_rows"]) == 3
    assert snapshot["top_edges"][0]["source"] == 0
    assert snapshot["top_edges"][0]["target"] == 1

    snapshot_edges = _snapshot_tuples(snapshot["top_edges"])
    assert _sorted_snapshot_edges(snapshot["top_edges"]) == snapshot_edges

    # JSON round-trip remains audit-safe and preserves core topology labels.
    restored = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    assert restored["domainpack"] == "manufacturing_spc"
    assert isinstance(restored, dict)
