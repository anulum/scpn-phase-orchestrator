# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plasma topology adaptation demo tests

from __future__ import annotations

import numpy as np

from domainpacks.plasma_control.topology_adaptation_demo import run_demo
from scpn_phase_orchestrator.supervisor import (
    HigherOrderTopologySupervisor,
    TopologyMutationPolicy,
)


def test_plasma_topology_demo_adds_auditable_guarded_simplices() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "plasma_control"
    audit = payload["audit"]
    assert isinstance(audit, dict)
    assert audit["hyperedge_count"] >= 1
    assert audit["pairwise_delta_norm"] > 0.0
    assert audit["global_coherence"] < payload["policy"]["coherence_floor"]
    assert audit["added_simplices"]
    assert all(edge["strength"] <= 0.25 for edge in audit["added_simplices"])
    lyapunov = payload["lyapunov_validation"]
    assert lyapunov["non_increasing"] is True
    assert lyapunov["delta_V"] < 0.0


def test_simplex_pairwise_support_floor_blocks_unsupported_triads() -> None:
    phases = np.array([0.0, 0.02, 0.04, np.pi, np.pi + 0.02, np.pi + 0.04])
    knm = np.zeros((6, 6), dtype=np.float64)
    knm[:3, :3] = 0.2
    np.fill_diagonal(knm, 0.0)
    knm[1, 2] = 0.01
    knm[2, 1] = 0.01
    supervisor = HigherOrderTopologySupervisor(
        TopologyMutationPolicy(
            mutation_rate=0.5,
            coherence_floor=0.9,
            simplex_threshold=0.99,
            max_new_simplices=4,
            simplex_pairwise_support_floor=0.1,
        )
    )

    result = supervisor.mutate(phases, knm)

    assert all(edge.nodes != (0, 1, 2) for edge in result.added_simplices)
