# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Traffic topology adaptation demo tests

from __future__ import annotations

import numpy as np

from domainpacks.traffic_flow.topology_adaptation_demo import (
    run_demo,
    topology_lyapunov_validation,
    traffic_demo_phase_history,
    traffic_demo_phases,
    transfer_entropy_supported_knm,
)
from scpn_phase_orchestrator.supervisor import (
    HigherOrderTopologySupervisor,
    TopologyMutationPolicy,
)


def test_traffic_topology_demo_adds_te_supported_simplices() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "traffic_flow"
    assert payload["transfer_entropy"]["max"] > 0.0
    assert payload["transfer_entropy"]["support_edges"] > 0
    audit = payload["audit"]
    assert isinstance(audit, dict)
    assert audit["global_coherence"] < payload["policy"]["coherence_floor"]
    assert audit["pairwise_delta_norm"] > 0.0
    assert audit["added_simplices"]
    assert audit["hyperedge_count"] >= len(audit["added_simplices"])
    lyapunov = payload["lyapunov_validation"]
    assert lyapunov["non_increasing"] is True
    assert lyapunov["delta_V"] < 0.0


def test_transfer_entropy_support_knm_is_symmetric_and_bounded() -> None:
    history = traffic_demo_phase_history()
    knm, te = transfer_entropy_supported_knm(history)

    assert history.shape == (6, 80)
    assert te.shape == (6, 6)
    assert knm.shape == (6, 6)
    assert np.allclose(knm, knm.T)
    assert np.all(np.diag(knm) == 0.0)
    assert float(np.max(knm)) <= 0.20
    assert float(np.max(te)) > 0.0


def test_topology_lyapunov_validation_reports_energy_delta() -> None:
    history = traffic_demo_phase_history()
    knm, _te = transfer_entropy_supported_knm(history)
    phases = traffic_demo_phases()
    result = HigherOrderTopologySupervisor(
        TopologyMutationPolicy(
            mutation_rate=0.40,
            coherence_floor=0.70,
            pairwise_threshold=0.85,
            simplex_threshold=0.995,
            max_new_simplices=2,
            max_simplex_strength=0.22,
            simplex_pairwise_support_floor=0.10,
        )
    ).mutate(phases, knm)

    validation = topology_lyapunov_validation(phases, knm, result.knm)

    assert validation["before_V"] > validation["after_V"]
    assert validation["non_increasing"] is True
    assert isinstance(validation["after_in_basin"], bool)
    assert validation["after_max_phase_diff"] >= 0.0


def test_traffic_phase_history_rejects_too_few_steps() -> None:
    try:
        traffic_demo_phase_history(steps=7)
    except ValueError as exc:
        assert "steps" in str(exc)
    else:
        raise AssertionError("expected ValueError for too-short history")
