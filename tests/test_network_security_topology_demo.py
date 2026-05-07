# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network-security topology demo tests

from __future__ import annotations

import numpy as np
import pytest

from domainpacks.network_security.topology_adaptation_demo import (
    network_security_phase_history,
    network_security_phase_snapshot,
    network_security_te_supported_knm,
    run_demo,
    topology_lyapunov_validation,
)
from scpn_phase_orchestrator.supervisor import (
    HigherOrderTopologySupervisor,
    TopologyMutationPolicy,
)


def test_network_security_topology_demo_adds_te_supported_simplices() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "network_security"
    assert payload["scenario"] == "lateral_movement_te_supported_topology_mutation"
    assert payload["actuating"] is False
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


def test_network_security_te_support_knm_is_symmetric_and_bounded() -> None:
    history = network_security_phase_history()
    knm, te = network_security_te_supported_knm(history)

    assert history.shape == (8, 80)
    assert te.shape == (8, 8)
    assert knm.shape == (8, 8)
    assert np.allclose(knm, knm.T)
    assert np.all(np.diag(knm) == 0.0)
    assert float(np.max(knm)) <= 0.20
    assert float(np.max(te)) > 0.0


def test_network_security_topology_lyapunov_validation_reports_energy_delta() -> None:
    history = network_security_phase_history()
    knm, _te = network_security_te_supported_knm(history)
    phases = network_security_phase_snapshot()
    result = HigherOrderTopologySupervisor(
        TopologyMutationPolicy(
            mutation_rate=0.35,
            coherence_floor=0.78,
            pairwise_threshold=0.85,
            simplex_threshold=0.995,
            max_new_simplices=3,
            max_simplex_strength=0.22,
            simplex_pairwise_support_floor=0.10,
        )
    ).mutate(phases, knm)

    validation = topology_lyapunov_validation(phases, knm, result.knm)

    assert validation["before_V"] > validation["after_V"]
    assert validation["delta_V"] == pytest.approx(
        validation["after_V"] - validation["before_V"]
    )
    assert validation["non_increasing"] is True
    assert isinstance(validation["after_in_basin"], bool)
    assert validation["after_max_phase_diff"] >= 0.0


def test_network_security_phase_history_rejects_too_few_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        network_security_phase_history(steps=7)
