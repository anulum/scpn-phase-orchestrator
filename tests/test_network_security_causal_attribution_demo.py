# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network-security causal attribution demo tests

from __future__ import annotations

import numpy as np
import pytest

from domainpacks.network_security.causal_attribution_demo import (
    lateral_movement_disturbance_state,
    run_demo,
)


def test_network_security_causal_demo_emits_stabilising_attribution() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "network_security"
    assert payload["scenario"] == "lateral_movement_firewall_coupling_counterfactual"
    assert payload["actuating"] is False
    attribution = payload["attribution"]
    assert isinstance(attribution, dict)
    assert attribution["effect"] == "stabilising"
    # Trajectory-consistency: every step after the shared initial state (delta==0
    # at t0) holds the stabilising sign, so 35 of 36 steps agree.
    assert attribution["trajectory_consistency"] == pytest.approx(35 / 36)
    assert attribution["delta_R_final"] > 0.0
    assert attribution["delta_R_mean"] > 0.0

    counterfactual = payload["counterfactual"]
    assert counterfactual["actions"][0]["knob"] == "K"
    assert counterfactual["actions"][0]["scope"] == "global"
    assert len(counterfactual["baseline_R"]) == 36
    assert len(counterfactual["intervention_R"]) == 36
    assert counterfactual["intervention_R"][-1] > counterfactual["baseline_R"][-1]


def test_lateral_movement_state_shapes_and_attack_stress_values() -> None:
    phases, omegas = lateral_movement_disturbance_state()

    assert phases.shape == (8,)
    assert omegas.shape == (8,)
    assert np.all(np.isfinite(phases))
    assert np.all(np.isfinite(omegas))
    assert float(np.max(phases) - np.min(phases)) > np.pi
    assert omegas[3] > omegas[0]
    assert omegas[4] > omegas[1]
    assert omegas[5] < omegas[0]
