# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power-grid causal attribution demo tests

from __future__ import annotations

import numpy as np
import pytest

from domainpacks.power_grid.causal_attribution_demo import (
    power_grid_disturbance_state,
    run_demo,
)


def test_power_grid_causal_demo_emits_stabilising_attribution() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "power_grid"
    assert payload["scenario"] == "load_step_governor_counterfactual"
    attribution = payload["attribution"]
    assert isinstance(attribution, dict)
    assert attribution["effect"] == "stabilising"
    assert attribution["confidence"] == pytest.approx(1.0)
    assert attribution["delta_R_final"] > 0.0
    assert attribution["delta_R_mean"] > 0.0
    counterfactual = payload["counterfactual"]
    assert counterfactual["actions"][0]["knob"] == "K"
    assert counterfactual["actions"][0]["scope"] == "global"
    assert len(counterfactual["baseline_R"]) == 41
    assert len(counterfactual["intervention_R"]) == 41


def test_power_grid_disturbance_state_shapes_and_values() -> None:
    phases, omegas = power_grid_disturbance_state()

    assert phases.shape == (12,)
    assert omegas.shape == (12,)
    assert np.all(np.isfinite(phases))
    assert np.all(np.isfinite(omegas))
    assert float(np.max(phases) - np.min(phases)) > np.pi
