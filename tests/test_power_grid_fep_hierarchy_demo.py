# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power-grid FEP hierarchy demo tests

from __future__ import annotations

import numpy as np

from domainpacks.power_grid.fep_hierarchy_demo import (
    power_grid_hierarchy_state,
    run_demo,
)


def test_power_grid_fep_hierarchy_demo_emits_child_and_parent_records() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "power_grid"
    assert payload["hierarchy"] == "two_child_regions_to_parent_fep_supervisor"
    children = payload["children"]
    assert len(children) == 2
    assert {child["name"] for child in children} == {
        "generation_area",
        "demand_renewable_area",
    }
    for child in children:
        assessment = child["assessment"]
        assert 0.0 <= assessment["observed_R"] <= 1.0
        assert np.isfinite(assessment["free_energy"])
        assert child["actions"]
    parent = payload["parent"]
    assert 0.0 <= parent["assessment"]["observed_R"] <= 1.0
    assert parent["actions"]


def test_power_grid_hierarchy_state_has_distinct_child_coherence() -> None:
    states = power_grid_hierarchy_state()

    assert set(states) == {"generation_area", "demand_renewable_area"}
    generation_phases, generation_omegas = states["generation_area"]
    demand_phases, demand_omegas = states["demand_renewable_area"]
    assert generation_phases.shape == generation_omegas.shape
    assert demand_phases.shape == demand_omegas.shape
    assert generation_phases.size == 5
    assert demand_phases.size == 5
    assert np.std(generation_phases) < np.std(demand_phases)
