# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cardiac FEP hierarchy demo tests

from __future__ import annotations

import numpy as np

from domainpacks.cardiac_rhythm.fep_hierarchy_demo import (
    cardiac_hierarchy_state,
    run_demo,
)


def test_cardiac_fep_hierarchy_demo_emits_child_and_parent_records() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "cardiac_rhythm"
    assert payload["hierarchy"] == "cardiac_child_axes_to_parent_fep_supervisor"
    children = payload["children"]
    assert {child["name"] for child in children} == {
        "pacemaker_atrial_axis",
        "ventricular_recovery_axis",
    }
    for child in children:
        assessment = child["assessment"]
        assert 0.0 <= assessment["observed_R"] <= 1.0
        assert np.isfinite(assessment["free_energy"])
        assert child["actions"]
    parent = payload["parent"]
    assert 0.0 <= parent["assessment"]["observed_R"] <= 1.0
    assert parent["actions"]
    assert len(payload["child_R_values"]) == 2
    assert len(payload["parent_phase_encoding"]) == 2


def test_cardiac_hierarchy_state_has_distinct_child_axes() -> None:
    states = cardiac_hierarchy_state()

    assert set(states) == {"pacemaker_atrial_axis", "ventricular_recovery_axis"}
    pacemaker_phases, pacemaker_omegas = states["pacemaker_atrial_axis"]
    ventricular_phases, ventricular_omegas = states["ventricular_recovery_axis"]
    assert pacemaker_phases.shape == pacemaker_omegas.shape
    assert ventricular_phases.shape == ventricular_omegas.shape
    assert pacemaker_phases.size == 5
    assert ventricular_phases.size == 5
    assert np.mean(pacemaker_omegas) > np.mean(ventricular_omegas)
