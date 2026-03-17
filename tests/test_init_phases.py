# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for P/I/S initial phase extraction

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    BindingSpec,
    BoundaryDef,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ObjectivePartition,
    OscillatorFamily,
)
from scpn_phase_orchestrator.oscillators.init_phases import extract_initial_phases

TWO_PI = 2.0 * np.pi


def _make_pis_spec() -> BindingSpec:
    """BindingSpec with all three channel types (P, I, S)."""
    layers = [
        HierarchyLayer(name="L1", index=0, oscillator_ids=["osc0", "osc1", "osc2"]),
    ]
    families = {
        "phys": OscillatorFamily(channel="P", extractor_type="hilbert", config={}),
        "info": OscillatorFamily(channel="I", extractor_type="event", config={}),
        "symb": OscillatorFamily(
            channel="S", extractor_type="ring", config={"n_states": 5}
        ),
    }
    coupling = CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={})
    drivers = DriverSpec(physical={"freq": 10.0}, informational={}, symbolic={})
    objectives = ObjectivePartition(good_layers=[0], bad_layers=[])
    boundaries = [
        BoundaryDef(
            name="R_floor", variable="R", lower=0.2, upper=None, severity="hard"
        ),
    ]
    actuators = [
        ActuatorMapping(name="K_global", knob="K", scope="global", limits=(0.0, 1.0)),
    ]
    return BindingSpec(
        name="test-pis",
        version="1.0.0",
        safety_tier="research",
        sample_period_s=0.01,
        control_period_s=0.1,
        layers=layers,
        oscillator_families=families,
        coupling=coupling,
        drivers=drivers,
        objectives=objectives,
        boundaries=boundaries,
        actuators=actuators,
    )


def test_output_shape_matches_omegas():
    spec = _make_pis_spec()
    omegas = np.array([1.0, 2.0, 3.0])
    phases = extract_initial_phases(spec, omegas, seed=42)
    assert phases.shape == (3,)


def test_phases_in_valid_range():
    spec = _make_pis_spec()
    omegas = np.array([1.0, 2.0, 3.0])
    phases = extract_initial_phases(spec, omegas, seed=42)
    assert np.all(phases >= 0.0)
    assert np.all(phases < TWO_PI)


def test_deterministic_with_same_seed():
    spec = _make_pis_spec()
    omegas = np.array([1.0, 2.0, 3.0])
    a = extract_initial_phases(spec, omegas, seed=99)
    b = extract_initial_phases(spec, omegas, seed=99)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_differ():
    spec = _make_pis_spec()
    omegas = np.array([1.0, 2.0, 3.0])
    a = extract_initial_phases(spec, omegas, seed=1)
    b = extract_initial_phases(spec, omegas, seed=2)
    assert not np.array_equal(a, b)
