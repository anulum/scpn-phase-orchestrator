# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Shared test fixtures

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

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
from scpn_phase_orchestrator.upde.engine import UPDEEngine

settings.register_profile(
    "ci", max_examples=500, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("dev", max_examples=50)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))

TWO_PI = 2.0 * np.pi


@pytest.fixture
def sample_phases():
    rng = np.random.default_rng(42)
    return rng.uniform(0, TWO_PI, size=8)


@pytest.fixture
def sample_knm():
    rng = np.random.default_rng(99)
    raw = rng.uniform(0, 0.5, size=(8, 8))
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


@pytest.fixture
def sample_omegas():
    rng = np.random.default_rng(7)
    return rng.uniform(0.5, 2.0, size=8)


@pytest.fixture
def sample_binding_spec():
    layers = [
        HierarchyLayer(name="L1", index=0, oscillator_ids=["osc0", "osc1"]),
        HierarchyLayer(name="L2", index=1, oscillator_ids=["osc2", "osc3"]),
    ]
    families = {
        "phys": OscillatorFamily(channel="P", extractor_type="hilbert", config={}),
        "info": OscillatorFamily(channel="I", extractor_type="event", config={}),
    }
    coupling = CouplingSpec(base_strength=0.45, decay_alpha=0.3, templates={})
    drivers = DriverSpec(physical={"freq": 10.0}, informational={}, symbolic={})
    objectives = ObjectivePartition(good_layers=[0], bad_layers=[1])
    boundaries = [
        BoundaryDef(
            name="R_floor", variable="R", lower=0.2, upper=None, severity="hard"
        ),
    ]
    actuators = [
        ActuatorMapping(name="K_global", knob="K", scope="global", limits=(0.0, 1.0)),
        ActuatorMapping(
            name="zeta_global", knob="zeta", scope="global", limits=(0.0, 0.5)
        ),
    ]
    return BindingSpec(
        name="test-domain",
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


@pytest.fixture
def upde_engine():
    return UPDEEngine(n_oscillators=8, dt=0.01)
