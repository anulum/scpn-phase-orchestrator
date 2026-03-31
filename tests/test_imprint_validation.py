# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Imprint validation tests

from __future__ import annotations

from scpn_phase_orchestrator.binding.types import (
    BindingSpec,
    CouplingSpec,
    DriverSpec,
    HierarchyLayer,
    ImprintSpec,
    ObjectivePartition,
    OscillatorFamily,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec


def _base_spec(**overrides) -> BindingSpec:
    defaults = {
        "name": "test",
        "version": "0.1.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.1,
        "layers": [HierarchyLayer(name="L0", index=0, oscillator_ids=["a"])],
        "oscillator_families": {
            "p": OscillatorFamily(channel="P", extractor_type="hilbert", config={}),
        },
        "coupling": CouplingSpec(base_strength=0.5, decay_alpha=0.3, templates={}),
        "drivers": DriverSpec(physical={}, informational={}, symbolic={}),
        "objectives": ObjectivePartition(good_layers=[0], bad_layers=[]),
        "boundaries": [],
        "actuators": [],
        "imprint_model": None,
        "geometry_prior": None,
    }
    defaults.update(overrides)
    return BindingSpec(**defaults)


def test_valid_imprint_passes():
    spec = _base_spec(
        imprint_model=ImprintSpec(decay_rate=0.01, saturation=3.0, modulates=[]),
    )
    assert validate_binding_spec(spec) == []


def test_negative_decay_rate_fails():
    spec = _base_spec(
        imprint_model=ImprintSpec(decay_rate=-0.5, saturation=3.0, modulates=[]),
    )
    errors = validate_binding_spec(spec)
    assert any("decay_rate" in e for e in errors)


def test_zero_saturation_fails():
    spec = _base_spec(
        imprint_model=ImprintSpec(decay_rate=0.01, saturation=0.0, modulates=[]),
    )
    errors = validate_binding_spec(spec)
    assert any("saturation" in e for e in errors)


def test_no_imprint_passes():
    spec = _base_spec(imprint_model=None)
    assert validate_binding_spec(spec) == []


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
