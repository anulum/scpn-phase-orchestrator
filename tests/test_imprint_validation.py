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


class TestImprintValidationPipelineWiring:
    """Pipeline: validated imprint spec → ImprintModel construction."""

    def test_validated_imprint_builds_model(self):
        """Valid imprint spec → ImprintModel: validates config drives
        model construction without errors."""
        from scpn_phase_orchestrator.imprint.update import ImprintModel

        spec = _base_spec(
            imprint_model=ImprintSpec(decay_rate=0.05, saturation=2.0, modulates=[]),
        )
        errors = validate_binding_spec(spec)
        assert errors == []

        model = ImprintModel(
            spec.imprint_model.decay_rate,
            spec.imprint_model.saturation,
        )
        assert model._decay_rate == 0.05
        assert model._saturation == 2.0
