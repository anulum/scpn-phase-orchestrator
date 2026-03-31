# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding validator tests

from __future__ import annotations

from dataclasses import replace

from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    BoundaryDef,
    OscillatorFamily,
)
from scpn_phase_orchestrator.binding.validator import validate_binding_spec


def test_valid_spec_no_errors(sample_binding_spec):
    errors = validate_binding_spec(sample_binding_spec)
    assert errors == []


def test_empty_name_error(sample_binding_spec):
    bad = replace(sample_binding_spec, name="")
    errors = validate_binding_spec(bad)
    assert any("name" in e for e in errors)


def test_bad_version_format(sample_binding_spec):
    bad = replace(sample_binding_spec, version="1.0")
    errors = validate_binding_spec(bad)
    assert any("version" in e for e in errors)

    bad2 = replace(sample_binding_spec, version="1.a.0")
    errors2 = validate_binding_spec(bad2)
    assert any("version" in e for e in errors2)


def test_invalid_safety_tier(sample_binding_spec):
    bad = replace(sample_binding_spec, safety_tier="military")
    errors = validate_binding_spec(bad)
    assert any("safety_tier" in e for e in errors)


def test_negative_sample_period(sample_binding_spec):
    bad = replace(sample_binding_spec, sample_period_s=-0.01)
    errors = validate_binding_spec(bad)
    assert any("sample_period_s" in e for e in errors)


def test_invalid_channel(sample_binding_spec):
    bad_families = {
        "x": OscillatorFamily(channel="X", extractor_type="hilbert", config={}),
    }
    bad = replace(sample_binding_spec, oscillator_families=bad_families)
    errors = validate_binding_spec(bad)
    assert any("channel" in e for e in errors)


def test_bad_actuator_knob(sample_binding_spec):
    bad_actuators = [
        ActuatorMapping(name="bad", knob="omega", scope="global", limits=(0.0, 1.0)),
    ]
    bad = replace(sample_binding_spec, actuators=bad_actuators)
    errors = validate_binding_spec(bad)
    assert any("knob" in e for e in errors)


def test_bad_boundary_severity(sample_binding_spec):
    bad_bounds = [
        BoundaryDef(name="b", variable="R", lower=0.0, upper=1.0, severity="extreme"),
    ]
    bad = replace(sample_binding_spec, boundaries=bad_bounds)
    errors = validate_binding_spec(bad)
    assert any("severity" in e for e in errors)


def test_empty_layers_error(sample_binding_spec):
    bad = replace(sample_binding_spec, layers=[])
    errors = validate_binding_spec(bad)
    assert any("at least one layer" in e for e in errors)


def test_objective_references_missing_layer(sample_binding_spec):
    from scpn_phase_orchestrator.binding.types import ObjectivePartition

    bad_obj = ObjectivePartition(good_layers=[0, 99], bad_layers=[])
    bad = replace(sample_binding_spec, objectives=bad_obj)
    errors = validate_binding_spec(bad)
    assert any("layer index 99" in e for e in errors)


def test_control_period_must_be_positive(sample_binding_spec):
    bad = replace(sample_binding_spec, control_period_s=0.0)
    errors = validate_binding_spec(bad)
    assert any("control_period_s" in e and "> 0" in e for e in errors)


def test_control_period_ge_sample_period(sample_binding_spec):
    bad = replace(sample_binding_spec, sample_period_s=0.1, control_period_s=0.05)
    errors = validate_binding_spec(bad)
    assert any("control_period_s must be >= sample_period_s" in e for e in errors)


def test_actuator_limits_ordering(sample_binding_spec):
    bad_actuators = [
        ActuatorMapping(name="inv", knob="K", scope="global", limits=(1.0, 0.0)),
    ]
    bad = replace(sample_binding_spec, actuators=bad_actuators)
    errors = validate_binding_spec(bad)
    assert any("limits" in e and "lo <= hi" in e for e in errors)


def test_empty_objectives_error(sample_binding_spec):
    from scpn_phase_orchestrator.binding.types import ObjectivePartition

    bad_obj = ObjectivePartition(good_layers=[], bad_layers=[])
    bad = replace(sample_binding_spec, objectives=bad_obj)
    errors = validate_binding_spec(bad)
    assert any("at least one good or bad layer" in e for e in errors)


# ---------------------------------------------------------------------------
# Discriminatory: valid values must NOT produce errors
# ---------------------------------------------------------------------------


def test_all_valid_safety_tiers_accepted(sample_binding_spec):
    """Every valid safety_tier must produce zero tier-related errors."""
    for tier in ["research", "clinical", "production", "consumer"]:
        spec = replace(sample_binding_spec, safety_tier=tier)
        errors = validate_binding_spec(spec)
        tier_errors = [e for e in errors if "safety_tier" in e]
        assert tier_errors == [], f"tier={tier!r} should be valid, got: {tier_errors}"


def test_valid_channels_accepted(sample_binding_spec):
    """Channels P, I, S must all pass validation."""
    for ch in ["P", "I", "S"]:
        families = {
            "f": OscillatorFamily(channel=ch, extractor_type="hilbert", config={}),
        }
        spec = replace(sample_binding_spec, oscillator_families=families)
        errors = validate_binding_spec(spec)
        ch_errors = [e for e in errors if "channel" in e]
        assert ch_errors == [], f"channel={ch!r} should be valid, got: {ch_errors}"


def test_valid_knobs_accepted(sample_binding_spec):
    """All valid knobs (K, alpha, zeta, Psi) must pass validation."""
    for knob in ["K", "alpha", "zeta", "Psi"]:
        actuators = [
            ActuatorMapping(
                name=f"{knob}_g", knob=knob, scope="global", limits=(0.0, 1.0)
            ),
        ]
        spec = replace(sample_binding_spec, actuators=actuators)
        errors = validate_binding_spec(spec)
        knob_errors = [e for e in errors if "knob" in e]
        assert knob_errors == [], f"knob={knob!r} should be valid, got: {knob_errors}"


def test_multiple_errors_all_reported(sample_binding_spec):
    """Multiple simultaneous violations must all be reported (not fail-fast)."""
    from scpn_phase_orchestrator.binding.types import ObjectivePartition

    bad = replace(
        sample_binding_spec,
        name="",
        version="bad",
        safety_tier="military",
        sample_period_s=-1.0,
        objectives=ObjectivePartition(good_layers=[], bad_layers=[]),
    )
    errors = validate_binding_spec(bad)
    assert len(errors) >= 4, f"Should report ≥4 errors, got {len(errors)}: {errors}"


def test_valid_version_formats(sample_binding_spec):
    """Semver versions with all-numeric parts must pass."""
    for v in ["0.0.1", "1.0.0", "10.20.30"]:
        spec = replace(sample_binding_spec, version=v)
        errors = validate_binding_spec(spec)
        ver_errors = [e for e in errors if "version" in e]
        assert ver_errors == [], f"version={v!r} should be valid, got: {ver_errors}"


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
