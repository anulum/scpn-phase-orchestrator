# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

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
