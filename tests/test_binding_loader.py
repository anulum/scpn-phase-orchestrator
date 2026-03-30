# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding loader tests

from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.binding.loader import load_binding_spec

_SPEC_DATA = {
    "name": "loader-test",
    "version": "0.2.0",
    "safety_tier": "research",
    "sample_period_s": 0.01,
    "control_period_s": 0.1,
    "layers": [
        {"name": "L1", "index": 0, "oscillator_ids": ["a"]},
    ],
    "oscillator_families": {
        "phys": {"channel": "P", "extractor_type": "hilbert", "config": {}},
    },
    "coupling": {"base_strength": 0.5, "decay_alpha": 0.3},
    "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
    "objectives": {"good_layers": [0], "bad_layers": []},
    "boundaries": [
        {"name": "b1", "variable": "R", "lower": 0.1, "upper": 1.0, "severity": "soft"},
    ],
    "actuators": [
        {"name": "K_glob", "knob": "K", "scope": "global", "limits": [0.0, 1.0]},
    ],
}


def test_load_json(tmp_path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_SPEC_DATA), encoding="utf-8")
    spec = load_binding_spec(p)
    assert spec.name == "loader-test"
    assert spec.version == "0.2.0"
    assert len(spec.layers) == 1
    assert spec.layers[0].index == 0
    assert spec.coupling.base_strength == 0.5
    assert len(spec.actuators) == 1


def test_load_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.dump(_SPEC_DATA), encoding="utf-8")
    spec = load_binding_spec(p)
    assert spec.name == "loader-test"
    assert spec.safety_tier == "research"
    assert spec.objectives.good_layers == [0]


def test_nonexistent_file_raises():
    from scpn_phase_orchestrator.binding.loader import BindingLoadError

    with pytest.raises(BindingLoadError, match="cannot read"):
        load_binding_spec("/nonexistent/path/spec.json")


def test_unsupported_extension(tmp_path):
    p = tmp_path / "spec.txt"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_binding_spec(p)


def test_load_with_imprint_model(tmp_path):
    data = {**_SPEC_DATA, "imprint_model": {"decay_rate": 0.1, "saturation": 0.9}}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    spec = load_binding_spec(p)
    assert spec.imprint_model is not None
    assert spec.imprint_model.decay_rate == 0.1
    assert spec.imprint_model.saturation == 0.9
    assert spec.imprint_model.modulates == []


def test_load_with_geometry_prior(tmp_path):
    geo = {"constraint_type": "spherical", "params": {"radius": 1.0}}
    data = {**_SPEC_DATA, "geometry_prior": geo}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    spec = load_binding_spec(p)
    assert spec.geometry_prior is not None
    assert spec.geometry_prior.constraint_type == "spherical"
    assert spec.geometry_prior.params == {"radius": 1.0}


def test_loader_resolves_extractor_aliases(tmp_path):
    """Aliases (physical/informational/symbolic) resolve to algorithm names."""
    data = {
        **_SPEC_DATA,
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "physical", "config": {}},
            "i": {"channel": "I", "extractor_type": "informational", "config": {}},
            "s": {"channel": "S", "extractor_type": "symbolic", "config": {}},
        },
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    spec = load_binding_spec(p)
    assert spec.oscillator_families["p"].extractor_type == "hilbert"
    assert spec.oscillator_families["i"].extractor_type == "event"
    assert spec.oscillator_families["s"].extractor_type == "ring"


def test_loader_passes_algorithm_names_through(tmp_path):
    """Algorithm names (hilbert/event/graph) pass through unchanged."""
    data = {
        **_SPEC_DATA,
        "oscillator_families": {
            "h": {"channel": "P", "extractor_type": "hilbert", "config": {}},
            "g": {"channel": "S", "extractor_type": "graph", "config": {}},
        },
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    spec = load_binding_spec(p)
    assert spec.oscillator_families["h"].extractor_type == "hilbert"
    assert spec.oscillator_families["g"].extractor_type == "graph"


def test_json_yaml_produce_identical_spec(tmp_path):
    """Loading the same spec from JSON and YAML must produce identical BindingSpec."""
    yaml = pytest.importorskip("yaml")
    pj = tmp_path / "spec.json"
    py = tmp_path / "spec.yaml"
    pj.write_text(json.dumps(_SPEC_DATA), encoding="utf-8")
    py.write_text(yaml.dump(_SPEC_DATA), encoding="utf-8")
    spec_j = load_binding_spec(pj)
    spec_y = load_binding_spec(py)
    assert spec_j.name == spec_y.name
    assert spec_j.version == spec_y.version
    assert spec_j.safety_tier == spec_y.safety_tier
    assert len(spec_j.layers) == len(spec_y.layers)
    assert spec_j.coupling.base_strength == spec_y.coupling.base_strength


def test_boundaries_preserved(tmp_path):
    """Boundary definitions must survive load with correct field values."""
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_SPEC_DATA), encoding="utf-8")
    spec = load_binding_spec(p)
    assert len(spec.boundaries) == 1
    b = spec.boundaries[0]
    assert b.name == "b1"
    assert b.variable == "R"
    assert b.lower == 0.1
    assert b.upper == 1.0
    assert b.severity == "soft"


def test_actuator_limits_tuple(tmp_path):
    """Actuator limits (from JSON list) must be accessible as (lo, hi) tuple."""
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_SPEC_DATA), encoding="utf-8")
    spec = load_binding_spec(p)
    assert spec.actuators[0].limits == (0.0, 1.0)


def test_loaded_spec_passes_validation(tmp_path):
    """A well-formed spec must pass validation after loading."""
    from scpn_phase_orchestrator.binding import validate_binding_spec

    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_SPEC_DATA), encoding="utf-8")
    spec = load_binding_spec(p)
    errors = validate_binding_spec(spec)
    assert errors == [], f"Valid spec should produce no errors: {errors}"


def test_loader_validate_control_period_and_channels(tmp_path):
    from scpn_phase_orchestrator.binding import validate_binding_spec

    data = {
        **_SPEC_DATA,
        "control_period_s": -1.0,
        "oscillator_families": {
            "bad": {"channel": "X", "extractor_type": "hilbert", "config": {}},
        },
        "boundaries": [
            {
                "name": "b",
                "variable": "R",
                "lower": 0.0,
                "upper": 1.0,
                "severity": "extreme",
            },
        ],
        "actuators": [
            {"name": "a", "knob": "bogus", "scope": "global", "limits": [1.0, 0.0]},
        ],
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    spec = load_binding_spec(p)
    errors = validate_binding_spec(spec)
    assert any("control_period_s" in e for e in errors)
    assert any("channel" in e for e in errors)
    assert any("severity" in e for e in errors)
    assert any("knob" in e for e in errors)
    assert any("limits" in e for e in errors)
