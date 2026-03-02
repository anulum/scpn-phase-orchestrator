# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

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
    with pytest.raises((FileNotFoundError, OSError)):
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
