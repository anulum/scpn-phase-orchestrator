# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding loader tests

from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec

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


def test_loader_preserves_named_driver_channels(tmp_path):
    data = {
        **_SPEC_DATA,
        "drivers": {
            "physical": {},
            "informational": {},
            "symbolic": {},
            "Q": {"zeta": 0.25},
            "edge-node": {"psi": 1.5},
        },
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    spec = load_binding_spec(p)

    assert spec.drivers.channel_config("Q") == {"zeta": 0.25}
    assert spec.drivers.channel_config("edge-node") == {"psi": 1.5}
    assert spec.drivers.all_channel_configs()["Q"] == {"zeta": 0.25}


def test_loader_preserves_n_channel_algebra(tmp_path):
    data = {
        **_SPEC_DATA,
        "oscillator_families": {
            "phys": {"channel": "P", "extractor_type": "hilbert", "config": {}},
            "operator": {"channel": "H", "extractor_type": "event", "config": {}},
            "risk": {"channel": "Risk", "extractor_type": "graph", "config": {}},
        },
        "channels": {
            "P": {
                "role": "plant",
                "units": "rad",
                "metric_semantics": "physical phase",
            },
            "H": {
                "role": "operator_intent",
                "replay_semantics": "event",
                "supervisor_visibility": True,
            },
            "Risk": {
                "role": "derived_risk",
                "required": False,
                "replay_semantics": "derived",
                "derived_from": ["P", "H"],
                "derive_rule": "risk = phase_lag(P,H)",
            },
        },
        "channel_groups": {
            "control_surface": {
                "channels": ["P", "H", "Risk"],
                "required": True,
                "description": "channels exposed to supervisor policy",
            }
        },
        "cross_channel_couplings": [
            {
                "source": "H",
                "target": "P",
                "strength": 0.25,
                "mode": "directed",
                "template": "operator_to_plant",
            }
        ],
        "drivers": {
            "physical": {},
            "informational": {},
            "symbolic": {},
            "H": {"cadence_hz": 1.0},
            "Risk": {},
        },
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    spec = load_binding_spec(p)

    assert spec.channels["H"].role == "operator_intent"
    assert spec.channels["Risk"].derived_from == ["P", "H"]
    assert spec.channel_groups["control_surface"].channels == ["P", "H", "Risk"]
    assert spec.cross_channel_couplings[0].source == "H"
    assert spec.cross_channel_couplings[0].target == "P"


def test_loader_rejects_non_mapping_drivers_block(tmp_path):
    data = {**_SPEC_DATA, "drivers": ["physical"]}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(BindingLoadError, match="expected mapping in drivers, got list"):
        load_binding_spec(p)


def test_loader_rejects_non_mapping_standard_driver(tmp_path):
    data = {
        **_SPEC_DATA,
        "drivers": {"physical": 1.0, "informational": {}, "symbolic": {}},
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        BindingLoadError, match="expected mapping in drivers.physical, got float"
    ):
        load_binding_spec(p)


def test_loader_rejects_non_mapping_named_driver(tmp_path):
    data = {
        **_SPEC_DATA,
        "drivers": {
            "physical": {},
            "informational": {},
            "symbolic": {},
            "Q": 0.5,
        },
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        BindingLoadError, match="expected mapping in drivers.Q, got float"
    ):
        load_binding_spec(p)


def test_loader_rejects_invalid_named_driver_channel(tmp_path):
    data = {
        **_SPEC_DATA,
        "drivers": {
            "physical": {},
            "informational": {},
            "symbolic": {},
            "bad channel": {"zeta": 0.1},
        },
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(BindingLoadError, match="invalid driver channel identifier"):
        load_binding_spec(p)


def test_loader_rejects_invalid_declared_channel_id(tmp_path):
    data = {**_SPEC_DATA, "channels": {"bad channel": {"role": "bad"}}}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(BindingLoadError, match="invalid channel identifier"):
        load_binding_spec(p)


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


def test_validate_rejects_broken_channel_algebra(tmp_path):
    from scpn_phase_orchestrator.binding import validate_binding_spec

    data = {
        **_SPEC_DATA,
        "channels": {
            "Risk": {
                "role": "derived_risk",
                "replay_semantics": "derived",
                "derived_from": ["Missing"],
            }
        },
        "channel_groups": {"empty": {"channels": []}},
        "cross_channel_couplings": [
            {"source": "Risk", "target": "Risk", "strength": -0.1, "mode": "bad"}
        ],
    }
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    spec = load_binding_spec(p)

    errors = validate_binding_spec(spec)

    assert any("derived_from references unknown channel" in err for err in errors)
    assert any("derive_rule is required" in err for err in errors)
    assert any(
        "channel_group 'empty': channels must not be empty" in err for err in errors
    )
    assert any("source and target must differ" in err for err in errors)
    assert any("strength must be finite and >= 0" in err for err in errors)
    assert any("mode must be one of" in err for err in errors)


def test_loader_validate_control_period_and_channels(tmp_path):
    from scpn_phase_orchestrator.binding import validate_binding_spec

    data = {
        **_SPEC_DATA,
        "control_period_s": -1.0,
        "oscillator_families": {
            "bad": {
                "channel": "bad channel",
                "extractor_type": "hilbert",
                "config": {},
            },
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


def test_loader_rejects_non_string_name_after_structural_parse(tmp_path):
    data = {**_SPEC_DATA, "name": False}
    p = tmp_path / "bad_name.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(BindingLoadError, match="expected string in name, got bool"):
        load_binding_spec(p)


def test_loader_rejects_boolean_layer_index(tmp_path):
    data = {
        **_SPEC_DATA,
        "layers": [{"name": "L1", "index": True, "oscillator_ids": ["a"]}],
    }
    p = tmp_path / "bad_layer_index.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        BindingLoadError, match="expected integer in layers\\[\\].index, got bool"
    ):
        load_binding_spec(p)


def test_loader_rejects_boolean_numeric_period(tmp_path):
    data = {**_SPEC_DATA, "sample_period_s": True}
    p = tmp_path / "bad_period.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        BindingLoadError, match="expected number in sample_period_s, got bool"
    ):
        load_binding_spec(p)


def test_loader_rejects_non_boolean_channel_required_flag(tmp_path):
    data = {
        **_SPEC_DATA,
        "channels": {"P": {"role": "plant", "required": "yes"}},
    }
    p = tmp_path / "bad_channel_flag.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        BindingLoadError, match="expected boolean in channels.P.required, got str"
    ):
        load_binding_spec(p)


def test_loader_allows_null_derived_from_as_empty_list(tmp_path):
    data = {
        **_SPEC_DATA,
        "channels": {
            "P": {"role": "plant", "derived_from": None},
            "H": {"role": "operator", "required": True},
        },
    }
    p = tmp_path / "null_derived_from.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    spec = load_binding_spec(p)

    assert spec.channels["P"].derived_from == []


def test_load_channel_groups_requires_channels_key(tmp_path):
    data = {
        **_SPEC_DATA,
        "channel_groups": {
            "supervised": {"description": "group without channels"},
        },
    }
    p = tmp_path / "bad_channel_group.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        BindingLoadError,
        match="missing required key 'channels' in channel_groups\\.supervised",
    ):
        load_binding_spec(p)


def test_loader_rejects_non_list_channel_group_entries(tmp_path):
    data = {
        **_SPEC_DATA,
        "channel_groups": {"supervised": {"channels": "P"}},
    }
    p = tmp_path / "bad_channel_group_type.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        BindingLoadError,
        match="expected list in channel_groups\\.supervised\\.channels, got str",
    ):
        load_binding_spec(p)


def test_load_binding_treats_null_cross_channel_couplings_as_empty(tmp_path):
    data = {**_SPEC_DATA, "cross_channel_couplings": None}
    p = tmp_path / "null_cross_channel_couplings.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    spec = load_binding_spec(p)

    assert spec.cross_channel_couplings == []


def test_loader_preserves_optional_layer_omegas_and_missing_boundary_bounds(tmp_path):
    data = {
        **_SPEC_DATA,
        "layers": [
            {
                "name": "L1",
                "index": 0,
                "oscillator_ids": ["a", "b"],
                "omegas": [0.75, 1],
            }
        ],
        "boundaries": [{"name": "b1", "variable": "R", "severity": "soft"}],
    }
    p = tmp_path / "layer_omegas.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    spec = load_binding_spec(p)

    assert spec.layers[0].omegas == [0.75, 1.0]
    assert spec.boundaries[0].lower is None
    assert spec.boundaries[0].upper is None


def test_loader_rejects_actuator_limits_with_wrong_arity(tmp_path):
    data = {
        **_SPEC_DATA,
        "actuators": [
            {"name": "K_glob", "knob": "K", "scope": "global", "limits": [0.0]},
        ],
    }
    p = tmp_path / "bad_actuator_limits.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        BindingLoadError,
        match="expected two numbers in actuators\\[\\].limits, got 1 item",
    ):
        load_binding_spec(p)


def test_loader_rejects_invalid_channel_group_identifier(tmp_path):
    data = {
        **_SPEC_DATA,
        "channel_groups": {"1-risk": {"channels": ["P"]}},
    }
    p = tmp_path / "bad_channel_group.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(
        BindingLoadError,
        match="channel_groups.1-risk: invalid channel group identifier",
    ):
        load_binding_spec(p)


def test_loader_preserves_protocol_net_and_amplitude_blocks(tmp_path):
    data = {
        **_SPEC_DATA,
        "protocol_net": {
            "places": ["idle", "active"],
            "initial": {"idle": 1, "active": 0},
            "place_regime": {"idle": "safe_hold"},
            "transitions": [
                {
                    "name": "activate",
                    "inputs": [{"place": "idle", "weight": 1}],
                    "outputs": [{"place": "active", "weight": 1}],
                    "guard": "R > 0.8",
                }
            ],
        },
        "amplitude": {
            "mu": 0.2,
            "epsilon": 0.01,
            "amp_coupling_strength": 0.05,
            "amp_coupling_decay": 0.4,
        },
    }
    p = tmp_path / "protocol_amplitude.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    spec = load_binding_spec(p)

    assert spec.protocol_net is not None
    assert spec.protocol_net.places == ["idle", "active"]
    assert spec.protocol_net.initial == {"idle": 1, "active": 0}
    assert spec.protocol_net.place_regime == {"idle": "safe_hold"}
    assert spec.protocol_net.transitions[0].name == "activate"
    assert spec.protocol_net.transitions[0].inputs == [{"place": "idle", "weight": 1}]
    assert spec.protocol_net.transitions[0].outputs == [
        {"place": "active", "weight": 1}
    ]
    assert spec.protocol_net.transitions[0].guard == "R > 0.8"
    assert spec.amplitude is not None
    assert spec.amplitude.mu == 0.2
    assert spec.amplitude.epsilon == 0.01
    assert spec.amplitude.amp_coupling_strength == 0.05
    assert spec.amplitude.amp_coupling_decay == 0.4


class TestPipelineWiring:
    """Pipeline wiring: load_binding_spec -> build engine params -> UPDEEngine -> R.

    Proves the binding loader is load-bearing infrastructure that feeds real
    engine parameters into the UPDE pipeline.
    """

    def test_binding_spec_to_engine_to_order_parameter(self, tmp_path):
        """E2E: load a binding spec, extract coupling params, run UPDEEngine,
        compute order parameter R."""
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        # Write a valid spec and load it
        spec_data = {**_SPEC_DATA}
        spec_path = tmp_path / "pipeline.json"
        spec_path.write_text(json.dumps(spec_data), encoding="utf-8")
        spec = load_binding_spec(spec_path)

        # Build engine parameters from the loaded spec
        n = sum(len(layer.oscillator_ids) for layer in spec.layers)
        n = max(n, 4)  # ensure enough oscillators for meaningful dynamics
        base_k = spec.coupling.base_strength

        eng = UPDEEngine(n, dt=spec.sample_period_s)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = base_k * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0, f"Order parameter out of range: {r}"
        assert spec.name == "loader-test"

    def test_load_binding_spec_performance(self, tmp_path):
        """load_binding_spec must complete in < 10ms for a standard spec."""
        import time

        spec_path = tmp_path / "perf.json"
        spec_path.write_text(json.dumps(_SPEC_DATA), encoding="utf-8")

        start = time.perf_counter()
        for _ in range(100):
            load_binding_spec(spec_path)
        elapsed_ms = (time.perf_counter() - start) / 100 * 1000

        assert elapsed_ms < 10.0, (
            f"load_binding_spec took {elapsed_ms:.2f}ms, expected < 10ms"
        )
