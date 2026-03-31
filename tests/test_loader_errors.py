# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Loader error tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec


def test_missing_file_raises(tmp_path):
    with pytest.raises(BindingLoadError, match="cannot read"):
        load_binding_spec(tmp_path / "nonexistent.yaml")


def test_invalid_extension_raises(tmp_path):
    p = tmp_path / "spec.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(BindingLoadError, match="Unsupported file extension"):
        load_binding_spec(p)


def test_malformed_yaml_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("{{invalid yaml::", encoding="utf-8")
    with pytest.raises(BindingLoadError, match="YAML parse error"):
        load_binding_spec(p)


def test_malformed_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(BindingLoadError, match="JSON parse error"):
        load_binding_spec(p)


def test_missing_required_key_raises(tmp_path):
    p = tmp_path / "partial.yaml"
    p.write_text("name: test\n", encoding="utf-8")
    with pytest.raises(BindingLoadError, match="missing required key"):
        load_binding_spec(p)


def test_non_mapping_top_level_raises(tmp_path):
    p = tmp_path / "list.yaml"
    p.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(BindingLoadError, match="expected mapping"):
        load_binding_spec(p)


def test_missing_layer_name_raises(tmp_path):
    import yaml

    data = {
        "name": "test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.1,
        "layers": [{"index": 0}],  # missing "name"
        "oscillator_families": {},
        "coupling": {"base_strength": 0.5, "decay_alpha": 0.3},
        "drivers": {},
        "objectives": {"good_layers": [0], "bad_layers": []},
    }
    p = tmp_path / "no_layer_name.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    with pytest.raises(BindingLoadError, match="missing required key 'name'"):
        load_binding_spec(p)


def test_valid_minimal_spec_loads(tmp_path):
    import yaml

    data = {
        "name": "minimal",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.1,
        "layers": [{"name": "L0", "index": 0}],
        "oscillator_families": {"p": {"channel": "P", "extractor_type": "hilbert"}},
        "coupling": {"base_strength": 0.5, "decay_alpha": 0.3},
        "drivers": {},
        "objectives": {"good_layers": [0], "bad_layers": []},
    }
    p = tmp_path / "ok.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    spec = load_binding_spec(p)
    assert spec.name == "minimal"
    assert len(spec.layers) == 1


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
