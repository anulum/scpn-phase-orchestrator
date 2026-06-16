# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power grid domainpack tests

from __future__ import annotations

from pathlib import Path

import pytest

from domainpacks.power_grid import run as power_grid_run
from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec

SPEC_PATH = Path("domainpacks/power_grid/binding_spec.yaml")
EXPECTED_LAYERS = [
    "generator_rotor",
    "area_frequency",
    "tie_line",
    "load_demand",
    "renewable_intermittency",
]


@pytest.fixture(scope="module")
def spec():
    return load_binding_spec(SPEC_PATH)


class TestSpec:
    def test_validates_without_errors(self, spec) -> None:
        assert validate_binding_spec(spec) == []

    def test_name(self, spec) -> None:
        assert spec.name == "power_grid"

    def test_layer_names_match_documented_structure(self, spec) -> None:
        assert [layer.name for layer in spec.layers] == EXPECTED_LAYERS

    def test_layer_indices_are_contiguous(self, spec) -> None:
        assert [layer.index for layer in spec.layers] == list(range(len(spec.layers)))

    def test_every_layer_has_oscillators(self, spec) -> None:
        assert all(len(layer.oscillator_ids) >= 1 for layer in spec.layers)

    def test_omegas_count_matches_oscillator_total(self, spec) -> None:
        total = sum(len(layer.oscillator_ids) for layer in spec.layers)
        assert len(spec.get_omegas()) == total

    def test_objectives_reference_valid_layers(self, spec) -> None:
        valid = {layer.index for layer in spec.layers}
        assert set(spec.objectives.good_layers) <= valid
        assert set(spec.objectives.bad_layers) <= valid
        assert not set(spec.objectives.good_layers) & set(spec.objectives.bad_layers)

    def test_boundaries_and_actuators_present(self, spec) -> None:
        assert len(spec.boundaries) >= 1
        assert len(spec.actuators) >= 1

    def test_production_safety_tier(self, spec) -> None:
        # Power-grid control is safety-critical: the spec must not declare the
        # research tier, so live `spo run` stays gated behind formal export.
        assert str(spec.safety_tier) != "research"


class TestLayerMap:
    def test_layer_map_is_contiguous_and_complete(self, spec) -> None:
        layer_map = power_grid_run._build_layer_map(spec)
        assert set(layer_map) == {layer.index for layer in spec.layers}
        flat = [idx for indices in layer_map.values() for idx in indices]
        total = sum(len(layer.oscillator_ids) for layer in spec.layers)
        assert sorted(flat) == list(range(total))


class TestRun:
    def test_main_executes_end_to_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        power_grid_run.main()
        out = capsys.readouterr().out
        assert "Final" in out
        assert "R_good=" in out
        assert "R_bad=" in out
