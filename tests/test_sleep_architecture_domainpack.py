# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep architecture domainpack tests

from __future__ import annotations

from pathlib import Path

import pytest

from domainpacks.sleep_architecture import run as sleep_run
from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec

SPEC_PATH = Path("domainpacks/sleep_architecture/binding_spec.yaml")
EXPECTED_LAYERS = ["delta", "theta", "alpha", "gamma"]


@pytest.fixture(scope="module")
def spec():
    return load_binding_spec(SPEC_PATH)


class TestSpec:
    def test_validates_without_errors(self, spec) -> None:
        assert validate_binding_spec(spec) == []

    def test_name(self, spec) -> None:
        assert spec.name == "sleep_architecture"

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


class TestRun:
    def test_main_executes_end_to_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        sleep_run.main()
        out = capsys.readouterr().out
        assert "Final R=" in out
