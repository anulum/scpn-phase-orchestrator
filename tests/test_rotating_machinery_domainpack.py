# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Rotating machinery domainpack tests

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from domainpacks.rotating_machinery import run as rotating_machinery_run
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.coupling.knm import CouplingState

SPEC_PATH = Path("domainpacks/rotating_machinery/binding_spec.yaml")
EXPECTED_LAYERS = [
    "shaft_rotation",
    "blade_dynamics",
    "bearing_condition",
    "structural_resonance",
]


@pytest.fixture(scope="module")
def spec() -> BindingSpec:
    return load_binding_spec(SPEC_PATH)


class TestSpec:
    def test_validates_without_errors(self, spec: BindingSpec) -> None:
        assert validate_binding_spec(spec) == []

    def test_name(self, spec: BindingSpec) -> None:
        assert spec.name == "rotating_machinery"

    def test_layer_names_match_documented_structure(self, spec: BindingSpec) -> None:
        assert [layer.name for layer in spec.layers] == EXPECTED_LAYERS

    def test_layer_indices_are_contiguous(self, spec: BindingSpec) -> None:
        assert [layer.index for layer in spec.layers] == list(range(len(spec.layers)))

    def test_every_layer_has_oscillators(self, spec: BindingSpec) -> None:
        assert all(len(layer.oscillator_ids) >= 1 for layer in spec.layers)

    def test_omegas_count_matches_oscillator_total(self, spec: BindingSpec) -> None:
        total = sum(len(layer.oscillator_ids) for layer in spec.layers)
        assert len(spec.get_omegas()) == total

    def test_objectives_reference_valid_layers(self, spec: BindingSpec) -> None:
        valid = {layer.index for layer in spec.layers}
        assert set(spec.objectives.good_layers) <= valid
        assert set(spec.objectives.bad_layers) <= valid
        assert not set(spec.objectives.good_layers) & set(spec.objectives.bad_layers)

    def test_boundaries_and_actuators_present(self, spec: BindingSpec) -> None:
        assert len(spec.boundaries) >= 1
        assert len(spec.actuators) >= 1


class TestLayerMap:
    def test_layer_map_is_contiguous_and_complete(self, spec: BindingSpec) -> None:
        layer_map = rotating_machinery_run._build_layer_map(spec)
        assert set(layer_map) == {layer.index for layer in spec.layers}
        flat = [idx for indices in layer_map.values() for idx in indices]
        total = sum(len(layer.oscillator_ids) for layer in spec.layers)
        assert sorted(flat) == list(range(total))


class TestRun:
    def test_main_executes_end_to_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        rotating_machinery_run.main()
        out = capsys.readouterr().out
        assert "Final" in out
        assert "R_good=" in out
        assert "R_bad=" in out


def test_rotating_machinery_policy_alpha_action_updates_blade_damper_lag() -> None:
    coupling = CouplingState(
        knm=np.ones((10, 10), dtype=float),
        alpha=np.zeros((10, 10), dtype=float),
        active_template="base",
        knm_r=None,
    )
    action = ControlAction(
        knob="alpha",
        scope="layer_1",
        value=0.4,
        ttl_s=5.0,
        justification="policy flutter suppression",
    )

    updated, zeta = rotating_machinery_run._apply_rotating_machinery_action(
        coupling,
        action,
        zeta=0.04,
        layer_map={0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7], 3: [8, 9]},
    )

    assert updated.alpha[3, 4] == pytest.approx(0.4)
    assert updated.alpha[4, 3] == pytest.approx(0.4)
    assert np.count_nonzero(updated.alpha[:3, :]) == 0
    assert np.count_nonzero(updated.alpha[:, :3]) == 0
    assert np.count_nonzero(updated.alpha[5:, :]) == 0
    assert np.count_nonzero(updated.alpha[:, 5:]) == 0
    assert np.array_equal(updated.knm, coupling.knm)
    assert zeta == pytest.approx(0.04)
