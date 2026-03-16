# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plasma-control bridge tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.plasma_control_bridge import (
    PlasmaControlBridge,
)

TWO_PI = 2.0 * np.pi


class TestConstructorValidation:
    def test_n_layers_zero_raises(self):
        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            PlasmaControlBridge(n_layers=0)


class TestKnmSpecImport:
    def test_kronecker_expansion_shape(self):
        bridge = PlasmaControlBridge(n_layers=4)
        layer_knm = np.array(
            [
                [0.0, 0.5, 0.2, 0.1],
                [0.5, 0.0, 0.4, 0.15],
                [0.2, 0.4, 0.0, 0.3],
                [0.1, 0.15, 0.3, 0.0],
            ]
        )
        spec = {"matrix": layer_knm.tolist(), "n_osc_per_layer": 3}
        coupling = bridge.import_knm_spec(spec)
        assert coupling.knm.shape == (12, 12)
        assert coupling.alpha.shape == (12, 12)
        assert np.all(np.diag(coupling.knm) == 0.0)

    def test_kronecker_expansion_preserves_structure(self):
        bridge = PlasmaControlBridge(n_layers=2)
        layer_knm = np.array([[0.0, 0.8], [0.8, 0.0]])
        coupling = bridge.import_knm_spec(layer_knm)
        # n_osc_per_layer defaults to 2 → shape (4,4)
        assert coupling.knm.shape == (4, 4)
        # Off-diagonal blocks should carry the 0.8 coupling
        assert coupling.knm[0, 2] == pytest.approx(0.8)
        assert coupling.knm[2, 0] == pytest.approx(0.8)

    def test_rejects_non_square(self):
        bridge = PlasmaControlBridge(n_layers=4)
        with pytest.raises(ValueError, match="square"):
            bridge.import_knm_spec(np.zeros((3, 4)))


class TestSnapshotImport:
    def test_snapshot_to_upde_state(self):
        bridge = PlasmaControlBridge(n_layers=4)
        phases = np.linspace(0, TWO_PI, 8, endpoint=False)
        snapshot = {
            "phases": phases.tolist(),
            "regime": "DEGRADED",
            "layer_sizes": [2, 2, 2, 2],
            "stability": 0.65,
        }
        state = bridge.import_snapshot(snapshot)
        assert state.regime_id == "DEGRADED"
        assert state.stability_proxy == pytest.approx(0.65)
        assert len(state.layers) == 4
        for ls in state.layers:
            assert 0.0 <= ls.R <= 1.0

    def test_snapshot_default_layer_sizes(self):
        bridge = PlasmaControlBridge(n_layers=4)
        snapshot = {"phases": [0.0] * 8}
        state = bridge.import_snapshot(snapshot)
        assert len(state.layers) == 4

    def test_snapshot_empty_layer_group(self):
        bridge = PlasmaControlBridge(n_layers=3)
        snapshot = {"phases": [0.1, 0.2], "layer_sizes": [1, 1, 0]}
        state = bridge.import_snapshot(snapshot)
        assert len(state.layers) == 3
        assert pytest.approx(0.0) == state.layers[2].R


class TestLyapunovVerdict:
    def test_stable_verdict(self):
        bridge = PlasmaControlBridge()
        result = bridge.import_lyapunov_verdict({"score": 0.8})
        assert result["lyapunov_score"] == pytest.approx(0.8)
        assert result["stable"] is True

    def test_unstable_verdict(self):
        bridge = PlasmaControlBridge()
        result = bridge.import_lyapunov_verdict({"score": 0.1})
        assert result["stable"] is False

    def test_verdict_from_object(self):
        from types import SimpleNamespace

        bridge = PlasmaControlBridge()
        verdict = SimpleNamespace(score=0.6)
        result = bridge.import_lyapunov_verdict(verdict)
        assert result["lyapunov_score"] == pytest.approx(0.6)
        assert result["stable"] is True


class TestPhysicsInvariants:
    def test_nominal_no_violations(self):
        bridge = PlasmaControlBridge()
        violations = bridge.check_physics_invariants(
            {
                "q_min": 1.5,
                "beta_n": 2.0,
                "greenwald": 0.8,
            }
        )
        assert violations == []

    def test_q_min_violation(self):
        bridge = PlasmaControlBridge()
        violations = bridge.check_physics_invariants({"q_min": 0.7})
        assert len(violations) == 1
        assert violations[0]["variable"] == "q_min"
        assert violations[0]["severity"] == "hard"

    def test_beta_n_violation(self):
        bridge = PlasmaControlBridge()
        violations = bridge.check_physics_invariants({"beta_n": 3.5})
        assert len(violations) == 1
        assert violations[0]["variable"] == "beta_n"

    def test_multiple_violations(self):
        bridge = PlasmaControlBridge()
        violations = bridge.check_physics_invariants(
            {
                "q_min": 0.5,
                "beta_n": 3.5,
                "greenwald": 1.5,
            }
        )
        assert len(violations) == 3

    def test_missing_values_no_crash(self):
        bridge = PlasmaControlBridge()
        violations = bridge.check_physics_invariants({})
        assert violations == []


class TestExportActions:
    def test_export_actions(self):
        bridge = PlasmaControlBridge()
        result = bridge.export_control_actions(
            [
                {"knob": "K", "scope": "global", "value": 0.5},
                {"knob": "zeta", "scope": "layer_0", "value": 0.1},
            ]
        )
        assert len(result["actions"]) == 2
        assert result["actions"][0]["knob"] == "K"


class TestPlasmaOmega:
    def test_omega_length(self):
        bridge = PlasmaControlBridge(n_layers=8)
        omegas = bridge.import_plasma_omega(n_osc_per_layer=2)
        assert len(omegas) == 16
        assert np.all(omegas > 0)

    def test_omega_single_oscillator(self):
        bridge = PlasmaControlBridge(n_layers=4)
        omegas = bridge.import_plasma_omega(n_osc_per_layer=1)
        assert len(omegas) == 4


class TestPlasmaDomainpack:
    def test_binding_spec_loads(self):
        from pathlib import Path

        from scpn_phase_orchestrator.binding.loader import load_binding_spec

        _root = Path(__file__).parent.parent
        spec_path = _root / "domainpacks" / "plasma_control" / "binding_spec.yaml"
        spec = load_binding_spec(spec_path)
        assert spec.name == "plasma_control"
        assert len(spec.layers) == 8
        n_osc = sum(len(ly.oscillator_ids) for ly in spec.layers)
        assert n_osc == 16

    def test_binding_spec_validates(self):
        from pathlib import Path

        from scpn_phase_orchestrator.binding import validate_binding_spec
        from scpn_phase_orchestrator.binding.loader import load_binding_spec

        _root = Path(__file__).parent.parent
        spec_path = _root / "domainpacks" / "plasma_control" / "binding_spec.yaml"
        spec = load_binding_spec(spec_path)
        errors = validate_binding_spec(spec)
        assert errors == [], f"Validation errors: {errors}"
