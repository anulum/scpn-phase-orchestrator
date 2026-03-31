# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fusion-Core bridge tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.adapters.fusion_core_bridge import (
    BETA_N_LIMIT,
    FusionCoreBridge,
)

TWO_PI = 2.0 * np.pi


class TestObservablesToPhases:
    def test_nominal_output_range(self):
        bridge = FusionCoreBridge(n_layers=6)
        snapshot = {
            "q_profile": 2.0,
            "q_min": 1.0,
            "q_max": 5.0,
            "beta_n": 1.4,
            "tau_e": 1.5,
            "sawtooth_count": 0,
            "elm_count": 0,
            "mhd_amplitude": 0.3,
        }
        phases = bridge.observables_to_phases(snapshot)
        assert phases.shape == (6,)
        assert np.all(phases >= 0.0)
        assert np.all(phases <= TWO_PI)

    def test_sawtooth_kick(self):
        bridge = FusionCoreBridge(n_layers=6)
        snap_0 = {"sawtooth_count": 0}
        snap_1 = {"sawtooth_count": 1}
        p0 = bridge.observables_to_phases(snap_0)
        p1 = bridge.observables_to_phases(snap_1)
        assert p1[3] == pytest.approx(np.pi)
        assert p0[3] == pytest.approx(0.0)

    def test_elm_kick(self):
        bridge = FusionCoreBridge(n_layers=6)
        snap = {"elm_count": 3}
        phases = bridge.observables_to_phases(snap)
        expected = (3 * np.pi) % TWO_PI
        assert phases[4] == pytest.approx(expected)

    def test_beta_n_at_limit(self):
        bridge = FusionCoreBridge(n_layers=6)
        snap = {"beta_n": BETA_N_LIMIT}
        phases = bridge.observables_to_phases(snap)
        assert phases[1] == pytest.approx(TWO_PI)

    def test_clamps_above_limit(self):
        bridge = FusionCoreBridge(n_layers=6)
        snap = {"beta_n": 5.0}
        phases = bridge.observables_to_phases(snap)
        assert phases[1] == pytest.approx(TWO_PI)


class TestPhasesToFeedback:
    def test_output_format(self):
        bridge = FusionCoreBridge(n_layers=6)
        phases = np.zeros(6)
        omegas = np.ones(6)
        fb = bridge.phases_to_feedback(phases, omegas)
        assert "R_global" in fb
        assert "mean_phase" in fb
        assert "mean_omega" in fb
        assert fb["n_oscillators"] == 6
        assert fb["R_global"] == pytest.approx(1.0)


class TestConstructorValidation:
    def test_n_layers_zero_raises(self):
        with pytest.raises(ValueError, match="n_layers must be >= 1"):
            FusionCoreBridge(n_layers=0)


class TestQProfileImport:
    def test_dict_import(self):
        bridge = FusionCoreBridge()
        q_data = {
            "q_min": 0.9,
            "q_max": 4.0,
            "q_axis": 0.95,
            "q_edge": 3.8,
        }
        result = bridge.import_q_profile(q_data)
        assert result["q_min"] == pytest.approx(0.9)
        assert result["q_edge"] == pytest.approx(3.8)

    def test_missing_keys_use_defaults(self):
        bridge = FusionCoreBridge()
        result = bridge.import_q_profile({})
        assert result["q_min"] == pytest.approx(1.0)
        assert result["q_max"] == pytest.approx(5.0)

    def test_object_import_via_getattr(self):
        bridge = FusionCoreBridge()

        class QProfile:
            q_min = 0.8
            q_max = 4.5
            q_axis = 0.85
            q_edge = 4.2

        result = bridge.import_q_profile(QProfile())
        assert result["q_min"] == pytest.approx(0.8)
        assert result["q_edge"] == pytest.approx(4.2)


class TestStabilityChecks:
    def test_nominal_no_violations(self):
        bridge = FusionCoreBridge()
        violations = bridge.check_stability(
            {
                "q_min": 1.5,
                "beta_n": 2.0,
                "tau_e_ratio": 0.8,
            }
        )
        assert violations == []

    def test_q_min_violation(self):
        bridge = FusionCoreBridge()
        violations = bridge.check_stability({"q_min": 0.7})
        assert len(violations) == 1
        assert violations[0]["variable"] == "q_min"

    def test_beta_n_violation(self):
        bridge = FusionCoreBridge()
        violations = bridge.check_stability({"beta_n": 3.5})
        assert len(violations) == 1
        assert violations[0]["variable"] == "beta_n"

    def test_tau_e_ratio_soft_violation(self):
        bridge = FusionCoreBridge()
        violations = bridge.check_stability({"tau_e_ratio": 0.3})
        assert len(violations) == 1
        assert violations[0]["severity"] == "soft"

    def test_q_profile_fallback_for_q_min(self):
        bridge = FusionCoreBridge()
        violations = bridge.check_stability({"q_profile": 0.8})
        assert len(violations) == 1
        assert violations[0]["variable"] == "q_min"


class TestEquilibriumImport:
    def test_import_equilibrium(self):
        bridge = FusionCoreBridge()
        result = bridge.import_equilibrium(
            {
                "q_profile": 2.0,
                "beta_n": 1.5,
                "tau_e": 2.0,
                "sawtooth_count": 3,
                "elm_count": 1,
                "mhd_amplitude": 0.2,
            }
        )
        assert result["q_profile"] == pytest.approx(2.0)
        assert result["sawtooth_count"] == 3


class TestFusionDomainpack:
    def test_binding_spec_loads(self):
        from pathlib import Path

        from scpn_phase_orchestrator.binding.loader import load_binding_spec

        _root = Path(__file__).parent.parent
        spec_path = _root / "domainpacks" / "fusion_equilibrium" / "binding_spec.yaml"
        spec = load_binding_spec(spec_path)
        assert spec.name == "fusion_equilibrium"
        assert len(spec.layers) == 6
        n_osc = sum(len(ly.oscillator_ids) for ly in spec.layers)
        assert n_osc == 12

    def test_binding_spec_validates(self):
        from pathlib import Path

        from scpn_phase_orchestrator.binding import validate_binding_spec
        from scpn_phase_orchestrator.binding.loader import load_binding_spec

        _root = Path(__file__).parent.parent
        spec_path = _root / "domainpacks" / "fusion_equilibrium" / "binding_spec.yaml"
        spec = load_binding_spec(spec_path)
        errors = validate_binding_spec(spec)
        assert errors == [], f"Validation errors: {errors}"


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
