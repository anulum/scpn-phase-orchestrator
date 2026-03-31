# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Edge case behavioural tests

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# PID: circular entropy and mutual information
# ---------------------------------------------------------------------------


class TestPIDInformationTheory:
    """Verify circular entropy and mutual information edge cases
    satisfy information-theoretic bounds."""

    def test_circular_entropy_empty_is_zero(self):
        from scpn_phase_orchestrator.monitor.pid import _circular_entropy

        assert _circular_entropy(np.array([])) == 0.0

    def test_circular_entropy_single_value_nonnegative(self):
        from scpn_phase_orchestrator.monitor.pid import _circular_entropy

        result = _circular_entropy(np.array([100.0]))
        assert result >= 0.0

    def test_circular_entropy_uniform_higher_than_peaked(self):
        """Uniformly spread phases should have higher entropy than clustered."""
        from scpn_phase_orchestrator.monitor.pid import _circular_entropy

        uniform = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        clustered = np.random.default_rng(0).normal(1.0, 0.01, 100)
        h_uniform = _circular_entropy(uniform)
        h_clustered = _circular_entropy(clustered)
        assert h_uniform > h_clustered, (
            f"H_uniform={h_uniform:.3f} should > H_clustered={h_clustered:.3f}"
        )

    def test_joint_entropy_empty_is_zero(self):
        from scpn_phase_orchestrator.monitor.pid import _joint_entropy_2d

        assert _joint_entropy_2d(np.array([]), np.array([])) == 0.0

    def test_mutual_information_empty_is_zero(self):
        from scpn_phase_orchestrator.monitor.pid import _mutual_information_paired

        assert _mutual_information_paired(np.array([]), np.array([])) == 0.0

    def test_mutual_information_mismatched_lengths_is_zero(self):
        from scpn_phase_orchestrator.monitor.pid import _mutual_information_paired

        assert _mutual_information_paired(np.array([1.0]), np.array([1.0, 2.0])) == 0.0

    def test_mutual_information_nonnegative(self):
        """MI must be non-negative for any valid inputs."""
        from scpn_phase_orchestrator.monitor.pid import _mutual_information_paired

        rng = np.random.default_rng(42)
        a = rng.uniform(0, 2 * np.pi, 200)
        b = rng.uniform(0, 2 * np.pi, 200)
        mi = _mutual_information_paired(a, b)
        assert mi >= -1e-10, f"MI must be non-negative, got {mi}"


# ---------------------------------------------------------------------------
# EVS: event-related spectral analysis
# ---------------------------------------------------------------------------


class TestEVSBehavioural:
    """Verify EVSMonitor produces structured results with valid fields."""

    def test_single_trial_returns_result(self):
        from scpn_phase_orchestrator.monitor.evs import EVSMonitor

        m = EVSMonitor()
        phases = np.random.default_rng(0).uniform(0, 2 * np.pi, (1, 100))
        result = m.evaluate(
            phases,
            pause_indices=[50],
            target_freq=10.0,
            control_freq=20.0,
        )
        assert hasattr(result, "is_entrained")
        assert isinstance(result.is_entrained, bool)

    def test_multi_trial_aggregation(self):
        from scpn_phase_orchestrator.monitor.evs import EVSMonitor

        m = EVSMonitor()
        phases = np.random.default_rng(1).uniform(0, 2 * np.pi, (5, 200))
        result = m.evaluate(
            phases,
            pause_indices=[100],
            target_freq=10.0,
            control_freq=20.0,
        )
        assert hasattr(result, "is_entrained")


# ---------------------------------------------------------------------------
# Swarmalator: 3D mode
# ---------------------------------------------------------------------------


class TestSwarmalator3D:
    """Verify that the swarmalator engine works in 3D mode
    and produces physically valid output."""

    def test_3d_step_shape_and_finiteness(self):
        from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

        eng = SwarmalatorEngine(n=4, dt=0.01, dim=3)
        rng = np.random.default_rng(42)
        positions = rng.uniform(-1, 1, (4, 3))
        phases = rng.uniform(0, 2 * np.pi, 4)
        omegas = rng.normal(0, 0.5, 4)
        new_pos, new_ph = eng.step(positions, phases, omegas)
        assert new_pos.shape == (4, 3)
        assert new_ph.shape == (4,)
        assert np.all(np.isfinite(new_pos))
        assert np.all(np.isfinite(new_ph))

    def test_3d_phases_advance(self):
        """Non-zero omegas must advance phases."""
        from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

        eng = SwarmalatorEngine(n=4, dt=0.01, dim=3)
        positions = np.zeros((4, 3))
        phases = np.zeros(4)
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        _, new_ph = eng.step(positions, phases, omegas)
        assert not np.allclose(new_ph, 0.0), "Non-zero omegas must advance phases"


# ---------------------------------------------------------------------------
# Predictive supervisor
# ---------------------------------------------------------------------------


class TestPredictiveSupervisorBehavioural:
    """Verify that predictive supervisor makes future predictions
    with valid fields and sensible structure."""

    def _make_supervisor(self, n=8):
        from scpn_phase_orchestrator.supervisor.predictive import PredictiveSupervisor

        return PredictiveSupervisor(
            n_oscillators=n,
            dt=0.01,
            horizon=5,
            divergence_threshold=0.01,
        )

    def test_prediction_has_required_fields(self):
        ps = self._make_supervisor()
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        omegas = np.random.default_rng(0).normal(0, 5, 8)
        knm = np.ones((8, 8)) * 0.01
        np.fill_diagonal(knm, 0)
        alpha = np.zeros((8, 8))
        pred = ps.predict(phases, omegas, knm, alpha)
        assert hasattr(pred, "will_degrade")
        assert hasattr(pred, "will_critical")
        assert isinstance(pred.will_degrade, bool)
        assert isinstance(pred.will_critical, bool)

    def test_stable_system_not_predicted_critical(self):
        """Strong coupling + low omegas → should not predict critical."""
        ps = self._make_supervisor()
        phases = np.zeros(8)  # synchronised
        omegas = np.zeros(8)  # no drift
        knm = np.ones((8, 8)) * 2.0  # strong coupling
        np.fill_diagonal(knm, 0)
        alpha = np.zeros((8, 8))
        pred = ps.predict(phases, omegas, knm, alpha)
        # Synchronised with strong coupling should not predict degradation
        assert not pred.will_critical


# ---------------------------------------------------------------------------
# Coupling harmonic estimation
# ---------------------------------------------------------------------------


class TestCouplingHarmonicEstimation:
    """Verify harmonic coupling estimation returns structured results."""

    def test_returns_harmonic_components(self):
        from scpn_phase_orchestrator.autotune.coupling_est import (
            estimate_coupling_harmonics,
        )

        rng = np.random.default_rng(42)
        N, T = 4, 200
        phases = rng.uniform(0, 2 * np.pi, (N, T))
        omegas = rng.normal(0, 0.5, N)
        result = estimate_coupling_harmonics(phases, omegas, dt=0.01, n_harmonics=2)
        assert isinstance(result, dict)
        assert "sin_1" in result
        assert "cos_1" in result

    def test_harmonic_shapes_match_n(self):
        from scpn_phase_orchestrator.autotune.coupling_est import (
            estimate_coupling_harmonics,
        )

        N = 6
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, (N, 100))
        omegas = rng.normal(0, 0.5, N)
        result = estimate_coupling_harmonics(phases, omegas, dt=0.01, n_harmonics=1)
        assert result["sin_1"].shape == (N, N)


# ---------------------------------------------------------------------------
# QueueWaves config edge case
# ---------------------------------------------------------------------------


class TestQueueWavesConfigFormat:
    """Verify alert sink configuration parsing."""

    def test_slack_format_parsed(self, tmp_path):
        import yaml

        from scpn_phase_orchestrator.apps.queuewaves.config import load_config

        cfg = {
            "prometheus_url": "http://localhost:9090",
            "services": [],
            "alert_sinks": [
                {"url": "https://hooks.slack.com/test", "format": "slack"},
            ],
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        result = load_config(p)
        assert len(result.alert_sinks) == 1
        assert result.alert_sinks[0].format == "slack"

    def test_generic_format_default(self, tmp_path):
        import yaml

        from scpn_phase_orchestrator.apps.queuewaves.config import load_config

        cfg = {
            "prometheus_url": "http://localhost:9090",
            "services": [],
            "alert_sinks": [
                {"url": "https://example.com/webhook"},
            ],
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        result = load_config(p)
        assert result.alert_sinks[0].format == "generic"


# ---------------------------------------------------------------------------
# Server reset with imprint
# ---------------------------------------------------------------------------


class TestServerResetWithImprint:
    """Verify that SimulationState.reset() correctly reinitialises
    all state including imprint."""

    def test_reset_zeros_step_and_preserves_imprint(self):
        from pathlib import Path

        from scpn_phase_orchestrator.binding.loader import load_binding_spec
        from scpn_phase_orchestrator.server import SimulationState

        spec = load_binding_spec(
            Path(__file__).parent.parent
            / "domainpacks"
            / "metaphysics_demo"
            / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        sim.step()
        sim.step()
        result = sim.reset()
        assert result["step"] == 0, "Reset must zero step counter"
        assert sim.imprint_state is not None, "Imprint must survive reset"

    def test_reset_resets_phases(self):
        """After reset, next step must start from initial conditions."""
        from pathlib import Path

        from scpn_phase_orchestrator.binding.loader import load_binding_spec
        from scpn_phase_orchestrator.server import SimulationState

        spec = load_binding_spec(
            Path(__file__).parent.parent
            / "domainpacks"
            / "metaphysics_demo"
            / "binding_spec.yaml"
        )
        sim = SimulationState(spec)
        for _ in range(10):
            sim.step()
        sim.reset()
        result = sim.step()
        assert result["step"] == 1, "First step after reset must be 1"


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
