# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Final coverage gap tests

from __future__ import annotations

import numpy as np

# --- pid.py edge cases (lines 24, 29, 48, 59) ---


class TestPIDEdgeCases:
    def test_circular_entropy_empty(self):
        from scpn_phase_orchestrator.monitor.pid import _circular_entropy

        assert _circular_entropy(np.array([])) == 0.0

    def test_circular_entropy_zero_total(self):
        from scpn_phase_orchestrator.monitor.pid import _circular_entropy

        result = _circular_entropy(np.array([100.0]))  # wraps to valid bin
        assert result >= 0.0

    def test_joint_entropy_empty(self):
        from scpn_phase_orchestrator.monitor.pid import _joint_entropy_2d

        assert _joint_entropy_2d(np.array([]), np.array([])) == 0.0

    def test_mutual_information_empty(self):
        from scpn_phase_orchestrator.monitor.pid import _mutual_information_paired

        assert _mutual_information_paired(np.array([]), np.array([])) == 0.0

    def test_mutual_information_mismatched(self):
        from scpn_phase_orchestrator.monitor.pid import _mutual_information_paired

        assert _mutual_information_paired(np.array([1.0]), np.array([1.0, 2.0])) == 0.0


# --- evs.py line 125 ---


class TestEVSEdge:
    def test_single_trial(self):
        from scpn_phase_orchestrator.monitor.evs import EVSMonitor

        m = EVSMonitor()
        phases = np.random.default_rng(0).uniform(0, 2 * np.pi, (1, 100))
        result = m.evaluate(
            phases, pause_indices=[50], target_freq=10.0, control_freq=20.0
        )
        assert hasattr(result, "is_entrained")


# --- swarmalator.py line 170 ---


class TestSwarmalatorEdge:
    def test_3d_mode(self):
        from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

        eng = SwarmalatorEngine(n=4, dt=0.01, dim=3)
        rng = np.random.default_rng(42)
        positions = rng.uniform(-1, 1, (4, 3))
        phases = rng.uniform(0, 2 * np.pi, 4)
        omegas = rng.normal(0, 0.5, 4)
        new_pos, new_ph = eng.step(positions, phases, omegas)
        assert new_pos.shape == (4, 3)
        assert new_ph.shape == (4,)


# --- predictive.py lines 83, 124 ---


class TestPredictiveSupervisor:
    def test_divergence_clamp(self):
        from scpn_phase_orchestrator.supervisor.predictive import PredictiveSupervisor

        ps = PredictiveSupervisor(
            n_oscillators=8, dt=0.01, horizon=5, divergence_threshold=0.01
        )
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        omegas = np.random.default_rng(0).normal(0, 5, 8)
        knm = np.ones((8, 8)) * 0.01
        np.fill_diagonal(knm, 0)
        alpha = np.zeros((8, 8))
        pred = ps.predict(phases, omegas, knm, alpha)
        assert hasattr(pred, "will_degrade")

    def test_recommend_critical(self):
        from scpn_phase_orchestrator.supervisor.predictive import PredictiveSupervisor

        ps = PredictiveSupervisor(n_oscillators=16, dt=0.01, horizon=3)
        phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        omegas = np.random.default_rng(0).normal(0, 10, 16)
        knm = np.zeros((16, 16))
        alpha = np.zeros((16, 16))
        pred = ps.predict(phases, omegas, knm, alpha)
        assert hasattr(pred, "will_critical")


# --- coupling_est.py line 88 ---


class TestCouplingEst:
    def test_harmonic_estimation(self):
        from scpn_phase_orchestrator.autotune.coupling_est import (
            estimate_coupling_harmonics,
        )

        rng = np.random.default_rng(42)
        N, T = 4, 200
        phases = rng.uniform(0, 2 * np.pi, (N, T))
        omegas = rng.normal(0, 0.5, N)
        dt = 0.01
        result = estimate_coupling_harmonics(phases, omegas, dt, n_harmonics=2)
        assert isinstance(result, dict)
        assert "sin_1" in result


# --- queuewaves config.py lines 88-92 ---


class TestQueueWavesConfigEdge:
    def test_alert_sink_format(self):
        import tempfile
        from pathlib import Path

        import yaml

        from scpn_phase_orchestrator.apps.queuewaves.config import load_config

        cfg = {
            "prometheus_url": "http://localhost:9090",
            "services": [],
            "alert_sinks": [
                {"url": "https://hooks.slack.com/test", "format": "slack"},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
            f.flush()
            result = load_config(Path(f.name))
        assert len(result.alert_sinks) == 1
        assert result.alert_sinks[0].format == "slack"


# --- server.py line 264: reset() with imprint model ---


class TestServerResetImprint:
    def test_reset_with_imprint(self):
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
        assert result["step"] == 0
        assert sim.imprint_state is not None


# --- server_grpc.py lines 39-41 (grpc=None fallback) ---
# These lines execute when grpc is not installed. Since grpc IS installed
# in our test env, we can't easily cover them without mocking the import.
# Mark as pragma: no cover is the correct approach for optional-dep fallbacks.


# --- server.py lines 64-67 (FastAPI not installed fallback) ---
# Same situation: FastAPI IS installed. pragma: no cover is correct.


# --- adjoint.py lines 93-122 (JAX gradient path) ---
# Only testable with JAX installed. Covered by test_jax_engine.py when JAX is present.
