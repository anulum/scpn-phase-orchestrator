# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov guard tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.lyapunov import LyapunovGuard


def _all_to_all(n: int, k: float = 1.0) -> np.ndarray:
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestLyapunovFunction:
    def test_synchronized_minimum(self):
        phases = np.zeros(4)
        knm = _all_to_all(4)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        # All phases equal → cos(0)=1 → V is at minimum
        assert state.V < 0
        assert state.in_basin

    def test_anti_phase_maximum(self):
        phases = np.array([0.0, np.pi, 0.0, np.pi])
        knm = _all_to_all(4)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        # Anti-phase → higher V (less negative)
        assert state.V > -1.0

    def test_V_decreases_toward_sync(self):
        knm = _all_to_all(4)
        guard = LyapunovGuard()
        # Start spread, move toward sync
        phases1 = np.array([0.0, 0.3, 0.6, 0.9])
        phases2 = np.array([0.0, 0.1, 0.2, 0.3])
        s1 = guard.evaluate(phases1, knm)
        s2 = guard.evaluate(phases2, knm)
        assert s2.V < s1.V
        assert s2.dV_dt < 0

    def test_dV_dt_zero_on_first_call(self):
        guard = LyapunovGuard()
        state = guard.evaluate(np.zeros(3), _all_to_all(3))
        assert state.dV_dt == 0.0

    def test_basin_inside(self):
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        knm = _all_to_all(4)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.max_phase_diff < np.pi / 2
        assert state.in_basin

    def test_basin_outside(self):
        phases = np.array([0.0, 0.0, 0.0, np.pi])
        knm = _all_to_all(4)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.max_phase_diff > np.pi / 2
        assert not state.in_basin

    def test_empty_phases(self):
        guard = LyapunovGuard()
        state = guard.evaluate(np.array([]), np.zeros((0, 0)))
        assert state.V == 0.0
        assert state.in_basin

    def test_no_connections(self):
        phases = np.array([0.0, 1.0, 2.0])
        knm = np.zeros((3, 3))
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.V == 0.0
        assert state.max_phase_diff == 0.0

    def test_reset_clears_prev(self):
        guard = LyapunovGuard()
        guard.evaluate(np.zeros(3), _all_to_all(3))
        guard.reset()
        state = guard.evaluate(np.ones(3), _all_to_all(3))
        assert state.dV_dt == 0.0

    def test_custom_basin_threshold(self):
        phases = np.array([0.0, 1.0])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        guard = LyapunovGuard(basin_threshold=0.5)
        state = guard.evaluate(phases, knm)
        assert not state.in_basin

    def test_wrapping_phase_diff(self):
        # Phases near 0 and 2π should have small diff, not large
        phases = np.array([0.1, 2 * np.pi - 0.1])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.max_phase_diff < 0.3
        assert state.in_basin


class TestLyapunovPipelineWiring:
    """Verify LyapunovGuard wires into the SPO engine pipeline:
    UPDEEngine → phases → LyapunovGuard → basin detection."""

    def test_engine_to_lyapunov_guard(self):
        """Run UPDEEngine 200 steps → feed phases to LyapunovGuard.
        Proves the Lyapunov monitor accepts engine output."""

        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = _all_to_all(n, k=0.5)
        alpha = np.zeros((n, n))
        for _ in range(200):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert isinstance(state.V, float)
        assert isinstance(state.in_basin, bool)
        assert state.max_phase_diff >= 0.0

    def test_lyapunov_evaluate_performance(self):
        """LyapunovGuard.evaluate(N=64) must complete in <1ms."""
        import time

        n = 64
        knm = _all_to_all(n, k=0.3)
        phases = np.random.default_rng(0).uniform(0, 2 * np.pi, n)
        guard = LyapunovGuard()

        # Warm up
        guard.evaluate(phases, knm)

        t0 = time.perf_counter()
        for _ in range(100):
            guard.evaluate(phases, knm)
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.001, f"evaluate(64) took {elapsed * 1000:.2f}ms, limit 1ms"
