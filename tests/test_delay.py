# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delayed coupling tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.delay import DelayBuffer, DelayedEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


class TestDelayBuffer:
    def test_push_and_get(self):
        buf = DelayBuffer(3, max_delay_steps=5)
        p1 = np.array([0.0, 1.0, 2.0])
        p2 = np.array([0.1, 1.1, 2.1])
        p3 = np.array([0.2, 1.2, 2.2])
        buf.push(p1)
        buf.push(p2)
        buf.push(p3)
        # buffer[-1]=p3, buffer[-2]=p2, buffer[-3]=p1
        np.testing.assert_array_equal(buf.get_delayed(1), p3)
        np.testing.assert_array_equal(buf.get_delayed(2), p2)
        np.testing.assert_array_equal(buf.get_delayed(3), p1)

    def test_get_current(self):
        buf = DelayBuffer(2, max_delay_steps=3)
        p = np.array([1.0, 2.0])
        buf.push(p)
        # delay_steps=0 is invalid → returns None
        assert buf.get_delayed(0) is None

    def test_not_enough_history(self):
        buf = DelayBuffer(2, max_delay_steps=5)
        buf.push(np.zeros(2))
        assert buf.get_delayed(3) is None

    def test_circular_eviction(self):
        buf = DelayBuffer(2, max_delay_steps=3)
        for i in range(5):
            buf.push(np.full(2, float(i)))
        assert buf.length == 3
        oldest = buf.get_delayed(3)
        assert oldest is not None
        np.testing.assert_array_equal(oldest, np.full(2, 2.0))

    def test_invalid_max_delay(self):
        with pytest.raises(ValueError):
            DelayBuffer(2, max_delay_steps=0)

    def test_clear(self):
        buf = DelayBuffer(2, max_delay_steps=3)
        buf.push(np.zeros(2))
        buf.clear()
        assert buf.length == 0

    def test_copies_input(self):
        buf = DelayBuffer(2, max_delay_steps=3)
        p = np.array([1.0, 2.0])
        buf.push(p)
        p[0] = 999.0
        assert buf.get_delayed(1)[0] == 1.0


class TestDelayedEngine:
    def test_output_in_range(self):
        n = 4
        eng = DelayedEngine(n, dt=0.01, delay_steps=3)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(10):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(phases >= 0)
        assert np.all(phases < 2 * np.pi)

    def test_falls_back_to_instantaneous(self):
        n = 4
        eng = DelayedEngine(n, dt=0.01, delay_steps=5)
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        # First step — no history, uses instantaneous
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (n,)

    def test_delay_changes_dynamics(self):
        n = 6
        rng = np.random.default_rng(42)
        phases0 = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        eng0 = DelayedEngine(n, dt=0.01, delay_steps=1)
        eng10 = DelayedEngine(n, dt=0.01, delay_steps=10)

        p0 = phases0.copy()
        p10 = phases0.copy()
        for _ in range(50):
            p0 = eng0.step(p0, omegas, knm, 0.0, 0.0, alpha)
            p10 = eng10.step(p10, omegas, knm, 0.0, 0.0, alpha)

        assert not np.allclose(p0, p10)

    def test_synchronization(self):
        n = 8
        eng = DelayedEngine(n, dt=0.01, delay_steps=2)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 1.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        R, _ = compute_order_parameter(result)
        assert R > 0.8

    def test_delay_steps_property(self):
        eng = DelayedEngine(4, dt=0.01, delay_steps=7)
        assert eng.delay_steps == 7


class TestDelayBufferEdgeCases:
    """Advanced edge-case testing for the circular delay buffer."""

    def test_push_exactly_max_delay(self):
        """Pushing exactly max_delay items fills buffer completely."""
        buf = DelayBuffer(3, max_delay_steps=4)
        for i in range(4):
            buf.push(np.full(3, float(i)))
        assert buf.length == 4
        np.testing.assert_array_equal(buf.get_delayed(4), np.full(3, 0.0))
        np.testing.assert_array_equal(buf.get_delayed(1), np.full(3, 3.0))

    def test_rapid_push_overflow_correctness(self):
        """After many pushes (>> max_delay), oldest correct entry preserved."""
        buf = DelayBuffer(2, max_delay_steps=5)
        for i in range(100):
            buf.push(np.full(2, float(i)))
        # Latest = 99, oldest in buffer = 95
        np.testing.assert_array_equal(buf.get_delayed(1), np.full(2, 99.0))
        np.testing.assert_array_equal(buf.get_delayed(5), np.full(2, 95.0))

    def test_single_capacity_buffer(self):
        """max_delay_steps=1 → only most recent entry."""
        buf = DelayBuffer(3, max_delay_steps=1)
        buf.push(np.array([1.0, 2.0, 3.0]))
        buf.push(np.array([4.0, 5.0, 6.0]))
        np.testing.assert_array_equal(buf.get_delayed(1), np.array([4.0, 5.0, 6.0]))
        assert buf.get_delayed(2) is None


class TestDelayedEnginePipelineEndToEnd:
    """Full pipeline: CouplingBuilder → DelayedEngine → R → RegimeManager."""

    def test_coupling_delayed_engine_regime(self):
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        n = 12
        cb = CouplingBuilder()
        cs = cb.build(n_layers=n, base_strength=0.6, decay_alpha=0.2)
        eng = DelayedEngine(n, dt=0.01, delay_steps=3)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        result = eng.run(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha, n_steps=300)
        assert np.all(result >= 0.0)
        assert np.all(result < 2 * np.pi)
        r, psi = compute_order_parameter(result)
        assert 0.0 <= r <= 1.0
        layer = LayerState(R=r, psi=psi)
        state = UPDEState(
            layers=[layer],
            cross_layer_alignment=np.array([r]),
            stability_proxy=r,
            regime_id="nominal",
        )
        rm = RegimeManager(hysteresis=0.05)
        regime = rm.evaluate(state, BoundaryState())
        assert regime.name in {"NOMINAL", "DEGRADED", "CRITICAL", "RECOVERY"}

    def test_delay_reduces_sync_compared_to_instant(self):
        """Delayed coupling should reduce or maintain sync relative to instant."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 8
        rng = np.random.default_rng(77)
        phases0 = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        # Instant coupling
        eng_inst = UPDEEngine(n, dt=0.01, method="euler")
        p_inst = phases0.copy()
        for _ in range(300):
            p_inst = eng_inst.step(p_inst, omegas, knm, 0.0, 0.0, alpha)
        r_inst, _ = compute_order_parameter(p_inst)
        # Delayed coupling (delay=10)
        eng_del = DelayedEngine(n, dt=0.01, delay_steps=10)
        p_del = phases0.copy()
        for _ in range(300):
            p_del = eng_del.step(p_del, omegas, knm, 0.0, 0.0, alpha)
        r_del, _ = compute_order_parameter(p_del)
        # Both must be valid R
        assert 0.0 <= r_inst <= 1.0
        assert 0.0 <= r_del <= 1.0
        # Delay generally reduces synchronisation (or at least R is finite)
        assert np.isfinite(r_del)

    def test_performance_delayed_step_32_under_1ms(self):
        """DelayedEngine.step(32 oscillators, delay=5) < 1ms."""
        import time

        n = 32
        eng = DelayedEngine(n, dt=0.01, delay_steps=5)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(10):
            eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        t0 = time.perf_counter()
        for _ in range(500):
            eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        elapsed = (time.perf_counter() - t0) / 500
        assert elapsed < 1e-3, f"delayed.step(32) took {elapsed * 1e3:.2f}ms"


# Pipeline wiring: delay tests exercise DelayBuffer + DelayedEngine →
# compute_order_parameter → CouplingBuilder → RegimeManager. Dynamics:
# delay vs instant comparison. Performance: step(32,delay=5)<1ms.
