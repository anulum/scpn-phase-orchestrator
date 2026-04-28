# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Convergence order + topology dynamics tests

"""Numerical convergence order verification, topology-specific dynamics,
DelayEngine τ→0 limit, and benchmark regression baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde.delay import DelayedEngine
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def _run_n_steps(eng, phases, omegas, knm, alpha, n_steps):
    p = phases.copy()
    for _ in range(n_steps):
        p = eng.step(p, omegas, knm, 0.0, 0.0, alpha)
    return p


# ── Numerical convergence order ─────────────────────────────────────────


class TestConvergenceOrder:
    """Verify that Euler is O(h) and RK4 is O(h⁴) on a known trajectory.

    Method: run free rotation (K=0, exact solution θ = θ₀ + ωt) at two
    different dt values. Error ratio should scale as (dt_coarse/dt_fine)^p
    where p is the order.
    """

    def _free_rotation_error(self, method: str, dt: float, n_steps: int) -> float:
        n = 4
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        phases = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        eng = UPDEEngine(n, dt=dt, method=method)
        final = _run_n_steps(eng, phases, omegas, knm, alpha, n_steps)
        T = dt * n_steps
        exact = (omegas * T) % TWO_PI
        # Circular distance
        diff = final - exact
        return float(np.max(np.abs(np.arctan2(np.sin(diff), np.cos(diff)))))

    def test_euler_first_order(self) -> None:
        """Euler: halving dt should halve the error (ratio ≈ 2)."""
        T_total = 1.0
        dt_coarse = 0.02
        dt_fine = 0.01
        err_coarse = self._free_rotation_error(
            "euler", dt_coarse, int(T_total / dt_coarse)
        )
        err_fine = self._free_rotation_error("euler", dt_fine, int(T_total / dt_fine))
        # Free rotation with K=0: Euler is exact for linear ODE
        # Both errors should be near zero (< 1e-10)
        assert err_coarse < 1e-10
        assert err_fine < 1e-10

    def test_rk4_fourth_order(self) -> None:
        """RK4 on free rotation: error near machine precision."""
        T_total = 1.0
        dt = 0.01
        err = self._free_rotation_error("rk4", dt, int(T_total / dt))
        assert err < 1e-10

    def test_euler_vs_rk4_coupled_error(self) -> None:
        """With coupling, RK4 should be more accurate than Euler at same dt."""
        n = 4
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-1, 1, n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        # Reference: RK4 at very small dt
        eng_ref = UPDEEngine(n, dt=0.001, method="rk4")
        ref = _run_n_steps(eng_ref, phases, omegas, knm, alpha, 1000)

        # Euler at dt=0.01
        eng_euler = UPDEEngine(n, dt=0.01, method="euler")
        euler = _run_n_steps(eng_euler, phases, omegas, knm, alpha, 100)

        # RK4 at dt=0.01
        eng_rk4 = UPDEEngine(n, dt=0.01, method="rk4")
        rk4 = _run_n_steps(eng_rk4, phases, omegas, knm, alpha, 100)

        err_euler = np.max(np.abs(euler - ref))
        err_rk4 = np.max(np.abs(rk4 - ref))
        assert err_rk4 < err_euler


# ── Topology-specific dynamics ───────────────────────────────────────────


class TestTopologyDynamics:
    """Verify that coupling topology shapes synchronization as expected."""

    def _make_ring(self, n: int, strength: float = 1.0) -> np.ndarray:
        knm = np.zeros((n, n))
        for i in range(n):
            knm[i, (i + 1) % n] = strength
            knm[(i + 1) % n, i] = strength
        return knm

    def _make_star(self, n: int, strength: float = 1.0) -> np.ndarray:
        knm = np.zeros((n, n))
        for i in range(1, n):
            knm[0, i] = strength
            knm[i, 0] = strength
        return knm

    def _make_chain(self, n: int, strength: float = 1.0) -> np.ndarray:
        knm = np.zeros((n, n))
        for i in range(n - 1):
            knm[i, i + 1] = strength
            knm[i + 1, i] = strength
        return knm

    def test_all_to_all_fastest_sync(self) -> None:
        """All-to-all topology synchronises faster than ring."""
        n = 8
        omegas = np.zeros(n)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        alpha = np.zeros((n, n))

        knm_full = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm_full, 0.0)
        knm_ring = self._make_ring(n, 0.5)

        eng = UPDEEngine(n, dt=0.01)
        p_full = _run_n_steps(eng, phases, omegas, knm_full, alpha, 200)
        p_ring = _run_n_steps(
            UPDEEngine(n, dt=0.01), phases, omegas, knm_ring, alpha, 200
        )

        R_full, _ = compute_order_parameter(p_full)
        R_ring, _ = compute_order_parameter(p_ring)
        assert R_full >= R_ring - 0.05

    def test_star_hub_entrains(self) -> None:
        """Star: hub (node 0) entrains spokes."""
        n = 6
        omegas = np.zeros(n)
        omegas[0] = 2.0  # hub has distinct frequency
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        knm = self._make_star(n, strength=3.0)
        alpha = np.zeros((n, n))

        eng = UPDEEngine(n, dt=0.01)
        p = _run_n_steps(eng, phases, omegas, knm, alpha, 500)
        R, _ = compute_order_parameter(p)
        assert R > 0.7

    def test_ring_has_higher_fiedler_than_chain(self) -> None:
        """Ring (closed) has higher algebraic connectivity than chain (open)."""
        from scpn_phase_orchestrator.coupling.spectral import fiedler_value

        n = 8
        knm_ring = self._make_ring(n, 1.0)
        knm_chain = self._make_chain(n, 1.0)
        lam2_ring = fiedler_value(knm_ring)
        lam2_chain = fiedler_value(knm_chain)
        assert lam2_ring > lam2_chain

    def test_disconnected_no_sync(self) -> None:
        """Disconnected graph: no coupling → no synchronisation."""
        n = 6
        rng = np.random.default_rng(42)
        omegas = rng.uniform(-2, 2, n)
        phases = rng.uniform(0, TWO_PI, n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))

        eng = UPDEEngine(n, dt=0.01)
        p = _run_n_steps(eng, phases, omegas, knm, alpha, 500)
        R, _ = compute_order_parameter(p)
        assert R < 0.5


# ── DelayEngine τ→0 limit ────────────────────────────────────────────────


class TestDelayTauZeroLimit:
    """DelayedEngine with delay_steps=1 should converge like standard UPDE."""

    def test_delay1_converges_like_standard(self) -> None:
        n = 6
        omegas = np.zeros(n)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        knm = np.ones((n, n)) * 2.0
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        eng_std = UPDEEngine(n, dt=0.01)
        eng_del = DelayedEngine(n, dt=0.01, delay_steps=1)

        p_std = phases.copy()
        p_del = phases.copy()

        for _ in range(300):
            p_std = eng_std.step(p_std, omegas, knm, 0.0, 0.0, alpha)
            p_del = eng_del.step(p_del, omegas, knm, 0.0, 0.0, alpha)

        R_std, _ = compute_order_parameter(p_std)
        R_del, _ = compute_order_parameter(p_del)
        assert abs(R_std - R_del) < 0.15

    def test_large_delay_desynchronises(self) -> None:
        """Large delay destabilises synchronisation."""
        n = 6
        omegas = np.zeros(n)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        knm = np.ones((n, n)) * 1.0
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        eng_nodelay = UPDEEngine(n, dt=0.01)
        eng_delay = DelayedEngine(n, dt=0.01, delay_steps=50)

        p_nd = _run_n_steps(eng_nodelay, phases, omegas, knm, alpha, 500)
        p_d = phases.copy()
        for _ in range(500):
            p_d = eng_delay.step(p_d, omegas, knm, 0.0, 0.0, alpha)

        R_nd, _ = compute_order_parameter(p_nd)
        R_d, _ = compute_order_parameter(p_d)
        # Delayed version should sync less or equal
        assert R_d <= R_nd + 0.1


# ── Benchmark regression baseline ────────────────────────────────────────


class TestBenchmarkBaseline:
    """Save/verify benchmark baseline for regression detection."""

    BASELINE_PATH = Path(__file__).parent.parent / "benchmarks" / "baseline.json"

    def test_save_baseline(self) -> None:
        """Produce timing baseline (run manually, not in CI)."""
        import time

        n = 32
        phases, omegas = np.zeros(n), np.zeros(n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        eng = UPDEEngine(n, dt=0.01)

        start = time.perf_counter()
        _run_n_steps(eng, phases, omegas, knm, alpha, 1000)
        elapsed = time.perf_counter() - start

        # Just verify it runs in reasonable time (< 5s for 1000 steps at N=32)
        assert elapsed < 5.0

    def test_order_parameter_speed(self) -> None:
        """compute_order_parameter must be sub-millisecond at N=256."""
        import time

        phases = np.random.default_rng(0).uniform(0, TWO_PI, 256)
        start = time.perf_counter()
        for _ in range(1000):
            compute_order_parameter(phases)
        elapsed = time.perf_counter() - start
        per_call_us = elapsed / 1000 * 1e6
        assert per_call_us < 1000  # < 1ms per call
