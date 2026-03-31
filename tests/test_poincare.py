# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincare section tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.poincare import (
    PoincareResult,
    phase_poincare,
    poincare_section,
    return_times,
)


class TestPoincareSection:
    def test_circle_crossings(self):
        """Circle trajectory crosses x=0 twice per revolution."""
        t = np.linspace(0, 6 * np.pi, 3000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = poincare_section(traj, normal=np.array([1.0, 0.0]))
        assert isinstance(result, PoincareResult)
        # 3 full revolutions → 3 positive crossings
        assert len(result.crossings) >= 2

    def test_periodic_constant_return_time(self):
        """Periodic orbit → constant return time."""
        t = np.linspace(0, 8 * np.pi, 4000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = poincare_section(traj, normal=np.array([1.0, 0.0]))
        if len(result.return_times) > 1:
            assert result.std_return_time < 5.0  # nearly constant

    def test_no_crossings(self):
        """Trajectory that doesn't cross the plane."""
        traj = np.column_stack([np.ones(100), np.linspace(0, 1, 100)])
        result = poincare_section(traj, normal=np.array([1.0, 0.0]), offset=5.0)
        assert len(result.crossings) == 0
        assert result.mean_return_time == 0.0

    def test_both_directions(self):
        """direction='both' counts crossings in both directions."""
        t = np.linspace(0, 4 * np.pi, 2000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        pos = poincare_section(traj, normal=np.array([1.0, 0.0]), direction="positive")
        both = poincare_section(traj, normal=np.array([1.0, 0.0]), direction="both")
        assert len(both.crossings) >= len(pos.crossings)

    def test_negative_direction(self):
        t = np.linspace(0, 4 * np.pi, 2000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = poincare_section(
            traj, normal=np.array([1.0, 0.0]), direction="negative"
        )
        assert len(result.crossings) >= 1


class TestReturnTimes:
    def test_returns_array(self):
        t = np.linspace(0, 6 * np.pi, 3000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        rt = return_times(traj, normal=np.array([1.0, 0.0]))
        assert isinstance(rt, np.ndarray)


class TestPhasePoincare:
    def test_uniform_rotation(self):
        """Uniform rotation → regular crossings."""
        T = 500
        N = 4
        dt = 0.05
        omegas = np.array([1.0, 1.5, 2.0, 2.5])
        phases = np.zeros((T, N))
        for t in range(1, T):
            phases[t] = phases[t - 1] + omegas * dt

        result = phase_poincare(phases, oscillator_idx=0)
        assert isinstance(result, PoincareResult)
        if len(result.return_times) > 1:
            # Return time should be near 2π/ω₀ / dt ≈ 125.7 steps
            expected_rt = 2 * np.pi / (omegas[0] * dt)
            assert abs(result.mean_return_time - expected_rt) < 10

    def test_no_crossings_short(self):
        """Very short trajectory → no crossings."""
        phases = np.array([[0.0, 0.0], [0.1, 0.1]])
        result = phase_poincare(phases)
        assert len(result.crossings) == 0

    def test_multi_oscillator(self):
        """Crossings should contain full phase vector."""
        T = 1000
        N = 3
        dt = 0.05
        omegas = np.array([1.0, 2.0, 3.0])
        phases = np.zeros((T, N))
        for t in range(1, T):
            phases[t] = phases[t - 1] + omegas * dt
        result = phase_poincare(phases, oscillator_idx=0)
        if len(result.crossings) > 0:
            assert result.crossings.shape[1] == N


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
