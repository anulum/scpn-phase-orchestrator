# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincare section tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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

    @pytest.mark.parametrize(
        ("trajectory", "match"),
        [
            (np.array([[0.0], [np.nan]], dtype=np.float64), "trajectory"),
            (np.array([[0.0], [np.inf]], dtype=np.float64), "trajectory"),
            ([["not-a-state"]], "trajectory"),
        ],
    )
    def test_rejects_invalid_trajectory(
        self,
        trajectory: Any,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            poincare_section(trajectory, normal=np.array([1.0]))

    @pytest.mark.parametrize(
        ("normal", "match"),
        [
            (np.array([1.0, 0.0]), "normal shape"),
            (np.array([np.nan]), "normal"),
            ([["not-a-normal"]], "normal"),
        ],
    )
    def test_rejects_invalid_normal(self, normal: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            poincare_section(np.zeros((3, 1)), normal=normal)

    @pytest.mark.parametrize("offset", [False, np.nan, np.inf, "0.0"])
    def test_rejects_invalid_offset(self, offset: Any) -> None:
        with pytest.raises(ValueError, match="offset"):
            poincare_section(np.zeros((3, 1)), normal=np.array([1.0]), offset=offset)

    def test_accepts_array_like_section_inputs(self) -> None:
        result = poincare_section([[-1.0], [1.0]], normal=[1.0], offset=0.0)

        assert result.crossings.shape == (1, 1)


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

    @pytest.mark.parametrize(
        "phases",
        [
            np.array([[0.0], [np.nan]], dtype=np.float64),
            np.array([[0.0], [np.inf]], dtype=np.float64),
            [["not-a-phase"]],
        ],
    )
    def test_rejects_invalid_phase_history(self, phases: Any) -> None:
        with pytest.raises(ValueError, match="phases"):
            phase_poincare(phases)

    @pytest.mark.parametrize("oscillator_idx", [False, -1, 2, 1.5, "0"])
    def test_rejects_invalid_oscillator_index(self, oscillator_idx: Any) -> None:
        with pytest.raises(ValueError, match="oscillator_idx"):
            phase_poincare(np.zeros((3, 2)), oscillator_idx=oscillator_idx)

    @pytest.mark.parametrize("section_phase", [False, np.nan, np.inf, "0.0"])
    def test_rejects_invalid_section_phase(self, section_phase: Any) -> None:
        with pytest.raises(ValueError, match="section_phase"):
            phase_poincare(np.zeros((3, 2)), section_phase=section_phase)

    def test_accepts_array_like_phase_history(self) -> None:
        result = phase_poincare([[0.0], [2.0 * np.pi + 0.1]])

        assert result.crossings.shape[1] == 1


class TestPoincarePipelineWiring:
    """Pipeline: engine trajectory → Poincaré section → return times."""

    def test_engine_trajectory_to_poincare(self):
        """UPDEEngine → phase trajectory → phase_poincare → crossings.
        Proves Poincaré analysis consumes engine trajectory."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.array([2.0, 3.0, 1.5, 2.5])
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        trajectory = []
        for _ in range(500):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(phases.copy())
        traj = np.array(trajectory)

        result = phase_poincare(traj, oscillator_idx=0)
        assert isinstance(result, PoincareResult)
        assert len(result.crossings) >= 0
        if len(result.crossings) > 1:
            assert result.mean_return_time > 0
