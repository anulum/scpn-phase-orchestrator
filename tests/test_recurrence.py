# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Recurrence analysis tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.recurrence import (
    RQAResult,
    cross_recurrence_matrix,
    cross_rqa,
    recurrence_matrix,
    rqa,
)


class TestRecurrenceMatrix:
    def test_identical_trajectory_is_all_recurrent(self):
        """Constant trajectory → every point recurs with every other."""
        traj = np.ones((20, 2))
        R = recurrence_matrix(traj, epsilon=0.1)
        assert R.shape == (20, 20)
        assert R.all()

    def test_diverging_trajectory_sparse(self):
        """Linearly increasing → few recurrences for small ε."""
        traj = np.arange(50).astype(float)[:, np.newaxis]
        R = recurrence_matrix(traj, epsilon=0.5)
        # Only adjacent points (|i-j|<=0.5) should recur
        np.fill_diagonal(R, False)
        assert R.sum() == 0  # no off-diagonal recurrences

    def test_periodic_trajectory_structured(self):
        """Sine wave → recurrence at period intervals."""
        t = np.linspace(0, 4 * np.pi, 200)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        R = recurrence_matrix(traj, epsilon=0.3)
        rr = R.sum() / R.size
        assert 0.01 < rr < 0.5  # structured, not trivial

    def test_angular_metric(self):
        """Angular metric uses chord distance on circle."""
        traj = np.array([0.0, np.pi, 0.1])[:, np.newaxis]
        R = recurrence_matrix(traj, epsilon=0.3, metric="angular")
        assert R[0, 2]  # 0.0 and 0.1 are close
        assert not R[0, 1]  # 0 and π are far

    def test_1d_input(self):
        """1D array input should work."""
        traj = np.zeros(10)
        R = recurrence_matrix(traj, epsilon=0.1)
        assert R.shape == (10, 10)


class TestRQA:
    def test_periodic_high_determinism(self):
        """Periodic signal → high DET (diagonal lines dominate)."""
        t = np.linspace(0, 6 * np.pi, 300)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = rqa(traj, epsilon=0.2)
        assert isinstance(result, RQAResult)
        assert result.determinism > 0.4
        assert result.max_diagonal > 5

    def test_random_low_determinism(self):
        """Random noise → low DET."""
        rng = np.random.default_rng(42)
        traj = rng.normal(0, 1, (100, 2))
        result = rqa(traj, epsilon=0.3)
        assert result.determinism < 0.5

    def test_laminar_trapping(self):
        """Trajectory with stuck segments → laminarity > 0."""
        # Stay at origin for 20 steps, then move, then return
        traj = np.zeros((60, 1))
        traj[20:40, 0] = np.linspace(0, 5, 20)
        result = rqa(traj, epsilon=0.1, v_min=3)
        assert result.laminarity > 0
        assert result.trapping_time > 0

    def test_empty_trajectory(self):
        """Very short trajectory still returns valid result."""
        traj = np.array([[0.0], [1.0]])
        result = rqa(traj, epsilon=0.01)
        assert result.recurrence_rate == 0.0

    def test_angular_rqa(self):
        """RQA with angular metric for phase data."""
        t = np.linspace(0, 8 * np.pi, 200)
        traj = t[:, np.newaxis] % (2 * np.pi)
        result = rqa(traj, epsilon=0.2, metric="angular")
        assert result.recurrence_rate > 0
        assert result.determinism > 0

    def test_rqa_fields(self):
        """All RQA fields present and finite."""
        t = np.linspace(0, 4 * np.pi, 100)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = rqa(traj, epsilon=0.3)
        assert np.isfinite(result.recurrence_rate)
        assert np.isfinite(result.determinism)
        assert np.isfinite(result.avg_diagonal)
        assert np.isfinite(result.entropy_diagonal)
        assert np.isfinite(result.laminarity)
        assert np.isfinite(result.trapping_time)
        assert isinstance(result.max_diagonal, int)
        assert isinstance(result.max_vertical, int)


class TestCrossRecurrence:
    def test_identical_trajectories(self):
        """Same trajectory → cross-recurrence = auto-recurrence."""
        t = np.linspace(0, 2 * np.pi, 50)
        traj = np.sin(t)[:, np.newaxis]
        CR = cross_recurrence_matrix(traj, traj, epsilon=0.1)
        R = recurrence_matrix(traj, epsilon=0.1)
        np.testing.assert_array_equal(CR, R)

    def test_phase_shifted_recurrence(self):
        """Phase-shifted signals should show off-diagonal structure."""
        t = np.linspace(0, 4 * np.pi, 100)
        a = np.sin(t)[:, np.newaxis]
        b = np.sin(t + np.pi / 4)[:, np.newaxis]
        CR = cross_recurrence_matrix(a, b, epsilon=0.3)
        assert CR.sum() > 0

    def test_cross_rqa_returns_result(self):
        """cross_rqa returns valid RQAResult."""
        t = np.linspace(0, 4 * np.pi, 100)
        a = np.column_stack([np.sin(t), np.cos(t)])
        b = np.column_stack([np.sin(t + 0.2), np.cos(t + 0.2)])
        result = cross_rqa(a, b, epsilon=0.3)
        assert isinstance(result, RQAResult)
        assert result.recurrence_rate > 0

    def test_uncorrelated_low_crqa(self):
        """Uncorrelated trajectories → low cross-determinism."""
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, (80, 2))
        b = rng.normal(0, 1, (80, 2))
        result = cross_rqa(a, b, epsilon=0.3)
        assert result.determinism < 0.5

    def test_cross_recurrence_angular_metric(self):
        """Angular metric in cross_recurrence_matrix."""
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 2 * np.pi, (20, 2))
        b = rng.uniform(0, 2 * np.pi, (20, 2))
        CR = cross_recurrence_matrix(a, b, epsilon=1.0, metric="angular")
        assert CR.shape == (20, 20)
        assert CR.dtype == bool

    def test_cross_rqa_angular_metric(self):
        """Angular metric in cross_rqa."""
        t = np.linspace(0, 4 * np.pi, 50)
        a = np.column_stack([np.sin(t), np.cos(t)])
        b = np.column_stack([np.sin(t + 0.1), np.cos(t + 0.1)])
        result = cross_rqa(a, b, epsilon=1.0, metric="angular")
        assert isinstance(result, RQAResult)
        assert 0.0 <= result.recurrence_rate <= 1.0


class TestRecurrencePipelineWiring:
    """Pipeline: engine trajectory → delay embed → RQA → determinism."""

    def test_engine_trajectory_to_rqa(self):
        """UPDEEngine → trajectory → delay_embed → rqa: determinism
        quantifies recurrence in coupled oscillator dynamics."""
        from scpn_phase_orchestrator.monitor.embedding import delay_embed
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.array([1.0, 1.5, 2.0, 0.5])
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        trajectory = []
        for _ in range(300):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(float(phases[0]))
        signal = np.array(trajectory)

        emb = delay_embed(signal, delay=5, dimension=3)
        result = rqa(emb, epsilon=0.3)
        assert isinstance(result, RQAResult)
        assert 0.0 <= result.recurrence_rate <= 1.0
        assert 0.0 <= result.determinism <= 1.0
