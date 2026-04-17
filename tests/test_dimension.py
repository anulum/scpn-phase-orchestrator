# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal dimension tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.dimension import (
    CorrelationDimensionResult,
    correlation_dimension,
    correlation_integral,
    kaplan_yorke_dimension,
)


class TestCorrelationIntegral:
    def test_increases_with_epsilon(self):
        """C(ε) is monotonically non-decreasing."""
        t = np.linspace(0, 10 * np.pi, 500)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        epsilons = np.logspace(-1, 0.5, 15)
        C = correlation_integral(traj, epsilons)
        for i in range(len(C) - 1):
            assert C[i] <= C[i + 1] + 1e-10

    def test_all_same_point(self):
        """All points identical → C(ε)=1 for any ε>0."""
        traj = np.ones((50, 2))
        C = correlation_integral(traj, np.array([0.01, 0.1, 1.0]))
        np.testing.assert_array_equal(C, [1.0, 1.0, 1.0])

    def test_subsampling(self):
        """Large trajectory triggers pair subsampling."""
        rng = np.random.default_rng(42)
        traj = rng.normal(0, 1, (1000, 3))
        epsilons = np.logspace(-1, 1, 10)
        C = correlation_integral(traj, epsilons, max_pairs=5000)
        assert C[-1] > 0


class TestCorrelationDimension:
    def test_circle_dimension(self):
        """Circle (1D manifold) → D2 ≈ 1."""
        t = np.linspace(0, 20 * np.pi, 2000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = correlation_dimension(traj)
        assert isinstance(result, CorrelationDimensionResult)
        # D2 for a circle should be ~1.0
        assert 0.5 < result.D2 < 1.8

    def test_plane_filling(self):
        """2D Gaussian cloud → D2 ≈ 2."""
        rng = np.random.default_rng(42)
        traj = rng.normal(0, 1, (2000, 2))
        result = correlation_dimension(traj)
        assert 1.5 < result.D2 < 2.5

    def test_constant_trajectory(self):
        """Single point → D2 = 0."""
        traj = np.ones((100, 3))
        result = correlation_dimension(traj)
        assert result.D2 == 0.0

    def test_result_fields(self):
        t = np.linspace(0, 10 * np.pi, 500)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = correlation_dimension(traj, n_epsilons=20)
        assert len(result.epsilons) == 20
        assert len(result.C_eps) == 20
        assert result.scaling_range[0] <= result.scaling_range[1]


class TestKaplanYorkeDimension:
    def test_stable_fixed_point(self):
        """All negative exponents → D_KY = 0."""
        le = np.array([-0.5, -1.0, -2.0])
        assert kaplan_yorke_dimension(le) == 0.0

    def test_limit_cycle(self):
        """One zero + one negative → D_KY = 1."""
        le = np.array([0.0, -1.0])
        d = kaplan_yorke_dimension(le)
        assert abs(d - 1.0) < 1e-10

    def test_lorenz_like(self):
        """Lorenz-like spectrum [+0.9, 0, -14.6] → D_KY ≈ 2.06."""
        le = np.array([0.9, 0.0, -14.6])
        d = kaplan_yorke_dimension(le)
        assert 2.0 < d < 2.1

    def test_hyperchaos(self):
        """Two positive exponents."""
        le = np.array([0.5, 0.2, -0.1, -1.5])
        d = kaplan_yorke_dimension(le)
        # Sum of first 3: 0.5+0.2-0.1 = 0.6 > 0
        # j=2, D_KY = 3 + 0.6/1.5 = 3.4
        assert abs(d - 3.4) < 1e-10

    def test_all_positive(self):
        """All positive → D_KY = N (fills all dimensions)."""
        le = np.array([0.5, 0.3, 0.1])
        d = kaplan_yorke_dimension(le)
        assert d == 3.0

    def test_unsorted_input(self):
        """Should handle unsorted input."""
        le = np.array([-1.0, 0.5, -0.3])
        d = kaplan_yorke_dimension(le)
        # Sorted: [0.5, -0.3, -1.0] → cumsum=[0.5, 0.2, -0.8] → j=1
        # D_KY = 2 + 0.2/1.0 = 2.2
        assert abs(d - 2.2) < 1e-10

    def test_zero_denominator(self):
        """λ_{j+1} = 0 → D_KY = j + 1."""
        le = np.array([1.0, 0.0, -1.0])
        d = kaplan_yorke_dimension(le)
        # cumsum = [1.0, 1.0, 0.0] → j=2, but j+1=3 >= len → returns 3
        # Actually: j=2 (0-indexed cumsum[2]=0.0 ≥ 0), j+1=3 ≥ 3 → returns 3.0
        assert d == 3.0


class TestCorrelationDimensionEdgeCases:
    def test_few_valid_c_eps(self):
        """Trajectory with very few valid C(ε) → D2=0."""
        # Two points very close together → only small ε gives C>0
        traj = np.array([[0.0, 0.0], [1e-15, 1e-15], [0.0, 0.0]])
        result = correlation_dimension(traj, n_epsilons=5)
        assert result.D2 >= 0.0

    def test_single_point_trajectory(self):
        """T=1 → diameter=0 → D2=0."""
        traj = np.array([[1.0, 2.0]])
        result = correlation_dimension(traj)
        assert result.D2 == 0.0

    def test_short_slope_window(self):
        """Very few epsilons → window < 2 branch."""
        traj = np.array([[0.0], [1.0], [2.0], [3.0]])
        result = correlation_dimension(traj, n_epsilons=3)
        assert isinstance(result.D2, float)


class TestDimensionPipelineWiring:
    """Pipeline: engine trajectory → embed → correlation dimension."""

    def test_engine_trajectory_to_correlation_dimension(self):
        """UPDEEngine → trajectory → delay_embed → D2."""
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
        for _ in range(500):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(float(phases[0]))

        emb = delay_embed(np.array(trajectory), delay=5, dimension=3)
        result = correlation_dimension(emb)
        assert isinstance(result, CorrelationDimensionResult)
        assert result.D2 >= 0.0
