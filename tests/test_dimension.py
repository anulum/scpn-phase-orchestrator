# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
        # Sorted: [0.5, -0.3, -1.0] → j=0, D_KY=1+0.5/0.3=2.67
        assert abs(d - (1 + 0.5 / 0.3)) < 1e-10
