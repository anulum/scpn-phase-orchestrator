# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for second-order inertial Kuramoto

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine

N = 4
DT = 0.01


@pytest.fixture()
def engine():
    return InertialKuramotoEngine(N, dt=DT)


@pytest.fixture()
def balanced_grid():
    """Simple balanced 4-bus power grid."""
    theta = np.zeros(N)
    omega_dot = np.zeros(N)
    # Two generators, two loads, balanced
    power = np.array([1.0, 1.0, -1.0, -1.0])
    knm = np.ones((N, N)) * 2.0
    np.fill_diagonal(knm, 0.0)
    inertia = np.ones(N) * 5.0
    damping = np.ones(N) * 1.0
    return theta, omega_dot, power, knm, inertia, damping


class TestInertialStep:
    def test_output_shapes(self, engine, balanced_grid):
        theta, omega_dot, power, knm, inertia, damping = balanced_grid
        new_th, new_od = engine.step(theta, omega_dot, power, knm, inertia, damping)
        assert new_th.shape == (N,)
        assert new_od.shape == (N,)

    def test_theta_in_range(self, engine, balanced_grid):
        theta, omega_dot, power, knm, inertia, damping = balanced_grid
        new_th, _ = engine.step(theta, omega_dot, power, knm, inertia, damping)
        assert np.all(new_th >= 0.0)
        assert np.all(new_th < 2.0 * np.pi)

    def test_zero_power_zero_coupling_no_acceleration(self, engine):
        theta = np.random.rand(N) * 2 * np.pi
        omega_dot = np.zeros(N)
        power = np.zeros(N)
        knm = np.zeros((N, N))
        inertia = np.ones(N)
        damping = np.ones(N)
        _, new_od = engine.step(theta, omega_dot, power, knm, inertia, damping)
        np.testing.assert_allclose(new_od, 0.0, atol=1e-10)

    def test_damping_reduces_velocity(self, engine):
        theta = np.zeros(N)
        omega_dot = np.ones(N) * 1.0
        power = np.zeros(N)
        knm = np.zeros((N, N))
        inertia = np.ones(N)
        damping = np.ones(N) * 2.0
        _, new_od = engine.step(theta, omega_dot, power, knm, inertia, damping)
        assert np.all(np.abs(new_od) < np.abs(omega_dot))


class TestInertialRun:
    def test_trajectory_shapes(self, engine, balanced_grid):
        theta, omega_dot, power, knm, inertia, damping = balanced_grid
        ft, fo, tt, to_ = engine.run(
            theta, omega_dot, power, knm, inertia, damping, 100
        )
        assert ft.shape == (N,)
        assert fo.shape == (N,)
        assert tt.shape == (100, N)
        assert to_.shape == (100, N)

    def test_balanced_grid_stays_synchronized(self, engine, balanced_grid):
        theta, omega_dot, power, knm, inertia, damping = balanced_grid
        ft, fo, _, _ = engine.run(theta, omega_dot, power, knm, inertia, damping, 500)
        R = engine.coherence(ft)
        assert R > 0.9

    def test_finite_values(self, engine, balanced_grid):
        theta, omega_dot, power, knm, inertia, damping = balanced_grid
        ft, fo, tt, to_ = engine.run(
            theta, omega_dot, power, knm, inertia, damping, 200
        )
        assert np.all(np.isfinite(ft))
        assert np.all(np.isfinite(fo))
        assert np.all(np.isfinite(tt))
        assert np.all(np.isfinite(to_))


class TestInertialMetrics:
    def test_frequency_deviation(self, engine):
        omega_dot = np.array([0.1, -0.2, 0.05, -0.15])
        dev = engine.frequency_deviation(omega_dot)
        expected = 0.2 / (2.0 * np.pi)
        assert abs(dev - expected) < 1e-10

    def test_coherence_perfect_sync(self, engine):
        theta = np.ones(N) * 1.5
        R = engine.coherence(theta)
        assert abs(R - 1.0) < 1e-10

    def test_coherence_incoherent(self, engine):
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        R = engine.coherence(theta)
        assert R < 0.2


class TestPowerGridScenarios:
    def test_generator_trip_causes_deviation(self, engine, balanced_grid):
        """Losing a generator increases frequency deviation."""
        theta, omega_dot, power, knm, inertia, damping = balanced_grid
        # Baseline
        _, fo_base, _, _ = engine.run(
            theta, omega_dot, power, knm, inertia, damping, 200
        )
        dev_base = engine.frequency_deviation(fo_base)
        # Trip generator 0
        power_trip = power.copy()
        power_trip[0] = 0.0
        _, fo_trip, _, _ = engine.run(
            theta, omega_dot, power_trip, knm, inertia, damping, 200
        )
        dev_trip = engine.frequency_deviation(fo_trip)
        assert dev_trip > dev_base

    def test_weak_coupling_desynchronizes(self, engine, balanced_grid):
        """Weak transmission lines lead to desynchronization."""
        theta, omega_dot, power, knm, inertia, damping = balanced_grid
        # Perturb initial angles
        theta_perturbed = theta + np.array([0, 0.5, 1.0, 1.5])
        # Very weak coupling
        knm_weak = knm * 0.01
        ft, _, _, _ = engine.run(
            theta_perturbed, omega_dot, power, knm_weak, inertia, damping, 500
        )
        R = engine.coherence(ft)
        # Should lose coherence
        assert R < 0.9


class TestInertialPipelineWiring:
    """Pipeline: InertialKuramotoEngine → coherence → R."""

    def test_inertial_engine_to_coherence(self):
        """InertialKuramotoEngine.run → coherence R∈[0,1]."""
        n = 4
        eng = InertialKuramotoEngine(n, dt=0.01)
        theta = np.zeros(n)
        omega_dot = np.zeros(n)
        power = np.array([1.0, -0.5, 0.3, -0.8])
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        inertia = np.ones(n) * 2.0
        damping = np.ones(n) * 0.5
        ft, _, _, _ = eng.run(
            theta,
            omega_dot,
            power,
            knm,
            inertia,
            damping,
            200,
        )
        R = eng.coherence(ft)
        assert 0.0 <= R <= 1.0
