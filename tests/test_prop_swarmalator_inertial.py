# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based swarmalator & inertial proofs

"""Hypothesis-driven invariant proofs for the Swarmalator and
second-order Inertial Kuramoto (swing equation) engines.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── 1. Swarmalator ──────────────────────────────────────────────────────


class TestSwarmalatorInvariants:
    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_output_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        eng = SwarmalatorEngine(n, dim=2, dt=0.01)
        phases = rng.uniform(0, TWO_PI, n)
        positions = rng.standard_normal((n, 2))
        omegas = rng.uniform(-1, 1, n)
        new_pos, new_phases = eng.step(positions, phases, omegas)
        assert np.all(np.isfinite(new_phases))
        assert np.all(np.isfinite(new_pos))

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_output_shapes(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        eng = SwarmalatorEngine(n, dim=2, dt=0.01)
        phases = rng.uniform(0, TWO_PI, n)
        positions = rng.standard_normal((n, 2))
        omegas = rng.uniform(-1, 1, n)
        new_pos, new_phases = eng.step(positions, phases, omegas)
        assert new_phases.shape == (n,)
        assert new_pos.shape == (n, 2)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_j_zero_phase_independent_of_position(self, seed: int) -> None:
        """J=0 → phase dynamics don't depend on spatial positions."""
        n = 4
        rng = np.random.default_rng(seed)
        eng = SwarmalatorEngine(n, dim=2, dt=0.01, J=0.0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-1, 1, n)
        pos1 = rng.standard_normal((n, 2))
        pos2 = pos1 + 10.0
        _, p1 = eng.step(pos1, phases.copy(), omegas)
        eng2 = SwarmalatorEngine(n, dim=2, dt=0.01, J=0.0)
        _, p2 = eng2.step(pos2, phases.copy(), omegas)
        np.testing.assert_allclose(p1, p2, atol=1e-10)


# ── 2. Inertial Kuramoto (Swing Equation) ───────────────────────────────


class TestInertialKuramotoInvariants:
    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_output_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        eng = InertialKuramotoEngine(n, dt=0.01)
        theta = rng.uniform(0, TWO_PI, n)
        omega_dot = rng.uniform(-1, 1, n)
        power = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        inertia = np.ones(n)
        damping = np.full(n, 0.5)
        new_theta, new_omega = eng.step(theta, omega_dot, power, knm, inertia, damping)
        assert np.all(np.isfinite(new_theta))
        assert np.all(np.isfinite(new_omega))

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_output_shapes(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        eng = InertialKuramotoEngine(n, dt=0.01)
        theta = rng.uniform(0, TWO_PI, n)
        omega_dot = np.zeros(n)
        power = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        inertia = np.ones(n)
        damping = np.full(n, 0.5)
        new_theta, new_omega = eng.step(theta, omega_dot, power, knm, inertia, damping)
        assert new_theta.shape == (n,)
        assert new_omega.shape == (n,)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_zero_coupling_evolves(self, seed: int) -> None:
        """K=0 → free swing: angles advance under power/damping."""
        n = 3
        rng = np.random.default_rng(seed)
        eng = InertialKuramotoEngine(n, dt=0.01)
        theta = rng.uniform(0, TWO_PI, n)
        omega_dot = rng.uniform(-2, 2, n)
        power = rng.uniform(-1, 1, n)
        knm = np.zeros((n, n))
        inertia = np.ones(n)
        damping = np.full(n, 0.1)
        new_theta, _ = eng.step(theta, omega_dot, power, knm, inertia, damping)
        assert not np.allclose(new_theta % TWO_PI, theta % TWO_PI, atol=1e-6)

    @pytest.mark.parametrize("damp_val", [0.1, 0.5, 1.0, 5.0])
    def test_theta_wrapped_to_twopi(self, damp_val: float) -> None:
        """Output theta should be in [0, 2π)."""
        n = 4
        eng = InertialKuramotoEngine(n, dt=0.01)
        theta = np.array([0.1, 3.0, 5.5, 6.2])
        omega_dot = np.ones(n) * 5.0
        power = np.zeros(n)
        knm = np.zeros((n, n))
        inertia = np.ones(n)
        damping = np.full(n, damp_val)
        new_theta, _ = eng.step(theta, omega_dot, power, knm, inertia, damping)
        assert np.all(new_theta >= 0.0)
        assert np.all(new_theta < TWO_PI + 1e-10)


# Pipeline wiring: swarmalator + inertial property tests use engine variants.
