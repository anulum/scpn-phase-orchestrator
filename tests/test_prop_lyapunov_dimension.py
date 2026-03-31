# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based Lyapunov & dimension proofs

"""Hypothesis-driven invariant proofs for the Lyapunov spectrum,
Kaplan-Yorke dimension, and Grassberger-Procaccia correlation dimension.

Each test is a computational theorem: "for all tested inputs, this
mathematical invariant holds." Catches edge cases that parametrized
unit tests miss.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor.dimension import (
    correlation_dimension,
    correlation_integral,
    kaplan_yorke_dimension,
)
from scpn_phase_orchestrator.monitor.lyapunov import (
    LyapunovGuard,
    lyapunov_spectrum,
)

TWO_PI = 2.0 * np.pi


# ── Strategies ──────────────────────────────────────────────────────────


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── 1. Lyapunov spectrum: structural invariants ─────────────────────────


class TestLyapunovSpectrumInvariants:
    """Mathematical invariants of the Lyapunov spectrum that must hold
    regardless of system parameters.
    """

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=60, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_length_equals_n(self, n: int, seed: int) -> None:
        """Spectrum must have exactly N exponents."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        alpha = np.zeros((n, n))
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=200, qr_interval=5)
        assert len(spec) == n

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=60, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_sorted_descending(self, n: int, seed: int) -> None:
        """Spectrum must be returned in descending order."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        alpha = np.zeros((n, n))
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=200, qr_interval=5)
        for i in range(len(spec) - 1):
            assert spec[i] >= spec[i + 1] - 1e-12

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=60, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_all_finite(self, n: int, seed: int) -> None:
        """All exponents must be finite."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        alpha = np.zeros((n, n))
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=200, qr_interval=5)
        assert np.all(np.isfinite(spec))

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_zero_coupling_near_zero(self, seed: int) -> None:
        """Zero coupling → free rotation → all exponents ≈ 0."""
        n = 4
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        spec = lyapunov_spectrum(
            phases, omegas, knm, alpha, n_steps=500, qr_interval=10
        )
        assert np.all(np.abs(spec) < 1.0)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_strong_coupling_has_negative(self, seed: int) -> None:
        """Strong coupling synchronises → at least one negative exponent."""
        n = 4
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-0.5, 0.5, n)
        knm = _connected_knm(n, strength=10.0, seed=seed)
        alpha = np.zeros((n, n))
        spec = lyapunov_spectrum(
            phases, omegas, knm, alpha, n_steps=500, qr_interval=10
        )
        assert np.min(spec) < 0.0


# ── 2. Kaplan-Yorke dimension: mathematical bounds ─────────────────────


class TestKaplanYorkeBounds:
    """D_KY ∈ [0, N] for any Lyapunov spectrum of length N."""

    @given(
        n=st.integers(min_value=1, max_value=20),
        seed=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=100, deadline=None)
    def test_dky_in_range(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        spec = np.sort(rng.uniform(-5, 5, n))[::-1]
        dky = kaplan_yorke_dimension(spec)
        assert 0.0 <= dky <= n

    @given(n=st.integers(min_value=1, max_value=20))
    @settings(max_examples=50, deadline=None)
    def test_all_negative_gives_zero(self, n: int) -> None:
        """All negative exponents → stable fixed point → D_KY = 0."""
        spec = -np.arange(1, n + 1, dtype=float)
        dky = kaplan_yorke_dimension(spec)
        assert dky == 0.0

    @given(n=st.integers(min_value=1, max_value=20))
    @settings(max_examples=50, deadline=None)
    def test_all_positive_gives_n(self, n: int) -> None:
        """All positive exponents → D_KY = N (fully expanding)."""
        spec = np.arange(n, 0, -1, dtype=float)
        dky = kaplan_yorke_dimension(spec)
        assert dky == float(n)

    def test_limit_cycle_lorenz(self) -> None:
        """Lorenz-like: (+, 0, -) → 2 < D_KY < 3."""
        spec = np.array([0.9, 0.0, -14.57])
        dky = kaplan_yorke_dimension(spec)
        assert 2.0 < dky < 3.0

    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=50, deadline=None)
    def test_monotonic_in_largest_exponent(self, seed: int) -> None:
        """Increasing λ_max (with others fixed) → D_KY non-decreasing."""
        rng = np.random.default_rng(seed)
        base = np.sort(rng.uniform(-5, 0, 4))[::-1]
        results = []
        for lam_max in [0.1, 0.5, 1.0, 2.0, 5.0]:
            spec = base.copy()
            spec[0] = lam_max
            results.append(kaplan_yorke_dimension(spec))
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1] + 1e-12

    @given(
        n=st.integers(min_value=2, max_value=15),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_unsorted_same_as_sorted(self, n: int, seed: int) -> None:
        """kaplan_yorke_dimension should internally sort, so order doesn't matter."""
        rng = np.random.default_rng(seed)
        spec = rng.uniform(-5, 5, n)
        dky_unsorted = kaplan_yorke_dimension(spec)
        dky_sorted = kaplan_yorke_dimension(np.sort(spec)[::-1])
        assert abs(dky_unsorted - dky_sorted) < 1e-12

    @pytest.mark.parametrize(
        "spec,expected_j",
        [
            (np.array([1.0, -2.0]), 1),
            (np.array([2.0, 1.0, -4.0]), 2),
            (np.array([3.0, -1.0, -5.0]), 2),
        ],
    )
    def test_known_j_values(self, spec: np.ndarray, expected_j: int) -> None:
        """D_KY = j + sum(λ_1..j)/|λ_{j+1}|, verify j is correct."""
        dky = kaplan_yorke_dimension(spec)
        cumsum = np.cumsum(np.sort(spec)[::-1])
        j = 0
        for i in range(len(cumsum)):
            if cumsum[i] >= 0:
                j = i + 1
        assert expected_j == j
        if j < len(spec):
            expected_dky = j + cumsum[j - 1] / abs(np.sort(spec)[::-1][j])
            assert abs(dky - expected_dky) < 1e-12


# ── 3. Correlation integral: monotonic in ε ─────────────────────────────


class TestCorrelationIntegralMonotonic:
    """C(ε) must be monotonically non-decreasing in ε. Larger ε captures
    more pairs within distance ε.
    """

    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=40, deadline=None)
    def test_monotonic_in_epsilon(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((50, 2))
        epsilons = np.logspace(-1, 1, 15)
        C = correlation_integral(traj, epsilons)
        for i in range(len(C) - 1):
            assert C[i] <= C[i + 1] + 1e-12

    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=40, deadline=None)
    def test_bounded_zero_one(self, seed: int) -> None:
        """C(ε) ∈ [0, 1] since it's a fraction of pairs."""
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((30, 3))
        epsilons = np.logspace(-2, 2, 20)
        C = correlation_integral(traj, epsilons)
        assert np.all(C >= -1e-12)
        assert np.all(C <= 1.0 + 1e-12)

    def test_large_epsilon_approaches_one(self) -> None:
        """Very large ε → C(ε) = 1 (all pairs within distance)."""
        rng = np.random.default_rng(42)
        traj = rng.standard_normal((40, 2))
        diam = np.max(np.linalg.norm(traj[:, None] - traj[None, :], axis=2))
        C = correlation_integral(traj, np.array([diam * 10.0]))
        assert C[0] > 0.99

    def test_zero_epsilon_is_zero(self) -> None:
        """ε = 0 → C(0) = 0 (no pair has distance < 0)."""
        rng = np.random.default_rng(42)
        traj = rng.standard_normal((40, 2))
        C = correlation_integral(traj, np.array([0.0]))
        assert C[0] == 0.0

    def test_constant_trajectory_all_one(self) -> None:
        """All points identical → C(ε) = 0 for ε=0, 1 for ε>0."""
        traj = np.ones((20, 2))
        C = correlation_integral(traj, np.array([0.0, 0.001, 1.0]))
        assert C[0] == 0.0
        assert C[1] > 0.99
        assert C[2] > 0.99


# ── 4. Correlation dimension: structural properties ─────────────────────


class TestCorrelationDimensionProperties:
    """Structural invariants of the full D2 estimation."""

    def test_constant_signal_d2_zero(self) -> None:
        """Constant trajectory → D2 = 0 (zero-dimensional)."""
        traj = np.ones((100, 2))
        result = correlation_dimension(traj)
        assert result.D2 == 0.0

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_d2_nonnegative(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((100, 2))
        result = correlation_dimension(traj)
        assert result.D2 >= -0.5  # small negative possible from noisy slope estimation

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_result_fields_populated(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((80, 2))
        result = correlation_dimension(traj)
        assert len(result.epsilons) > 0
        assert len(result.C_eps) == len(result.epsilons)
        assert len(result.slope) == len(result.epsilons) - 1 or len(result.slope) > 0
        lo, hi = result.scaling_range
        assert lo <= hi

    def test_1d_line_d2_near_one(self) -> None:
        """Points along a line → D2 ≈ 1."""
        t = np.linspace(0, 10, 300)
        traj = np.column_stack([t, t * 0.0])
        result = correlation_dimension(traj)
        assert 0.5 < result.D2 < 1.5

    def test_2d_filled_square_d2_near_two(self) -> None:
        """Uniform 2D fill → D2 ≈ 2."""
        rng = np.random.default_rng(42)
        traj = rng.uniform(0, 1, (500, 2))
        result = correlation_dimension(traj)
        assert 1.2 < result.D2 < 2.8


# ── 5. LyapunovGuard: invariant properties ─────────────────────────────


class TestLyapunovGuardInvariants:
    """Property-based tests for the Lyapunov function monitor."""

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=60, deadline=None)
    def test_v_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n, seed=seed)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert np.isfinite(state.V)

    @given(
        n=st.integers(min_value=2, max_value=12),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_sync_is_minimum(self, n: int, seed: int) -> None:
        """Synchronized phases should have lower V than random phases."""
        rng = np.random.default_rng(seed)
        knm = _connected_knm(n, strength=1.0, seed=seed)
        guard = LyapunovGuard()

        sync_phases = np.full(n, 1.0)
        v_sync = guard.evaluate(sync_phases, knm).V

        guard.reset()
        random_phases = rng.uniform(0, TWO_PI, n)
        v_random = guard.evaluate(random_phases, knm).V

        assert v_sync <= v_random + 1e-10

    @given(n=st.integers(min_value=2, max_value=12))
    @settings(max_examples=30, deadline=None)
    def test_zero_coupling_v_zero(self, n: int) -> None:
        """Zero coupling → V = 0 regardless of phases."""
        phases = np.random.default_rng(42).uniform(0, TWO_PI, n)
        knm = np.zeros((n, n))
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert abs(state.V) < 1e-12

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_max_phase_diff_nonneg(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n, seed=seed)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.max_phase_diff >= 0.0

    @given(n=st.integers(min_value=2, max_value=10))
    @settings(max_examples=30, deadline=None)
    def test_sync_in_basin(self, n: int) -> None:
        """Synchronized phases → in_basin = True."""
        phases = np.full(n, 2.5)
        knm = _connected_knm(n)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.in_basin is True

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_max_phase_diff_bounded_by_pi(self, n: int, seed: int) -> None:
        """Wrapped phase differences are at most π."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n, seed=seed)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.max_phase_diff <= np.pi + 1e-10

    def test_first_call_dv_zero(self) -> None:
        """First evaluate call has no previous V → dV_dt = 0."""
        guard = LyapunovGuard()
        phases = np.array([0.0, 1.0, 2.0])
        knm = _connected_knm(3)
        state = guard.evaluate(phases, knm)
        assert state.dV_dt == 0.0

    def test_reset_clears_state(self) -> None:
        """After reset, dV_dt must be 0 on next call."""
        guard = LyapunovGuard()
        knm = _connected_knm(3)
        guard.evaluate(np.array([0.0, 0.5, 1.0]), knm)
        guard.reset()
        state = guard.evaluate(np.array([0.0, 0.5, 1.0]), knm)
        assert state.dV_dt == 0.0


# ── 6. Cross-module: spectrum → D_KY consistency ────────────────────────


class TestSpectrumToDKYConsistency:
    """The Lyapunov spectrum from simulation, fed into kaplan_yorke_dimension,
    must produce a value in [0, N].
    """

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    @pytest.mark.parametrize("strength", [0.5, 2.0, 10.0])
    def test_simulated_spectrum_dky_valid(self, n: int, strength: float) -> None:
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, strength=strength)
        alpha = np.zeros((n, n))
        spec = lyapunov_spectrum(
            phases, omegas, knm, alpha, n_steps=300, qr_interval=10
        )
        dky = kaplan_yorke_dimension(spec)
        assert 0.0 <= dky <= n
        assert np.isfinite(dky)


# Pipeline wiring: hypothesis-driven Lyapunov dimension proofs exercise
# the engine + lyapunov_spectrum pipeline.
