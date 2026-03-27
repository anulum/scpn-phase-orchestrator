# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based recurrence & RQA proofs

"""Hypothesis-driven invariant proofs for recurrence matrices and
Recurrence Quantification Analysis (RQA).

Mathematical invariants from Eckmann et al. 1987, Marwan et al. 2007.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor.recurrence import (
    cross_recurrence_matrix,
    cross_rqa,
    recurrence_matrix,
    rqa,
)

TWO_PI = 2.0 * np.pi


# ── 1. Recurrence matrix ────────────────────────────────────────────────


class TestRecurrenceMatrixInvariants:
    """R_ij = Θ(ε - ||x_i - x_j||): symmetric, diagonal=True, boolean."""

    @given(
        t=st.integers(min_value=3, max_value=30),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_symmetric(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        R = recurrence_matrix(traj, epsilon=1.0)
        np.testing.assert_array_equal(R, R.T)

    @given(
        t=st.integers(min_value=3, max_value=30),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_diagonal_true(self, t: int, seed: int) -> None:
        """Every point is recurrent with itself: R_ii = True."""
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        R = recurrence_matrix(traj, epsilon=1.0)
        assert np.all(np.diag(R))

    @given(
        t=st.integers(min_value=3, max_value=30),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_boolean_dtype(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        R = recurrence_matrix(traj, epsilon=1.0)
        assert R.dtype == bool

    @given(
        t=st.integers(min_value=3, max_value=20),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_shape_t_by_t(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        R = recurrence_matrix(traj, epsilon=1.0)
        assert R.shape == (t, t)

    def test_epsilon_zero_only_diagonal(self) -> None:
        """ε=0 → only identical points recur (just diagonal for distinct points)."""
        traj = np.array([[0.0], [1.0], [2.0], [3.0]])
        R = recurrence_matrix(traj, epsilon=0.0)
        np.testing.assert_array_equal(R, np.eye(4, dtype=bool))

    def test_large_epsilon_all_true(self) -> None:
        """Very large ε → all points within threshold."""
        rng = np.random.default_rng(42)
        traj = rng.standard_normal((10, 2))
        R = recurrence_matrix(traj, epsilon=1e6)
        assert np.all(R)

    def test_constant_trajectory_all_true(self) -> None:
        traj = np.ones((10, 2))
        R = recurrence_matrix(traj, epsilon=0.1)
        assert np.all(R)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_1d_input(self, seed: int) -> None:
        """1D trajectory (single oscillator) should work."""
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal(20)
        R = recurrence_matrix(traj, epsilon=1.0)
        assert R.shape == (20, 20)
        assert np.all(np.diag(R))

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_angular_metric_symmetric(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.uniform(0, TWO_PI, (15, 2))
        R = recurrence_matrix(traj, epsilon=1.0, metric="angular")
        np.testing.assert_array_equal(R, R.T)


# ── 2. RQA measures: bounded invariants ──────────────────────────────────


class TestRQABounds:
    """RR, DET, LAM ∈ [0, 1]. Entropy ≥ 0. Integer measures ≥ 0."""

    @given(
        t=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_rr_in_unit(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        result = rqa(traj, epsilon=1.0)
        assert 0.0 <= result.recurrence_rate <= 1.0

    @given(
        t=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_det_in_unit(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        result = rqa(traj, epsilon=1.0)
        assert 0.0 <= result.determinism <= 1.0

    @given(
        t=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_lam_in_unit(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        result = rqa(traj, epsilon=1.0)
        assert 0.0 <= result.laminarity <= 1.0

    @given(
        t=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_entropy_nonneg(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        result = rqa(traj, epsilon=1.0)
        assert result.entropy_diagonal >= -1e-12

    @given(
        t=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_max_diagonal_nonneg(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        result = rqa(traj, epsilon=1.0)
        assert result.max_diagonal >= 0

    @given(
        t=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_max_vertical_nonneg(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        result = rqa(traj, epsilon=1.0)
        assert result.max_vertical >= 0

    @given(
        t=st.integers(min_value=5, max_value=30),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_trapping_time_nonneg(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.standard_normal((t, 2))
        result = rqa(traj, epsilon=1.0)
        assert result.trapping_time >= 0.0


# ── 3. RQA limiting cases ───────────────────────────────────────────────


class TestRQALimits:

    def test_constant_trajectory_rr_high(self) -> None:
        """All points identical → RR = 1 (after diagonal exclusion, all recur)."""
        traj = np.ones((10, 2))
        result = rqa(traj, epsilon=0.1)
        assert result.recurrence_rate > 0.99

    def test_epsilon_zero_rr_zero(self) -> None:
        """ε=0 with distinct points → RR = 0 (diagonal excluded from RR)."""
        traj = np.arange(20, dtype=float).reshape(20, 1)
        result = rqa(traj, epsilon=0.0)
        assert result.recurrence_rate == 0.0

    def test_large_epsilon_rr_one(self) -> None:
        rng = np.random.default_rng(42)
        traj = rng.standard_normal((15, 2))
        result = rqa(traj, epsilon=1e6)
        assert result.recurrence_rate > 0.99


# ── 4. Cross-recurrence matrix ───────────────────────────────────────────


class TestCrossRecurrenceInvariants:

    @given(
        t=st.integers(min_value=3, max_value=20),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_shape(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((t, 2))
        b = rng.standard_normal((t, 2))
        CR = cross_recurrence_matrix(a, b, epsilon=1.0)
        assert CR.shape == (t, t)

    @given(
        t=st.integers(min_value=3, max_value=20),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_boolean_dtype(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((t, 2))
        b = rng.standard_normal((t, 2))
        CR = cross_recurrence_matrix(a, b, epsilon=1.0)
        assert CR.dtype == bool

    def test_identical_trajectories_equals_recurrence(self) -> None:
        """CR(A, A) should match R(A) on the diagonal."""
        rng = np.random.default_rng(42)
        traj = rng.standard_normal((10, 2))
        R = recurrence_matrix(traj, epsilon=1.0)
        CR = cross_recurrence_matrix(traj, traj, epsilon=1.0)
        np.testing.assert_array_equal(R, CR)

    def test_large_epsilon_all_true(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal((8, 2))
        b = rng.standard_normal((8, 2))
        CR = cross_recurrence_matrix(a, b, epsilon=1e6)
        assert np.all(CR)


# ── 5. Cross-RQA bounds ─────────────────────────────────────────────────


class TestCrossRQABounds:

    @given(
        t=st.integers(min_value=5, max_value=20),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_rr_bounded(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((t, 2))
        b = rng.standard_normal((t, 2))
        result = cross_rqa(a, b, epsilon=1.0)
        assert 0.0 <= result.recurrence_rate <= 1.0

    @given(
        t=st.integers(min_value=5, max_value=20),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_det_bounded(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((t, 2))
        b = rng.standard_normal((t, 2))
        result = cross_rqa(a, b, epsilon=1.0)
        assert 0.0 <= result.determinism <= 1.0

    @given(
        t=st.integers(min_value=5, max_value=20),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_lam_bounded(self, t: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((t, 2))
        b = rng.standard_normal((t, 2))
        result = cross_rqa(a, b, epsilon=1.0)
        assert 0.0 <= result.laminarity <= 1.0
