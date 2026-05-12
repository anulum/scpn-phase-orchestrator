# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for recurrence analysis

"""Algorithmic properties of :func:`recurrence_matrix`,
:func:`cross_recurrence_matrix`, :func:`rqa`, :func:`cross_rqa`.

Covered: self-recurrence on the main diagonal, symmetry, threshold
monotonicity, angular vs euclidean metric on circular data, edge
cases (T=1, 1-D input), cross-recurrence sanity, RQA positivity +
boundedness, Hypothesis coverage.
"""

from __future__ import annotations

import functools
import math
import sys
import types

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import recurrence as r_mod
from scpn_phase_orchestrator.monitor.recurrence import (
    cross_recurrence_matrix,
    cross_rqa,
    recurrence_matrix,
    rqa,
)

TWO_PI = 2.0 * math.pi


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = r_mod.ACTIVE_BACKEND
        r_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            r_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestRecurrenceMatrix:
    @_python
    def test_main_diagonal_always_true(self):
        rng = np.random.default_rng(0)
        traj = rng.normal(0, 1, (20, 3))
        R = recurrence_matrix(traj, 0.3)
        assert np.all(np.diag(R))

    @_python
    def test_symmetric(self):
        rng = np.random.default_rng(1)
        traj = rng.normal(0, 1, (20, 3))
        R = recurrence_matrix(traj, 0.5)
        np.testing.assert_array_equal(R, R.T)

    @_python
    def test_monotone_in_epsilon(self):
        rng = np.random.default_rng(2)
        traj = rng.normal(0, 1, (25, 3))
        R_small = recurrence_matrix(traj, 0.3)
        R_big = recurrence_matrix(traj, 3.0)
        assert int(R_big.sum()) >= int(R_small.sum())

    @_python
    def test_large_epsilon_fully_recurrent(self):
        rng = np.random.default_rng(3)
        traj = rng.normal(0, 1, (15, 3))
        R = recurrence_matrix(traj, 1e12)
        assert R.all()

    @_python
    def test_small_epsilon_only_diagonal(self):
        rng = np.random.default_rng(4)
        traj = rng.normal(0, 1, (15, 3))
        R = recurrence_matrix(traj, 1e-12)
        # Only the main diagonal should be recurrent (self-distance = 0).
        assert int(R.sum()) == R.shape[0]

    @_python
    def test_rejects_rank_three_trajectory(self):
        with pytest.raises(ValueError, match="trajectory must be 1D or 2D"):
            recurrence_matrix(np.zeros((2, 3, 4)), 0.1)


class TestAngularMetric:
    @_python
    def test_angular_wraps_phase_boundary(self):
        """θ = 0 and θ = 2π − ε are nearly equal under chord distance
        but far apart under Euclidean."""
        eps = 0.05
        phases = np.array([0.0, TWO_PI - eps])[:, np.newaxis]
        R_eu = recurrence_matrix(phases, 0.1, metric="euclidean")
        R_ang = recurrence_matrix(phases, 0.1, metric="angular")
        # Euclidean says not recurrent (|0 − 2π| ≫ 0.1).
        assert not R_eu[0, 1]
        # Angular says recurrent (chord distance ≈ ε).
        assert R_ang[0, 1]


class TestCrossRecurrence:
    @_python
    def test_self_cross_equals_plain_recurrence(self):
        rng = np.random.default_rng(5)
        traj = rng.normal(0, 1, (20, 3))
        R = recurrence_matrix(traj, 0.5)
        CR = cross_recurrence_matrix(traj, traj, 0.5)
        np.testing.assert_array_equal(R, CR)

    @_python
    def test_one_dimensional_inputs_coerce_to_single_state_dimension(self):
        t = np.linspace(0.0, 2.0 * math.pi, 32)
        a = np.sin(t)
        b = np.sin(t + 0.05)
        CR = cross_recurrence_matrix(a, b, 0.1)
        assert CR.shape == (32, 32)
        assert CR.dtype == bool
        assert np.any(np.diag(CR))

    @_python
    def test_mismatched_shape_raises(self):
        a = np.zeros((10, 3))
        b = np.zeros((8, 3))
        with pytest.raises(ValueError, match="trajectories must match"):
            cross_recurrence_matrix(a, b, 0.5)

    @_python
    def test_rejects_rank_three_inputs(self):
        good = np.zeros((4, 2))
        with pytest.raises(ValueError, match="traj_a must be 1D or 2D"):
            cross_recurrence_matrix(np.zeros((2, 3, 4)), good, 0.1)
        with pytest.raises(ValueError, match="traj_b must be 1D or 2D"):
            cross_recurrence_matrix(good, np.zeros((2, 3, 4)), 0.1)

    @_python
    def test_cross_angular_wraps_phase_boundary(self):
        a = np.array([0.0, math.pi])[:, np.newaxis]
        b = np.array([TWO_PI - 0.02, math.pi + 0.03])[:, np.newaxis]
        CR = cross_recurrence_matrix(a, b, 0.05, metric="angular")
        assert CR[0, 0]
        assert CR[1, 1]
        assert not CR[0, 1]


class TestRQA:
    @_python
    def test_measures_in_unit_interval(self):
        rng = np.random.default_rng(7)
        traj = rng.normal(0, 1, (40, 3))
        res = rqa(traj, 0.8)
        assert 0.0 <= res.recurrence_rate <= 1.0
        assert 0.0 <= res.determinism <= 1.0
        assert 0.0 <= res.laminarity <= 1.0
        assert res.max_diagonal >= 0
        assert res.max_vertical >= 0
        assert res.avg_diagonal >= 0.0
        assert res.trapping_time >= 0.0
        assert res.entropy_diagonal >= 0.0

    @_python
    def test_dense_recurrence_has_long_diagonals(self):
        """A near-constant trajectory has every off-diagonal point
        recurrent. ``_diagonal_lines`` only scans the upper triangle,
        so even for a fully-recurrent off-diagonal ``det`` tops out
        at ~0.5 by convention (Marwan 2007). The important signals
        are that the recurrence rate is near 1 and ``max_diagonal``
        is long."""
        traj = np.zeros((30, 2)) + 0.1
        res = rqa(traj, 1.0)
        assert res.recurrence_rate > 0.95
        assert res.determinism > 0.45
        assert res.max_diagonal >= 15

    @_python
    def test_cross_rqa_unit_bound(self):
        rng = np.random.default_rng(11)
        a = rng.normal(0, 1, (30, 3))
        b = rng.normal(0, 1, (30, 3))
        res = cross_rqa(a, b, 1.0)
        assert 0.0 <= res.recurrence_rate <= 1.0


class TestHypothesis:
    @_python
    @given(
        t=st.integers(min_value=3, max_value=30),
        d=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_rm_is_symmetric_bool(self, t: int, d: int, seed: int):
        rng = np.random.default_rng(seed)
        traj = rng.normal(0, 1, (t, d))
        R = recurrence_matrix(traj, 1.0)
        assert R.dtype == bool
        assert R.shape == (t, t)
        np.testing.assert_array_equal(R, R.T)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert r_mod.AVAILABLE_BACKENDS
        assert "python" in r_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert r_mod.AVAILABLE_BACKENDS[0] == r_mod.ACTIVE_BACKEND


class TestBackendLoaderContracts:
    def test_rust_loader_wraps_recurrence_kernels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def fake_rm(
            traj_flat: np.ndarray,
            t: int,
            d: int,
            epsilon: float,
            angular: bool,
        ) -> np.ndarray:
            seen["rm"] = (traj_flat.copy(), t, d, epsilon, angular)
            return np.eye(t, dtype=np.uint8).ravel()

        def fake_cross(
            a_flat: np.ndarray,
            b_flat: np.ndarray,
            t: int,
            d: int,
            epsilon: float,
            angular: bool,
        ) -> np.ndarray:
            seen["cross"] = (a_flat.copy(), b_flat.copy(), t, d, epsilon, angular)
            return np.ones(t * t, dtype=np.uint8)

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.recurrence_matrix_rust = fake_rm
        fake_spo.cross_recurrence_matrix_rust = fake_cross
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        loaded = r_mod._load_rust_fns()
        traj_flat = np.arange(6, dtype=np.float64)
        rm = loaded["rm"](traj_flat, 3, 2, 0.25, True).reshape(3, 3)
        cross = loaded["cross_rm"](traj_flat, traj_flat + 1.0, 3, 2, 0.5, False)

        np.testing.assert_array_equal(rm, np.eye(3, dtype=np.uint8))
        np.testing.assert_array_equal(cross, np.ones(9, dtype=np.uint8))
        rm_flat, t, d, epsilon, angular = seen["rm"]
        np.testing.assert_array_equal(rm_flat, traj_flat)
        assert (t, d, epsilon, angular) == (3, 2, 0.25, True)
        _, _, t, d, epsilon, angular = seen["cross"]
        assert (t, d, epsilon, angular) == (3, 2, 0.5, False)
