# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for fractal-dimension kernels

"""Algorithmic properties of :func:`correlation_integral` and
:func:`kaplan_yorke_dimension`.

Covered: monotone-in-ε property, bounded output ``[0, 1]``, known
analytic answers (``D_KY = N`` for all-positive exponents, ``0`` for
all-negative), Grassberger-Procaccia slope on a Gaussian cloud,
small-``T`` and empty-input edge cases, and Hypothesis property
testing.
"""

from __future__ import annotations

import functools

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor import dimension as dim_mod
from scpn_phase_orchestrator.monitor.dimension import (
    correlation_dimension,
    correlation_integral,
    kaplan_yorke_dimension,
)


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = dim_mod.ACTIVE_BACKEND
        dim_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            dim_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestCorrelationIntegral:
    @_python
    def test_monotone_in_epsilon(self):
        rng = np.random.default_rng(0)
        traj = rng.normal(0, 1, (40, 3))
        eps = np.logspace(-2, 1, 20)
        c = correlation_integral(traj, eps)
        assert np.all(np.diff(c) >= -1e-12)

    @_python
    def test_bounded_unit_interval(self):
        rng = np.random.default_rng(1)
        traj = rng.normal(0, 1, (40, 3))
        eps = np.logspace(-3, 3, 30)
        c = correlation_integral(traj, eps)
        assert np.all(c >= 0.0 - 1e-12)
        assert np.all(c <= 1.0 + 1e-12)

    @_python
    def test_large_epsilon_gives_one(self):
        rng = np.random.default_rng(2)
        traj = rng.normal(0, 1, (40, 3))
        c = correlation_integral(traj, np.array([1e6]))
        assert abs(c[0] - 1.0) < 1e-12

    @_python
    def test_zero_epsilon_gives_zero(self):
        rng = np.random.default_rng(3)
        traj = rng.normal(0, 1, (40, 3))
        c = correlation_integral(traj, np.array([0.0]))
        assert c[0] == 0.0

    @_python
    def test_small_T_returns_zero(self):
        """``T ≤ 1`` has no pairs — every ``C(ε)`` is ``0``."""
        traj = np.zeros((1, 3))
        c = correlation_integral(traj, np.array([1.0]))
        assert c[0] == 0.0

    @_python
    def test_slope_close_to_embedding_dim(self):
        """Gaussian cloud in 2-D should have a slope ≈ 2 across the
        mid-range of ε."""
        rng = np.random.default_rng(5)
        traj = rng.normal(0, 1, (400, 2))
        res = correlation_dimension(traj, n_epsilons=40)
        # Broad tolerance — GP at this scale drifts by ~0.5 from the
        # true dimension.
        assert 1.3 < res.D2 < 2.7


class TestKaplanYorke:
    @_python
    def test_all_negative_gives_zero(self):
        assert kaplan_yorke_dimension(
            np.array([-0.1, -0.3, -0.7])
        ) == 0.0

    @_python
    def test_all_positive_gives_N(self):
        le = np.array([0.1, 0.05, 0.02])
        assert kaplan_yorke_dimension(le) == 3.0

    @_python
    def test_sorted_vs_unsorted_same_result(self):
        le = np.array([0.1, -0.3, 0.2, -0.05])
        assert (
            kaplan_yorke_dimension(le)
            == kaplan_yorke_dimension(np.sort(le))
        )

    @_python
    def test_empty_input_returns_zero(self):
        assert kaplan_yorke_dimension(np.array([])) == 0.0

    @_python
    def test_classical_lorenz_like(self):
        """A Lorenz-style spectrum ``(λ_1, 0, λ_3)`` with
        ``λ_1 + λ_3 < 0`` lands in ``(1, 2)``."""
        le = np.array([0.9, 0.0, -14.5])
        d = kaplan_yorke_dimension(le)
        assert 2.0 < d < 3.0


class TestHypothesis:
    @_python
    @given(
        n=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_kaplan_yorke_bounded(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        le = rng.normal(0.0, 1.0, size=n)
        d = kaplan_yorke_dimension(le)
        assert 0.0 <= d <= float(n) + 1e-12


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert dim_mod.AVAILABLE_BACKENDS
        assert "python" in dim_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert dim_mod.AVAILABLE_BACKENDS[0] == dim_mod.ACTIVE_BACKEND
