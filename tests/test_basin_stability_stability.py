# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability invariants for basin_stability

"""Long-run + full-coverage invariants for ``upde.basin_stability``.

Marked ``slow`` — runs Monte Carlo sweeps with larger sample sizes
and longer transient windows. Also exercises the non-trivial
``alpha`` argument path (which the algorithm / backend tests skip
because the canonical problem uses ``alpha = 0``).
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import basin_stability as b_mod
from scpn_phase_orchestrator.upde.basin_stability import (
    basin_stability,
    multi_basin_stability,
    steady_state_r,
)

TWO_PI = 2.0 * math.pi

pytestmark = pytest.mark.slow


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = b_mod.ACTIVE_BACKEND
        b_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            b_mod.ACTIVE_BACKEND = prev

    return wrapper


def _all_to_all(n: int, strength: float) -> np.ndarray:
    k = np.ones((n, n)) * strength / n
    np.fill_diagonal(k, 0.0)
    return k


class TestLongRunMonteCarlo:
    @_python
    def test_sb_converges_toward_expected_range(self):
        """Strong coupling (K/N = 5) on a homogeneous population
        at ``R_threshold = 0.8`` should converge to ``S_B ≥ 0.8``
        as the sample size grows. The point estimate is stable
        enough at 30 samples that we can assert the lower bound."""
        n = 6
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=5.0)
        result = basin_stability(
            omegas, knm, dt=0.01, n_transient=600, n_measure=200,
            n_samples=30, R_threshold=0.8, seed=101,
        )
        assert result.S_B >= 0.8

    @_python
    def test_multi_threshold_monotone_over_sweep(self):
        """S_B(R≥θ) must be monotone non-increasing in θ."""
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=3.0)
        results = multi_basin_stability(
            omegas, knm, dt=0.01, n_transient=400, n_measure=150,
            n_samples=20,
            R_thresholds=(0.1, 0.3, 0.5, 0.7, 0.9),
            seed=77,
        )
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        sb_vals = [results[f"R>={t:.2f}"].S_B for t in thresholds]
        for prev, curr in zip(sb_vals, sb_vals[1:], strict=False):
            assert prev >= curr


class TestAlphaNonZero:
    """Cover the ``alpha != None`` branch in both entry points."""

    @_python
    def test_basin_stability_with_alpha_shift(self):
        """Non-zero phase lag reduces the steady-state R in the
        locked regime — the MC S_B should still be well-defined
        and inside the unit interval."""
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=5.0)
        alpha = np.full((n, n), 0.4)
        np.fill_diagonal(alpha, 0.0)
        result = basin_stability(
            omegas, knm, alpha=alpha,
            dt=0.01, n_transient=300, n_measure=100,
            n_samples=10, R_threshold=0.5, seed=5,
        )
        assert 0.0 <= result.S_B <= 1.0

    @_python
    def test_multi_basin_stability_with_alpha(self):
        n = 4
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=3.0)
        alpha = np.full((n, n), 0.2)
        np.fill_diagonal(alpha, 0.0)
        results = multi_basin_stability(
            omegas, knm, alpha=alpha,
            dt=0.01, n_transient=250, n_measure=100,
            n_samples=8, R_thresholds=(0.3, 0.7), seed=9,
        )
        for res in results.values():
            assert 0.0 <= res.S_B <= 1.0
            assert np.all(res.R_final >= 0.0)
            assert np.all(res.R_final <= 1.0 + 1e-12)


class TestSteadyStateRAlphaBranch:
    @_python
    def test_steady_state_r_with_alpha(self):
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=5.0)
        alpha = np.full((n, n), 0.3)
        np.fill_diagonal(alpha, 0.0)
        phases = np.zeros(n)  # locked start
        r_no_lag = steady_state_r(
            phases, omegas, knm, dt=0.01,
            n_transient=300, n_measure=100,
        )
        r_with_lag = steady_state_r(
            phases, omegas, knm, alpha=alpha, dt=0.01,
            n_transient=300, n_measure=100,
        )
        # Both are bounded in [0, 1]; with lag, R should be lower
        # (or equal in the degenerate case) than the no-lag value.
        assert 0.0 <= r_with_lag <= 1.0 + 1e-12
        assert r_with_lag <= r_no_lag + 1e-12
