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

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.basin_stability import (
    basin_stability,
    multi_basin_stability,
    steady_state_r,
)

FloatArray = NDArray[np.float64]

TWO_PI = 2.0 * math.pi

pytestmark = pytest.mark.slow


def _all_to_all(n: int, strength: float) -> FloatArray:
    k = np.ones((n, n)) * strength / n
    np.fill_diagonal(k, 0.0)
    return k


class TestLongRunMonteCarlo:
    """Specified longer finite-window samples retain bounded classifications."""

    def test_sb_converges_toward_expected_range(self) -> None:
        """The specified strong homogeneous graph passes most of thirty trials."""
        n = 6
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=5.0)
        result = basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=600,
            n_measure=200,
            n_samples=30,
            R_threshold=0.8,
            seed=101,
            backend="python",
        )
        assert result.S_B >= 0.8

    def test_multi_threshold_monotone_over_sweep(self) -> None:
        """S_B(R≥θ) must be monotone non-increasing in θ."""
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=3.0)
        results = multi_basin_stability(
            omegas,
            knm,
            dt=0.01,
            n_transient=400,
            n_measure=150,
            n_samples=20,
            R_thresholds=(0.1, 0.3, 0.5, 0.7, 0.9),
            seed=77,
            backend="python",
        )
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        sb_vals = [results[f"R>={t:.2f}"].S_B for t in thresholds]
        for prev, curr in zip(sb_vals, sb_vals[1:], strict=False):
            assert prev >= curr


class TestAlphaNonZero:
    """Cover the ``alpha != None`` branch in both entry points."""

    def test_basin_stability_with_alpha_shift(self) -> None:
        """The specified lagged population retains a bounded sampled fraction."""
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=5.0)
        alpha = np.full((n, n), 0.4)
        np.fill_diagonal(alpha, 0.0)
        result = basin_stability(
            omegas,
            knm,
            alpha=alpha,
            dt=0.01,
            n_transient=300,
            n_measure=100,
            n_samples=10,
            R_threshold=0.5,
            seed=5,
            backend="python",
        )
        assert 0.0 <= result.S_B <= 1.0

    def test_multi_basin_stability_with_alpha(self) -> None:
        """Finite nonzero lag retains bounded shared threshold classifications."""
        n = 4
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=3.0)
        alpha = np.full((n, n), 0.2)
        np.fill_diagonal(alpha, 0.0)
        results = multi_basin_stability(
            omegas,
            knm,
            alpha=alpha,
            dt=0.01,
            n_transient=250,
            n_measure=100,
            n_samples=8,
            R_thresholds=(0.3, 0.7),
            seed=9,
            backend="python",
        )
        for res in results.values():
            assert 0.0 <= res.S_B <= 1.0
            assert np.all(res.R_final >= 0.0)
            assert np.all(res.R_final <= 1.0 + 1e-12)


class TestSteadyStateRAlphaBranch:
    """Actual Python trials admit finite nonzero phase-lag matrices."""

    def test_steady_state_r_with_alpha(self) -> None:
        """The specified lagged population has a bounded finite-window measurement."""
        n = 5
        omegas = np.ones(n)
        knm = _all_to_all(n, strength=5.0)
        alpha = np.full((n, n), 0.3)
        np.fill_diagonal(alpha, 0.0)
        phases = np.zeros(n)  # locked start
        r_no_lag = steady_state_r(
            phases,
            omegas,
            knm,
            dt=0.01,
            n_transient=300,
            n_measure=100,
            backend="python",
        )
        r_with_lag = steady_state_r(
            phases,
            omegas,
            knm,
            alpha=alpha,
            dt=0.01,
            n_transient=300,
            n_measure=100,
            backend="python",
        )
        # Both are bounded in [0, 1]; with lag, R should be lower
        # (or equal in the degenerate case) than the no-lag value.
        assert 0.0 <= r_with_lag <= 1.0 + 1e-12
        assert r_with_lag <= r_no_lag + 1e-12
