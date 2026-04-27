# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stability invariants for market kernels

"""Long-run / edge-case invariants for the market PLV / R(t) chain.

Marked ``slow`` — uses longer time series and larger asset counts
to exercise the rolling-window bookkeeping and the sincos
precompute paths.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import market as m_mod
from scpn_phase_orchestrator.upde.market import (
    market_order_parameter,
    market_plv,
)

pytestmark = pytest.mark.slow


def _python(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev = m_mod.ACTIVE_BACKEND
        m_mod.ACTIVE_BACKEND = "python"
        try:
            return func(*args, **kwargs)
        finally:
            m_mod.ACTIVE_BACKEND = prev

    return wrapper


class TestLongTimeSeries:
    @_python
    def test_order_parameter_holds_bounds_over_long_T(self):
        rng = np.random.default_rng(0)
        T, N = 5000, 10
        phases = rng.uniform(0, 2 * np.pi, (T, N))
        R = market_order_parameter(phases)
        assert R.shape == (T,)
        assert np.all(R >= 0.0)
        assert np.all(R <= 1.0 + 1e-12)

    @_python
    def test_plv_holds_bounds_over_long_T(self):
        rng = np.random.default_rng(1)
        T, N, W = 1000, 6, 50
        phases = rng.uniform(0, 2 * np.pi, (T, N))
        plv = market_plv(phases, window=W)
        assert plv.shape == (T - W + 1, N, N)
        assert np.all(plv >= 0.0)
        assert np.all(plv <= 1.0 + 1e-12)


class TestPLVSymmetry:
    @_python
    def test_plv_is_symmetric(self):
        """``PLV_ij`` is a symmetric positive matrix because it
        equals the modulus of the expected phase-difference, which
        is invariant under ``(i, j) → (j, i)``."""
        rng = np.random.default_rng(2)
        T, N, W = 60, 5, 15
        phases = rng.uniform(0, 2 * np.pi, (T, N))
        plv = market_plv(phases, window=W)
        # plv[w] must equal plv[w].T for every window.
        for w in range(plv.shape[0]):
            np.testing.assert_allclose(
                plv[w],
                plv[w].T,
                atol=1e-12,
            )


class TestLockedSubpopulation:
    @_python
    def test_plv_of_locked_subgroup_is_near_one(self):
        """Two assets driven by the same trajectory stay
        phase-locked; their pairwise PLV entry must stay close
        to 1 over every rolling window."""
        T, N, W = 80, 4, 20
        rng = np.random.default_rng(3)
        phases = np.empty((T, N))
        # Assets 0 and 1 share a trajectory; 2 and 3 independent.
        shared = np.cumsum(rng.normal(0, 0.05, T))
        phases[:, 0] = shared
        phases[:, 1] = shared
        phases[:, 2] = np.cumsum(rng.normal(0, 0.05, T))
        phases[:, 3] = np.cumsum(rng.normal(0, 0.05, T))
        plv = market_plv(phases, window=W)
        # PLV[:, 0, 1] ≈ 1 across every window.
        assert np.all(plv[:, 0, 1] > 0.999)
        assert np.all(plv[:, 1, 0] > 0.999)


class TestEdgeCases:
    @_python
    def test_empty_phases_zero_output(self):
        phases = np.zeros((0, 3))
        R = market_order_parameter(phases)
        assert R.shape == (0,)
        plv = market_plv(phases, window=5)
        assert plv.size == 0
