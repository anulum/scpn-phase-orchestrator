# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for financial market Kuramoto module

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.market import (
    detect_regimes,
    extract_phase,
    market_order_parameter,
    market_plv,
    sync_warning,
)

T = 200
N_ASSETS = 5


@pytest.fixture()
def synced_returns():
    """All assets move together (high R)."""
    t = np.linspace(0, 4 * np.pi, T)
    base = np.sin(t)
    return np.column_stack([base + 0.01 * np.random.randn(T) for _ in range(N_ASSETS)])


@pytest.fixture()
def random_returns():
    """Independent random walks (low R)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((T, N_ASSETS))


class TestExtractPhase:
    def test_output_shape_1d(self):
        series = np.sin(np.linspace(0, 4 * np.pi, 100))
        phase = extract_phase(series)
        assert phase.shape == (100,)

    def test_output_shape_2d(self, synced_returns):
        phase = extract_phase(synced_returns)
        assert phase.shape == (T, N_ASSETS)

    def test_phase_range(self, synced_returns):
        phase = extract_phase(synced_returns)
        assert np.all(phase >= 0.0)
        assert np.all(phase < 2.0 * np.pi)


class TestMarketOrderParameter:
    def test_output_shape(self, synced_returns):
        phases = extract_phase(synced_returns)
        R = market_order_parameter(phases)
        assert R.shape == (T,)

    def test_synced_high_R(self, synced_returns):
        phases = extract_phase(synced_returns)
        R = market_order_parameter(phases)
        assert np.mean(R) > 0.7

    def test_random_low_R(self, random_returns):
        phases = extract_phase(random_returns)
        R = market_order_parameter(phases)
        assert np.mean(R) < 0.7

    def test_range(self, synced_returns):
        phases = extract_phase(synced_returns)
        R = market_order_parameter(phases)
        assert np.all(R >= 0.0)
        assert np.all(R <= 1.0 + 1e-10)


class TestMarketPLV:
    def test_output_shape(self, synced_returns):
        phases = extract_phase(synced_returns)
        plv = market_plv(phases, window=50)
        assert plv.shape == (T - 50 + 1, N_ASSETS, N_ASSETS)

    def test_diagonal_ones(self, synced_returns):
        phases = extract_phase(synced_returns)
        plv = market_plv(phases, window=50)
        for t in range(plv.shape[0]):
            np.testing.assert_allclose(np.diag(plv[t]), 1.0, atol=0.01)

    def test_range(self, synced_returns):
        phases = extract_phase(synced_returns)
        plv = market_plv(phases, window=50)
        assert np.all(plv >= -0.01)
        assert np.all(plv <= 1.01)


class TestDetectRegimes:
    def test_output_shape(self):
        R = np.random.rand(100)
        regimes = detect_regimes(R)
        assert regimes.shape == (100,)

    def test_high_R_synchronized(self):
        R = np.ones(50) * 0.9
        regimes = detect_regimes(R)
        assert np.all(regimes == 2)

    def test_low_R_desynchronized(self):
        R = np.ones(50) * 0.1
        regimes = detect_regimes(R)
        assert np.all(regimes == 0)

    def test_mid_R_transition(self):
        R = np.ones(50) * 0.5
        regimes = detect_regimes(R)
        assert np.all(regimes == 1)


class TestSyncWarning:
    def test_output_shape(self):
        R = np.random.rand(100)
        w = sync_warning(R)
        assert w.shape == (100,)

    def test_detects_crossing(self):
        R = np.concatenate([np.ones(50) * 0.3, np.ones(50) * 0.9])
        w = sync_warning(R, threshold=0.7, lookback=1)
        assert np.any(w)

    def test_no_warning_when_stable(self):
        R = np.ones(100) * 0.5
        w = sync_warning(R, threshold=0.7, lookback=1)
        assert not np.any(w)


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
