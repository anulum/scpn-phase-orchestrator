# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for financial market Kuramoto module

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import market as m_mod
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

    def test_invalid_series_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="one- or two-dimensional"):
            extract_phase(np.zeros((2, 2, 2), dtype=np.float64))


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

    def test_too_long_window_returns_empty(self, monkeypatch):
        monkeypatch.setattr(m_mod, "ACTIVE_BACKEND", "python")
        phases = np.zeros((10, 3), dtype=np.float64)
        plv = market_plv(phases, window=20)
        assert plv.shape == (0, 3, 3)


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

    def test_threshold_order_rejected(self):
        with pytest.raises(
            ValueError, match="sync_threshold must be greater than or equal"
        ):
            detect_regimes(np.ones(10), sync_threshold=0.2, desync_threshold=0.7)


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

    def test_invalid_lookback_rejected(self) -> None:
        with pytest.raises(ValueError, match="lookback"):
            sync_warning(np.ones(10), lookback=0)


class TestMarketPipelineWiring:
    """Pipeline: market returns → extract_phase → R → regime detection."""

    def test_returns_to_sync_regime(self):
        """extract_phase → market_order_parameter → detect_regimes:
        proves the market module processes real-like data end-to-end."""
        rng = np.random.default_rng(42)
        T, N = 200, 5
        # Correlated returns (sync regime)
        base = np.sin(np.linspace(0, 4 * np.pi, T))
        returns = np.column_stack(
            [base + 0.05 * rng.standard_normal(T) for _ in range(N)]
        )

        phases = np.column_stack([extract_phase(returns[:, i]) for i in range(N)])
        R = market_order_parameter(phases)
        assert R.shape == (T,)
        assert np.all(R >= 0.0) and np.all(R <= 1.0)

        regimes = detect_regimes(R)
        assert len(regimes) == T
        # detect_regimes returns integer labels
        assert all(isinstance(r, (int, np.integer)) for r in regimes)


class TestMarketDispatchSurface:
    def test_resolve_backends_prefers_first_available(self, monkeypatch):
        def fail_once() -> tuple:
            raise RuntimeError("backend unavailable")

        op_calls: list[str] = []

        def good_loader() -> tuple:
            op_calls.append("mojo")
            return (
                lambda *args: np.array([0.2, 0.2], dtype=np.float64),
                lambda *args: np.zeros((0,), dtype=np.float64),
            )

        monkeypatch.setattr(
            m_mod,
            "_LOADERS",
            {
                "rust": fail_once,
                "mojo": good_loader,
                "julia": fail_once,
                "go": fail_once,
            },
        )

        backend, available = m_mod._resolve_backends()
        assert backend == "mojo"
        assert available == ["mojo", "python"]
        assert op_calls == ["mojo"]

    def test_resolve_backends_falls_back_to_python_only(self, monkeypatch):
        def loader_a() -> tuple:
            raise ImportError("rust unavailable")

        def loader_b() -> tuple:
            raise RuntimeError("go unavailable")

        monkeypatch.setattr(
            m_mod,
            "_LOADERS",
            {
                "rust": loader_a,
                "mojo": loader_b,
                "julia": loader_b,
                "go": loader_b,
            },
        )
        backend, available = m_mod._resolve_backends()
        assert backend == "python"
        assert available == ["python"]

    def test_dispatch_returns_none_for_python_backend(self, monkeypatch):
        monkeypatch.setattr(m_mod, "ACTIVE_BACKEND", "python")
        assert m_mod._dispatch() is None
