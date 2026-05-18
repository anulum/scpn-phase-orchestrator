# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Algorithmic tests for market PLV / R(t)

"""Algorithmic properties of ``upde.market``.

Covered: locked ensemble → ``R ≈ 1``; uniform ensemble → ``R ≈ 0``;
PLV matrix diagonal is 1 (identity coupling); PLV is bounded in
``[0, 1]``; ``detect_regimes`` classifier + ``sync_warning``
crossing detector; ``extract_phase`` Hilbert transform shape
invariant; Hypothesis property that output shapes match the spec.
"""

from __future__ import annotations

import functools
import sys
import types
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde import market as m_mod
from scpn_phase_orchestrator.upde.market import (
    detect_regimes,
    extract_phase,
    market_order_parameter,
    market_plv,
    sync_warning,
)


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


class TestOrderParameter:
    @_python
    def test_locked_ensemble_gives_R_one(self):
        T, N = 10, 5
        phases = np.full((T, N), 1.3)
        R = market_order_parameter(phases)
        assert R.shape == (T,)
        np.testing.assert_allclose(R, 1.0, atol=1e-12)

    @_python
    def test_uniform_ensemble_gives_R_zero(self):
        T, N = 5, 1000
        phases = np.tile(
            np.linspace(0, 2 * np.pi, N, endpoint=False),
            (T, 1),
        )
        R = market_order_parameter(phases)
        assert R.shape == (T,)
        assert np.all(R < 1e-10)

    @_python
    def test_R_bounded_in_unit_interval(self):
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, (20, 8))
        R = market_order_parameter(phases)
        assert np.all(R >= 0.0)
        assert np.all(R <= 1.0 + 1e-12)

    @_python
    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="phases must be"):
            market_order_parameter(np.array([1.0, 2.0, 3.0]))


class TestPLV:
    @_python
    def test_diagonal_is_one(self):
        """PLV_ii = |<exp(i·(θ_i − θ_i))>| = 1."""
        rng = np.random.default_rng(1)
        T, N, W = 30, 4, 10
        phases = rng.uniform(0, 2 * np.pi, (T, N))
        plv = market_plv(phases, window=W)
        assert plv.shape == (T - W + 1, N, N)
        for w in range(T - W + 1):
            np.testing.assert_allclose(
                np.diag(plv[w]),
                1.0,
                atol=1e-12,
            )

    @_python
    def test_bounded_in_unit_interval(self):
        rng = np.random.default_rng(2)
        T, N, W = 40, 5, 10
        phases = rng.uniform(0, 2 * np.pi, (T, N))
        plv = market_plv(phases, window=W)
        assert np.all(plv >= 0.0)
        assert np.all(plv <= 1.0 + 1e-12)

    @_python
    def test_locked_pair_gives_plv_one(self):
        """Two assets with identical phases across the window
        give PLV_ij = 1."""
        T, N, W = 20, 3, 10
        phases = np.zeros((T, N))
        # All three assets locked to the same trajectory.
        for t in range(T):
            phases[t, :] = 0.1 * t
        plv = market_plv(phases, window=W)
        np.testing.assert_allclose(plv, 1.0, atol=1e-12)

    @_python
    def test_window_larger_than_t_returns_empty(self):
        T, N = 5, 3
        phases = np.zeros((T, N))
        plv = market_plv(phases, window=10)
        assert plv.shape == (0, N, N)

    @_python
    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="phases must be"):
            market_plv(np.array([1.0, 2.0, 3.0]))


class TestDetectRegimes:
    def test_sync_and_desync_classes(self):
        R = np.array([0.1, 0.5, 0.9, 0.25, 0.8])
        regimes = detect_regimes(R, sync_threshold=0.7, desync_threshold=0.3)
        # 0=desync, 1=transition, 2=synchronised
        np.testing.assert_array_equal(regimes, [0, 1, 2, 0, 2])

    def test_output_dtype_is_int32(self):
        R = np.array([0.5])
        regimes = detect_regimes(R)
        assert regimes.dtype == np.int32


class TestSyncWarning:
    def test_crossing_detected(self):
        # Cross threshold around index 6
        R = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 0.8])
        w = sync_warning(R, threshold=0.7, lookback=1)
        assert np.any(w)
        # First crossing from below at index 6
        assert w[6]

    def test_no_crossing_below_threshold(self):
        R = np.full(10, 0.3)
        w = sync_warning(R, threshold=0.7, lookback=1)
        assert not np.any(w)


class TestExtractPhase:
    def test_hilbert_output_shape_1d(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 100))
        phase = extract_phase(x)
        assert phase.shape == (100,)
        assert np.all(phase >= 0.0)
        assert np.all(phase < 2 * np.pi + 1e-12)

    def test_hilbert_output_shape_2d(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((200, 4))
        phase = extract_phase(x)
        assert phase.shape == (200, 4)
        assert np.all(phase >= 0.0)
        assert np.all(phase < 2 * np.pi + 1e-12)


class TestInputValidation:
    @pytest.mark.parametrize(
        ("series", "match"),
        [
            (np.array([], dtype=np.float64), "series"),
            (np.zeros((4, 2, 1), dtype=np.float64), "series shape"),
            (np.array([0.0, np.nan, 1.0], dtype=np.float64), "series"),
        ],
    )
    def test_extract_phase_rejects_invalid_series(
        self,
        series: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            extract_phase(series)

    @pytest.mark.parametrize(
        ("phases", "match"),
        [
            (np.zeros((4, 0), dtype=np.float64), "phases"),
            (np.array([[0.0, np.nan]], dtype=np.float64), "phases"),
        ],
    )
    def test_order_parameter_rejects_invalid_phases(
        self,
        phases: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            market_order_parameter(phases)

    @pytest.mark.parametrize("window", [False, 0, -1, 1.5, "3"])
    def test_plv_rejects_invalid_window(self, window: Any) -> None:
        with pytest.raises(ValueError, match="window"):
            market_plv(np.zeros((4, 2), dtype=np.float64), window=window)

    @pytest.mark.parametrize(
        ("R", "sync_threshold", "desync_threshold", "match"),
        [
            (np.array([], dtype=np.float64), 0.7, 0.3, "R"),
            (np.array([0.1, np.nan], dtype=np.float64), 0.7, 0.3, "R"),
            (np.array([0.1, 0.5], dtype=np.float64), np.nan, 0.3, "sync_threshold"),
            (np.array([0.1, 0.5], dtype=np.float64), 0.2, 0.3, "sync_threshold"),
        ],
    )
    def test_detect_regimes_rejects_invalid_contract(
        self,
        R: np.ndarray,
        sync_threshold: Any,
        desync_threshold: Any,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            detect_regimes(
                R,
                sync_threshold=sync_threshold,
                desync_threshold=desync_threshold,
            )

    @pytest.mark.parametrize("lookback", [False, 0, -1, 1.5, "10"])
    def test_sync_warning_rejects_invalid_lookback(self, lookback: Any) -> None:
        with pytest.raises(ValueError, match="lookback"):
            sync_warning(np.array([0.1, 0.8], dtype=np.float64), lookback=lookback)


class TestHypothesis:
    @_python
    @given(
        t=st.integers(min_value=2, max_value=20),
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=8,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_order_parameter_shape_and_bounds(self, t, n, seed):
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, 2 * np.pi, (t, n))
        R = market_order_parameter(phases)
        assert R.shape == (t,)
        assert np.all(R >= 0.0)
        assert np.all(R <= 1.0 + 1e-12)


class TestDispatcherSurface:
    def test_available_non_empty(self):
        assert m_mod.AVAILABLE_BACKENDS
        assert "python" in m_mod.AVAILABLE_BACKENDS

    def test_active_is_first(self):
        assert m_mod.AVAILABLE_BACKENDS[0] == m_mod.ACTIVE_BACKEND


class TestBackendLoaderContracts:
    def test_rust_loader_wraps_order_parameter_and_plv_kernels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def fake_order(phases_flat: np.ndarray, t: int, n: int) -> np.ndarray:
            seen["order"] = (phases_flat.flags.c_contiguous, t, n)
            return np.linspace(0.25, 0.75, t, dtype=np.float64)

        def fake_plv(
            phases_flat: np.ndarray,
            t: int,
            n: int,
            window: int,
        ) -> np.ndarray:
            seen["plv"] = (phases_flat.flags.c_contiguous, t, n, window)
            return np.eye(n, dtype=np.float64).ravel()

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.market_order_parameter_rust = fake_order
        fake_spo.market_plv_rust = fake_plv
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        op_fn, plv_fn = m_mod._load_rust_fn()
        phases = np.arange(6, dtype=np.float64)
        np.testing.assert_allclose(op_fn(phases, 3, 2), [0.25, 0.5, 0.75])
        np.testing.assert_allclose(plv_fn(phases, 3, 2, 3).reshape(2, 2), np.eye(2))
        assert seen == {
            "order": (True, 3, 2),
            "plv": (True, 3, 2, 3),
        }

    def test_optional_loader_contracts_return_order_and_plv_functions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_order(*_args: object) -> np.ndarray:
            return np.array([1.0])

        def fake_plv(*_args: object) -> np.ndarray:
            return np.array([1.0])

        mojo_mod = types.ModuleType(
            "scpn_phase_orchestrator.experimental.accelerators.upde._market_mojo"
        )
        mojo_mod._ensure_exe = lambda: None
        mojo_mod.market_order_parameter_mojo = fake_order
        mojo_mod.market_plv_mojo = fake_plv

        julia_mod = types.ModuleType(
            "scpn_phase_orchestrator.experimental.accelerators.upde._market_julia"
        )
        julia_mod.market_order_parameter_julia = fake_order
        julia_mod.market_plv_julia = fake_plv

        go_mod = types.ModuleType(
            "scpn_phase_orchestrator.experimental.accelerators.upde._market_go"
        )
        go_mod._load_lib = lambda: None
        go_mod.market_order_parameter_go = fake_order
        go_mod.market_plv_go = fake_plv

        monkeypatch.setitem(sys.modules, mojo_mod.__name__, mojo_mod)
        monkeypatch.setitem(sys.modules, "juliacall", types.ModuleType("juliacall"))
        monkeypatch.setitem(sys.modules, julia_mod.__name__, julia_mod)
        monkeypatch.setitem(sys.modules, go_mod.__name__, go_mod)

        assert m_mod._load_mojo_fn() == (fake_order, fake_plv)
        assert m_mod._load_julia_fn() == (fake_order, fake_plv)
        assert m_mod._load_go_fn() == (fake_order, fake_plv)

    def test_detect_regimes_uses_rust_classifier_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def fake_detect(
            r_flat: np.ndarray,
            sync_threshold: float,
            desync_threshold: float,
        ) -> np.ndarray:
            seen["args"] = (r_flat.copy(), sync_threshold, desync_threshold)
            return np.array([0, 2, 1], dtype=np.int32)

        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.detect_regimes_rust = fake_detect
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        R = np.array([0.1, 0.9, 0.5])
        regimes = detect_regimes(R, sync_threshold=0.8, desync_threshold=0.2)
        np.testing.assert_array_equal(regimes, [0, 2, 1])
        r_flat, sync_threshold, desync_threshold = seen["args"]
        np.testing.assert_array_equal(r_flat, R)
        assert (sync_threshold, desync_threshold) == (0.8, 0.2)
