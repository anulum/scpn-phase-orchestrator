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

import cmath
import cProfile
import functools
import importlib.util
import sys
import types
from collections.abc import Callable
from typing import Any, cast

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
        "surface",
        [extract_phase, market_order_parameter, detect_regimes],
    )
    def test_market_array_surfaces_reject_ragged_sequences(
        self,
        surface: Any,
    ) -> None:
        with pytest.raises(ValueError):
            surface([[0.0], [0.5, 1.0]])

    @pytest.mark.parametrize(
        "series",
        [
            np.array(["0.0", "1.0", "0.5", "-0.5"]),
            np.array([False, True, False, True]),
            np.array([0.0 + 1.0j, 1.0 + 0.0j, 0.5 + 0.0j, -0.5 + 0.0j]),
        ],
        ids=["numeric-string", "boolean", "complex"],
    )
    def test_extract_phase_rejects_non_real_numeric_series(
        self, series: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="series"):
            extract_phase(series)

    @pytest.mark.parametrize("surface", [market_order_parameter, market_plv])
    @pytest.mark.parametrize(
        "phases",
        [
            np.array([["0.0", "1.0"], ["0.5", "1.5"]]),
            np.array([[False, True], [True, False]]),
            np.array([[0.0 + 1.0j, 1.0], [0.5, 1.5]]),
        ],
        ids=["numeric-string", "boolean", "complex"],
    )
    def test_market_phase_surfaces_reject_non_real_numeric_arrays(
        self,
        surface: Any,
        phases: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="phases"):
            surface(phases, **({"window": 2} if surface is market_plv else {}))

    @pytest.mark.parametrize("surface", [detect_regimes, sync_warning])
    @pytest.mark.parametrize(
        "signal",
        [
            np.array(["0.2", "0.8"]),
            np.array([False, True]),
            np.array([0.2 + 0.1j, 0.8 + 0.0j]),
        ],
        ids=["numeric-string", "boolean", "complex"],
    )
    def test_market_signal_surfaces_reject_non_real_numeric_arrays(
        self,
        surface: Any,
        signal: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="R"):
            surface(signal)

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


def _oracle_order(rows: list[list[float]]) -> list[float]:
    """Return ``|mean_n exp(iθ_tn)|`` per time step with ``cmath``."""
    return [abs(sum(cmath.exp(1j * v) for v in row) / len(row)) for row in rows]


def _oracle_plv(rows: list[list[float]], window: int) -> list[float]:
    """Return the flattened rolling PLV matrices ``|mean_t exp(i(θ_j − θ_i))|``."""
    n = len(rows[0])
    out = []
    for start in range(len(rows) - window + 1):
        chunk = rows[start : start + window]
        for i in range(n):
            for j in range(n):
                total = sum(cmath.exp(1j * (r[j] - r[i])) for r in chunk)
                out.append(abs(total / window))
    return out


def _assert_loader_follows_the_laws(owner: str, loader: object) -> None:
    """Run the kernels a real loader returns against the ``cmath`` oracles."""
    if owner not in m_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"{owner} unavailable")
    op_fn, plv_fn = cast(
        "Callable[[], tuple[Callable[..., object], Callable[..., object]]]", loader
    )()
    rng = np.random.default_rng(17)
    phases = rng.uniform(-np.pi, np.pi, (9, 4))
    flat = np.ascontiguousarray(phases.ravel())
    atol = 1e-9 if owner == "mojo" else 1e-12
    np.testing.assert_allclose(
        np.asarray(op_fn(flat, 9, 4)),
        _oracle_order(phases.tolist()),
        rtol=0.0,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(plv_fn(flat, 9, 4, 5)),
        _oracle_plv(phases.tolist(), 5),
        rtol=0.0,
        atol=atol,
    )


class TestBackendLoaderContracts:
    @pytest.mark.native_runtime
    def test_rust_loader_wraps_order_parameter_and_plv_kernels(self) -> None:
        """The compiled Rust loader returns kernels that follow both laws."""
        _assert_loader_follows_the_laws("rust", m_mod._load_rust_fn)

    def test_rust_loader_rejects_numeric_string_raw_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_spo = types.ModuleType("spo_kernel")
        fake_spo.market_order_parameter_rust = lambda *_args: np.array(["0.2", "0.8"])
        fake_spo.market_plv_rust = lambda *_args: np.array(["1.0", "0.5", "0.5", "1.0"])
        monkeypatch.setitem(sys.modules, "spo_kernel", fake_spo)

        op_fn, plv_fn = m_mod._load_rust_fn()
        with pytest.raises(TypeError, match="order parameter must be numeric"):
            op_fn(np.zeros(4, dtype=np.float64), 2, 2)
        with pytest.raises(TypeError, match="phase-locking value must be numeric"):
            plv_fn(np.zeros(4, dtype=np.float64), 2, 2, 2)

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("owner", ["mojo", "julia", "go"])
    def test_optional_loader_contracts_return_order_and_plv_functions(
        self, owner: str
    ) -> None:
        """Each installed optional loader returns kernels that follow both laws."""
        loader = {
            "mojo": m_mod._load_mojo_fn,
            "julia": m_mod._load_julia_fn,
            "go": m_mod._load_go_fn,
        }[owner]
        _assert_loader_follows_the_laws(owner, loader)

    @pytest.mark.native_runtime
    def test_detect_regimes_uses_rust_classifier_when_available(self) -> None:
        """The compiled classifier runs when installed and matches the threshold law."""
        values = [0.1, 0.2, 0.5, 0.8, 0.9, 0.2000000001, 0.7999999999]
        profile = cProfile.Profile()
        profile.enable()
        try:
            regimes = detect_regimes(
                np.array(values), sync_threshold=0.8, desync_threshold=0.2
            )
        finally:
            profile.disable()
        rust_calls = sum(
            entry.callcount
            for entry in profile.getstats()
            if isinstance(entry.code, str) and "detect_regimes_rust" in entry.code
        )
        installed = importlib.util.find_spec("spo_kernel") is not None
        assert rust_calls == (1 if installed else 0)
        expected = [2 if v >= 0.8 else 0 if v <= 0.2 else 1 for v in values]
        np.testing.assert_array_equal(regimes, expected)


class TestNamedMarketOwners:
    """``backend=`` runs the named owner or refuses; it never substitutes."""

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("owner", ["python", "rust", "mojo", "julia", "go"])
    def test_each_named_owner_serves_both_public_functions(self, owner: str) -> None:
        """The public functions run the named owner against the oracles."""
        if owner not in m_mod.AVAILABLE_BACKENDS:
            pytest.skip(f"{owner} unavailable")
        assert m_mod.market_backend(owner) == owner
        rng = np.random.default_rng(23)
        phases = rng.uniform(0.0, 2 * np.pi, (8, 3))
        atol = 1e-9 if owner == "mojo" else 1e-12
        np.testing.assert_allclose(
            market_order_parameter(phases, backend=owner),
            _oracle_order(phases.tolist()),
            rtol=0.0,
            atol=atol,
        )
        np.testing.assert_allclose(
            m_mod.market_plv(phases, 4, backend=owner).ravel(),
            _oracle_plv(phases.tolist(), 4),
            rtol=0.0,
            atol=atol,
        )

    def test_absent_owner_and_unknown_name_are_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An absent named owner is a RuntimeError; an unknown name a ValueError."""

        def broken_loader() -> tuple[object, object]:
            """Fail like a missing shared library."""
            raise OSError("libmarket.so missing")

        monkeypatch.setattr(m_mod, "_BACKEND_CACHE", {})
        monkeypatch.setitem(m_mod._LOADERS, "go", broken_loader)
        with pytest.raises(RuntimeError, match="market backend 'go' unavailable"):
            market_order_parameter(np.zeros((3, 2)), backend="go")
        with pytest.raises(ValueError, match="unknown market backend"):
            m_mod.market_plv(np.zeros((3, 2)), 2, backend="cuda")
