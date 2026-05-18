# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for embedding primitives

"""Cross-backend parity for the three embedding primitives.

Tolerances:

* ``delay_embed`` — exact (pure indexing).
* ``mutual_information`` — 1e-9; the NumPy reference uses
  ``np.histogram2d`` which chooses bin edges through a slightly
  different interior path from our manual ``min/max`` binning
  (difference ≤ 3e-11 on random signals).
* ``nearest_neighbor_distances`` — 1e-9 float + array-exact
  indices.
"""

from __future__ import annotations

import sys
import types
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.monitor._embedding_go import (
    delay_embed_go,
    mutual_information_go,
    nearest_neighbor_distances_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._embedding_julia import (
    delay_embed_julia,
    mutual_information_julia,
    nearest_neighbor_distances_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._embedding_mojo import (
    delay_embed_mojo,
    mutual_information_mojo,
    nearest_neighbor_distances_mojo,
)
from scpn_phase_orchestrator.monitor import embedding as em_mod
from scpn_phase_orchestrator.monitor.embedding import (
    AVAILABLE_BACKENDS,
    delay_embed,
    mutual_information,
    nearest_neighbor_distances,
    optimal_delay,
    optimal_dimension,
)


def _force(backend: str) -> str:
    prev = em_mod.ACTIVE_BACKEND
    em_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    em_mod.ACTIVE_BACKEND = prev


def _reference_mi(sig: np.ndarray, lag: int, n_bins: int) -> float:
    prev = _force("python")
    try:
        return mutual_information(sig, lag, n_bins)
    finally:
        _reset(prev)


def _reference_nn(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prev = _force("python")
    try:
        return nearest_neighbor_distances(emb)
    finally:
        _reset(prev)


def _reference_de(sig, delay, dim) -> np.ndarray:
    prev = _force("python")
    try:
        return delay_embed(sig, delay, dim)
    finally:
        _reset(prev)


def _signal(seed: int, t: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sin(np.linspace(0, 10 * np.pi, t)) + 0.1 * rng.normal(0, 1, t)


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        delay_embed_go,
        delay_embed_julia,
        delay_embed_mojo,
        mutual_information_go,
        mutual_information_julia,
        mutual_information_mojo,
        nearest_neighbor_distances_go,
        nearest_neighbor_distances_julia,
        nearest_neighbor_distances_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        for key in {"signal", "embedded"}:
            if key in hints:
                assert "numpy.ndarray" in str(hints[key])
                assert "float64" in str(hints[key])
        if fn.__name__.startswith("delay_embed"):
            assert "numpy.ndarray" in str(hints["return"])
            assert "float64" in str(hints["return"])
        if fn.__name__.startswith("nearest_neighbor_distances"):
            assert "numpy.ndarray" in str(hints["return"])
            assert "float64" in str(hints["return"])
            assert "int64" in str(hints["return"])


class TestDelayEmbedParity:
    @given(
        t=st.integers(min_value=10, max_value=100),
        delay=st.integers(min_value=1, max_value=5),
        dim=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_all_backends_exact(
        self,
        t: int,
        delay: int,
        dim: int,
        seed: int,
    ):
        if t - (dim - 1) * delay <= 0:
            return
        sig = _signal(seed, t)
        ref = _reference_de(sig, delay, dim)
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                got = delay_embed(sig, delay, dim)
            finally:
                _reset(prev)
            np.testing.assert_array_equal(
                got,
                ref,
                err_msg=f"{backend} delay_embed diverged",
            )


class TestMutualInformationParity:
    @pytest.mark.parametrize("seed", [0, 42])
    @pytest.mark.parametrize("lag", [1, 5, 20])
    def test_non_python_backends_match(
        self,
        seed: int,
        lag: int,
    ) -> None:
        sig = _signal(seed)
        ref = _reference_mi(sig, lag, 16)
        for backend in AVAILABLE_BACKENDS:
            if backend == "python":
                continue
            prev = _force(backend)
            try:
                got = mutual_information(sig, lag, 16)
            finally:
                _reset(prev)
            assert abs(got - ref) < 1e-9, f"{backend} MI diverged: {got} vs {ref}"


class TestNearestNeighborParity:
    @pytest.mark.parametrize("seed", [0, 77])
    def test_dist_and_idx(self, seed: int) -> None:
        sig = _signal(seed)
        emb = delay_embed(sig, 5, 3)
        ref_dist, ref_idx = _reference_nn(emb)
        for backend in AVAILABLE_BACKENDS:
            if backend == "python":
                continue
            prev = _force(backend)
            try:
                dist, idx = nearest_neighbor_distances(emb)
            finally:
                _reset(prev)
            np.testing.assert_allclose(
                dist,
                ref_dist,
                atol=1e-9,
                err_msg=f"{backend} NN dist diverged",
            )
            np.testing.assert_array_equal(
                idx,
                ref_idx,
                err_msg=f"{backend} NN idx diverged",
            )


class TestDispatcherFallthroughForRust:
    def test_rust_active_still_resolves_mi_nn(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rust has no standalone MI / NN FFI; with Rust active the
        dispatcher must still produce a value via the next backend."""
        sig = _signal(0)
        emb = _reference_de(sig, 5, 3)

        def rust_loader() -> dict[str, object]:
            return {
                "de": lambda signal, delay, dim: _reference_de(signal, delay, dim),
                "mi": None,
                "nn": None,
            }

        def python_loader() -> dict[str, object]:
            return {
                "de": None,
                "mi": lambda signal, lag, n_bins: _reference_mi(signal, lag, n_bins),
                "nn": lambda embedded, _t, _m: _reference_nn(embedded),
            }

        monkeypatch.setattr(em_mod, "AVAILABLE_BACKENDS", ["rust", "go", "python"])
        monkeypatch.setitem(em_mod._LOADERS, "rust", rust_loader)
        monkeypatch.setitem(em_mod._LOADERS, "go", python_loader)
        prev = _force("rust")
        try:
            mi = mutual_information(sig, 5, 16)
            got_emb = delay_embed(sig, 5, 3)
            dist, idx = nearest_neighbor_distances(emb)
        finally:
            _reset(prev)
        assert np.isfinite(mi)
        np.testing.assert_array_equal(got_emb, emb)
        assert dist.size == emb.shape[0]
        assert idx.size == emb.shape[0]

    def test_dispatch_returns_python_fallback_when_chain_reaches_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(em_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(em_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
        monkeypatch.setitem(em_mod._LOADERS, "rust", lambda: {"mi": None})
        assert em_mod._dispatch("mi") is None

    def test_dispatch_returns_none_when_non_python_backends_have_no_kernel(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(em_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(em_mod, "AVAILABLE_BACKENDS", ["rust"])
        monkeypatch.setitem(em_mod._LOADERS, "rust", lambda: {"mi": None})
        assert em_mod._dispatch("mi") is None

    def test_rust_loader_wraps_flat_delay_kernel(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spo_kernel = types.ModuleType("spo_kernel")
        calls: dict[str, object] = {}

        def delay_embed_rust(signal: np.ndarray, delay: int, dim: int) -> np.ndarray:
            calls["signal"] = signal.copy()
            calls["delay"] = delay
            calls["dim"] = dim
            return np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]], dtype=np.float64)

        spo_kernel.delay_embed_rust = delay_embed_rust
        spo_kernel.optimal_delay_rust = lambda signal, max_lag, n_bins: 7
        spo_kernel.optimal_dimension_rust = lambda signal, delay, max_dim, rtol, atol: 3
        monkeypatch.setitem(sys.modules, "spo_kernel", spo_kernel)

        loaded = em_mod._load_rust_fns()
        got = loaded["de"](np.arange(6, dtype=np.float64).reshape(2, 3), 2, 3)

        np.testing.assert_array_equal(got, [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
        np.testing.assert_array_equal(
            calls["signal"],
            np.arange(6, dtype=np.float64).reshape(2, 3),
        )
        assert calls["delay"] == 2
        assert calls["dim"] == 3
        assert loaded["mi"] is None
        assert loaded["nn"] is None
        assert loaded["optimal_delay"](np.arange(10, dtype=np.float64), 20, 8) == 7
        assert (
            loaded["optimal_dimension"](
                np.arange(10, dtype=np.float64),
                1,
                5,
                15.0,
                2.0,
            )
            == 3
        )

    def test_rust_active_uses_native_delay_and_dimension_wrappers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, tuple[object, ...]] = {}

        def rust_loader() -> dict[str, object]:
            def rust_delay(signal: np.ndarray, max_lag: int, n_bins: int) -> int:
                calls["delay"] = (signal.copy(), max_lag, n_bins)
                return 5

            def rust_dim(
                signal: np.ndarray,
                delay: int,
                max_dim: int,
                rtol: float,
                atol: float,
            ) -> int:
                calls["dimension"] = (signal.copy(), delay, max_dim, rtol, atol)
                return 4

            return {
                "de": None,
                "mi": None,
                "nn": None,
                "optimal_delay": rust_delay,
                "optimal_dimension": rust_dim,
            }

        sig = _signal(12, t=64)
        monkeypatch.setitem(em_mod._LOADERS, "rust", rust_loader)
        prev = _force("rust")
        try:
            tau = optimal_delay(sig, max_lag=11, n_bins=9)
            dim = optimal_dimension(sig, delay=2, max_dim=6, rtol=12.5, atol=1.5)
        finally:
            _reset(prev)

        assert tau == 5
        assert dim == 4
        np.testing.assert_array_equal(calls["delay"][0], sig)
        assert calls["delay"][1:] == (11, 9)
        np.testing.assert_array_equal(calls["dimension"][0], sig)
        assert calls["dimension"][1:] == (2, 6, 12.5, 1.5)
