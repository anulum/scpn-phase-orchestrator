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

import subprocess
import sys
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _embedding_validation as embedding_validation,
)
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
from tests.typing_contracts import assert_precise_ndarray_hint

DelayBackend = Callable[[np.ndarray, object, object], np.ndarray]
MiBackend = Callable[[np.ndarray, object, object], float]
NnBackend = Callable[[np.ndarray, object, object], tuple[np.ndarray, np.ndarray]]


def test__embedding_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(embedding_validation.validate_delay_embed_backend_inputs)
    assert callable(embedding_validation.validate_mutual_information_backend_inputs)
    assert callable(embedding_validation.validate_nearest_neighbor_backend_inputs)


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


def _reference_de(sig: np.ndarray, delay: int, dim: int) -> np.ndarray:
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
                assert_precise_ndarray_hint(hints[key])
                assert "float64" in str(hints[key])
        if fn.__name__.startswith("delay_embed"):
            assert_precise_ndarray_hint(hints["return"])
            assert "float64" in str(hints["return"])
        if fn.__name__.startswith("nearest_neighbor_distances"):
            assert_precise_ndarray_hint(hints["return"])
            assert "float64" in str(hints["return"])
            assert "int64" in str(hints["return"])


class TestDirectBackendBoundaryContracts:
    def test_delay_backend_output_contract_requires_exact_indexing(self) -> None:
        signal = np.arange(8, dtype=np.float64)
        valid = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])

        embedded = embedding_validation.validate_delay_embed_backend_output(
            valid,
            signal=signal,
            delay=2,
            dimension=2,
            t_effective=3,
        )

        np.testing.assert_array_equal(embedded, valid)

    @pytest.mark.parametrize(
        ("embedded", "match"),
        [
            (np.array([0.0, 2.0, 1.0]), "shape"),
            (np.array([[0.0, np.inf], [1.0, 3.0], [2.0, 4.0]]), "finite"),
            (np.array([[0.0, True], [1.0, 3.0], [2.0, 4.0]], dtype=object), "boolean"),
            (
                np.array(
                    [["0.0", "2.0"], ["1.0", "3.0"], ["2.0", "4.0"]],
                    dtype=object,
                ),
                "numeric-string",
            ),
            (np.array([[0.0, 2.0j], [1.0, 3.0], [2.0, 4.0]]), "real"),
            (np.array([[0.0, 2.0], [1.0, 99.0], [2.0, 4.0]]), "exact indexing"),
        ],
    )
    def test_delay_backend_output_contract_rejects_invalid_payloads(
        self,
        embedded: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            embedding_validation.validate_delay_embed_backend_output(
                embedded,
                signal=np.arange(8, dtype=np.float64),
                delay=2,
                dimension=2,
                t_effective=3,
            )

    @pytest.mark.parametrize("value", [0.0, 1.5, np.array(2.0)])
    def test_mi_backend_output_contract_accepts_non_negative_scalars(
        self,
        value: object,
    ) -> None:
        result = embedding_validation.validate_mutual_information_backend_output(value)
        assert result >= 0.0

    @pytest.mark.parametrize(
        "value",
        [
            np.bool_(True),
            -1.0,
            np.inf,
            np.array("1.0", dtype=object),
            1.0 + 0.0j,
            np.array([1.0]),
        ],
    )
    def test_mi_backend_output_contract_rejects_invalid_payloads(
        self,
        value: object,
    ) -> None:
        with pytest.raises(ValueError, match="mutual information backend output"):
            embedding_validation.validate_mutual_information_backend_output(value)

    def test_nn_backend_output_contract_accepts_metric_payload(self) -> None:
        distances, indices = (
            embedding_validation.validate_nearest_neighbor_backend_outputs(
                np.array([1.0, 2.0, 1.0]),
                np.array([2.0, 0.0, 0.0]),
                t=3,
            )
        )

        np.testing.assert_allclose(distances, [1.0, 2.0, 1.0])
        np.testing.assert_array_equal(indices, [2, 0, 0])

    @pytest.mark.parametrize(
        ("distances", "indices", "match"),
        [
            (np.array([1.0, np.nan]), np.array([1.0, 0.0]), "finite"),
            (np.array([1.0, -0.1]), np.array([1.0, 0.0]), "non-negative"),
            (np.array([1.0, 2.0]), np.array([1.0, np.inf]), "finite"),
            (np.array([1.0, 2.0]), np.array([1.0, 0.5]), "integral"),
            (np.array([1.0, 2.0]), np.array([2.0, 0.0]), "in range"),
            (np.array([1.0, 2.0]), np.array([0.0, 1.0]), "self"),
            (np.array([1.0]), np.array([0.0, 1.0]), "shape"),
            (np.array([True, 1.0], dtype=object), np.array([1.0, 0.0]), "booleans"),
            (np.array([1.0, 2.0]), np.array([True, 0.0], dtype=object), "booleans"),
            (
                np.array(["1.0", "2.0"], dtype=object),
                np.array([1.0, 0.0]),
                "numeric-string",
            ),
            (
                np.array([1.0, 2.0]),
                np.array(["1", "0"], dtype=object),
                "numeric-string",
            ),
        ],
    )
    def test_nn_backend_output_contract_rejects_invalid_payloads(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            embedding_validation.validate_nearest_neighbor_backend_outputs(
                distances,
                indices,
                t=2,
            )

    @pytest.mark.parametrize(
        "fn",
        [delay_embed_go, delay_embed_julia, delay_embed_mojo],
    )
    @pytest.mark.parametrize(
        ("signal", "delay", "dimension", "message"),
        [
            (np.array([0.0, True], dtype=object), 1, 2, "signal"),
            (np.array([0.0 + 1.0j, 1.0 + 0.0j]), 1, 2, "signal"),
            (np.array(["0.0", "1.0", "2.0"], dtype=object), 1, 2, "numeric-string"),
            (np.array([0.0, np.nan, 2.0]), 1, 2, "signal"),
            (np.array([[0.0, 1.0]]), 1, 2, "signal"),
            (np.arange(8, dtype=np.float64), np.bool_(True), 2, "delay"),
            (np.arange(8, dtype=np.float64), 0, 2, "delay"),
            (np.arange(8, dtype=np.float64), 1, np.bool_(True), "dimension"),
            (np.arange(3, dtype=np.float64), 2, 3, "too short"),
        ],
    )
    def test_delay_backend_rejects_invalid_inputs_before_runtime_load(
        self,
        fn: DelayBackend,
        signal: np.ndarray,
        delay: object,
        dimension: object,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            fn(signal, delay, dimension)

    @pytest.mark.parametrize(
        "fn",
        [delay_embed_go, delay_embed_julia, delay_embed_mojo],
    )
    def test_delay_backend_rejects_object_complex_signal_alias_before_runtime_load(
        self,
        fn: DelayBackend,
    ) -> None:
        signal = np.array([0.0 + 0.0j, 1.0 + 0.25j, 2.0 + 0.0j], dtype=object)

        with pytest.raises(ValueError, match="real"):
            fn(signal, 1, 2)

    @pytest.mark.parametrize(
        "fn",
        [mutual_information_go, mutual_information_julia, mutual_information_mojo],
    )
    @pytest.mark.parametrize(
        ("signal", "lag", "n_bins", "message"),
        [
            (np.array([0.0, np.bool_(False)], dtype=object), 1, 8, "signal"),
            (np.array([0.0, 1.0j]), 1, 8, "signal"),
            (
                np.array(["0.0", "1.0", "0.0", "1.0"], dtype=object),
                1,
                8,
                "numeric-string",
            ),
            (np.array([0.0, np.inf]), 1, 8, "signal"),
            (np.arange(8, dtype=np.float64), np.bool_(True), 8, "lag"),
            (np.arange(8, dtype=np.float64), -1, 8, "lag"),
            (np.arange(8, dtype=np.float64), 1, np.bool_(True), "n_bins"),
            (np.arange(8, dtype=np.float64), 1, 1, "n_bins"),
        ],
    )
    def test_mi_backend_rejects_invalid_inputs_before_runtime_load(
        self,
        fn: MiBackend,
        signal: np.ndarray,
        lag: object,
        n_bins: object,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            fn(signal, lag, n_bins)

    @pytest.mark.parametrize(
        "fn",
        [mutual_information_go, mutual_information_julia, mutual_information_mojo],
    )
    def test_mi_backend_rejects_object_complex_signal_alias_before_runtime_load(
        self,
        fn: MiBackend,
    ) -> None:
        signal = np.array([0.0 + 0.0j, 1.0 + 0.25j, 0.0 + 0.0j], dtype=object)

        with pytest.raises(ValueError, match="real"):
            fn(signal, 1, 8)

    @pytest.mark.parametrize(
        "fn",
        [
            nearest_neighbor_distances_go,
            nearest_neighbor_distances_julia,
            nearest_neighbor_distances_mojo,
        ],
    )
    @pytest.mark.parametrize(
        ("embedded", "t", "m", "message"),
        [
            (np.array([0.0, True], dtype=object), 1, 2, "embedded"),
            (np.array([0.0 + 1.0j, 1.0 + 0.0j]), 1, 2, "embedded"),
            (np.array(["0.0", "1.0"], dtype=object), 1, 2, "numeric-string"),
            (np.array([0.0, np.nan]), 1, 2, "embedded"),
            (np.arange(6, dtype=np.float64), np.bool_(True), 3, "t"),
            (np.arange(6, dtype=np.float64), 3, 0, "m"),
            (np.arange(5, dtype=np.float64), 3, 2, "embedded length"),
        ],
    )
    def test_nn_backend_rejects_invalid_inputs_before_runtime_load(
        self,
        fn: NnBackend,
        embedded: np.ndarray,
        t: object,
        m: object,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            fn(embedded, t, m)

    @pytest.mark.parametrize(
        "fn",
        [
            nearest_neighbor_distances_go,
            nearest_neighbor_distances_julia,
            nearest_neighbor_distances_mojo,
        ],
    )
    def test_nn_backend_rejects_object_complex_embedded_alias_before_runtime_load(
        self,
        fn: NnBackend,
    ) -> None:
        embedded = np.array([0.0 + 0.0j, 1.0 + 0.25j], dtype=object)

        with pytest.raises(ValueError, match="real"):
            fn(embedded, 1, 2)

    def test_backend_output_validators_reject_object_complex_aliases_as_non_real(
        self,
    ) -> None:
        signal = np.arange(8, dtype=np.float64)
        with pytest.raises(ValueError, match="real"):
            embedding_validation.validate_delay_embed_backend_output(
                np.array(
                    [[0.0 + 0.0j, 2.0], [1.0, 3.0 + 0.25j], [2.0, 4.0]],
                    dtype=object,
                ),
                signal=signal,
                delay=2,
                dimension=2,
                t_effective=3,
            )
        with pytest.raises(ValueError, match="real"):
            embedding_validation.validate_mutual_information_backend_output(
                np.array(1.0 + 0.0j, dtype=object)
            )
        with pytest.raises(ValueError, match="real"):
            embedding_validation.validate_nearest_neighbor_backend_outputs(
                np.array([1.0 + 0.0j, 2.0 + 0.25j], dtype=object),
                np.array([1.0, 0.0]),
                t=2,
            )
        with pytest.raises(ValueError, match="integer"):
            embedding_validation.validate_nearest_neighbor_backend_outputs(
                np.array([1.0, 2.0]),
                np.array([1.0 + 0.0j, 0.0 + 0.25j], dtype=object),
                t=2,
            )

    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("0.0\n\n1.0\n", 2, "DE", "Mojo DE returned 3 lines"),
            ("\n0.0\n", 1, "MI", "Mojo MI returned 2 lines"),
            ("0.0\n1.0\n2.0\n", 2, "NN", "Mojo NN returned 3 lines"),
        ],
    )
    def test_mojo_runner_rejects_raw_stdout_cardinality_mismatches(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        expected_count: int,
        label: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_embedding_mojo._ensure_exe",
            lambda: "embedding_mojo",
        )
        from scpn_phase_orchestrator.experimental.accelerators.monitor import (
            _embedding_mojo,
        )

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            _embedding_mojo._run(
                "DE 2 1 1 0 1\n",
                expected_count=expected_count,
                label=label,
            )


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
    ) -> None:
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
    @pytest.mark.native_runtime
    def test_rust_active_still_resolves_mi_nn(self) -> None:
        """With Rust active, MI and kNN come from the next real owner.

        Rust exposes no standalone MI or kNN kernel. Automatic results equal
        the explicit results of the owner reported by ``embedding_backend``.
        """
        if em_mod.ACTIVE_BACKEND != "rust":
            pytest.fail("this native lane requires the installed Rust kernel")
        sig = _signal(0)
        emb = _reference_de(sig, 5, 3)
        owner = em_mod.embedding_backend("mi")
        assert owner != "rust"
        assert em_mod.embedding_backend("nn") == owner
        assert mutual_information(sig, 5, 16) == mutual_information(
            sig, 5, 16, backend=owner
        )
        np.testing.assert_array_equal(delay_embed(sig, 5, 3, backend="rust"), emb)
        auto_dist, auto_idx = nearest_neighbor_distances(emb)
        owner_dist, owner_idx = nearest_neighbor_distances(emb, backend=owner)
        np.testing.assert_array_equal(auto_dist, owner_dist)
        np.testing.assert_array_equal(auto_idx, owner_idx)

    def test_dispatch_rejects_shape_correct_wrong_delay_embedding(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sig = np.arange(8, dtype=np.float64)

        def bad_loader() -> dict[str, object]:
            return {
                "de": lambda _signal, _delay, _dim: np.array(
                    [[0.0, 2.0], [1.0, 99.0], [2.0, 4.0]],
                    dtype=np.float64,
                ),
                "mi": None,
                "nn": None,
            }

        monkeypatch.setattr(em_mod, "AVAILABLE_BACKENDS", ["go", "python"])
        monkeypatch.setitem(em_mod._LOADERS, "go", bad_loader)
        monkeypatch.setitem(em_mod._BACKEND_CACHE, "go", bad_loader())
        prev = _force("go")
        try:
            with pytest.raises(ValueError, match="output shape|exact indexing"):
                delay_embed(sig, 2, 2)
        finally:
            _reset(prev)

    def test_dispatch_rejects_fractional_nearest_neighbor_indices(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        emb = np.array([[0.0], [2.0], [5.0]], dtype=np.float64)

        def bad_loader() -> dict[str, object]:
            return {
                "de": None,
                "mi": None,
                "nn": lambda _embedded, _t, _m: (
                    np.array([2.0, 2.0, 3.0], dtype=np.float64),
                    np.array([1.5, 0.0, 1.0], dtype=np.float64),
                ),
            }

        monkeypatch.setattr(em_mod, "AVAILABLE_BACKENDS", ["go", "python"])
        monkeypatch.setitem(em_mod._LOADERS, "go", bad_loader)
        monkeypatch.setitem(em_mod._BACKEND_CACHE, "go", bad_loader())
        prev = _force("go")
        try:
            with pytest.raises(ValueError, match="indices must be integral"):
                nearest_neighbor_distances(emb)
        finally:
            _reset(prev)

    def test_dispatch_returns_python_fallback_when_chain_reaches_python(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(em_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(em_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
        monkeypatch.setitem(em_mod._LOADERS, "rust", lambda: {"mi": None})
        monkeypatch.setattr(em_mod, "_BACKEND_CACHE", {})
        assert em_mod._dispatch("mi") == ("python", None)

    def test_dispatch_returns_none_when_non_python_backends_have_no_kernel(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(em_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(em_mod, "AVAILABLE_BACKENDS", ["rust"])
        monkeypatch.setitem(em_mod._LOADERS, "rust", lambda: {"mi": None})
        monkeypatch.setattr(em_mod, "_BACKEND_CACHE", {})
        assert em_mod._dispatch("mi") == ("python", None)

    @pytest.mark.native_runtime
    def test_rust_loader_wraps_flat_delay_kernel(self) -> None:
        """The installed Rust kernel returns exact windows through the public API."""
        if em_mod.ACTIVE_BACKEND != "rust":
            pytest.fail("this native lane requires the installed Rust kernel")
        signal = np.arange(10, dtype=np.float64) ** 2
        assert em_mod.embedding_backend("de", "rust") == "rust"
        got = delay_embed(signal, 2, 3, backend="rust")
        np.testing.assert_array_equal(got, _reference_de(signal, 2, 3))
        np.testing.assert_array_equal(signal, np.arange(10, dtype=np.float64) ** 2)

    def test_rust_loader_refuses_a_kernel_without_its_exports(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A package lacking the delay-embedding exports is not admitted."""
        monkeypatch.setitem(sys.modules, "spo_kernel", types.ModuleType("spo_kernel"))
        with pytest.raises(ImportError, match="lacks callable delay-embedding"):
            em_mod._load_rust_fns()

    def test_julia_loader_returns_bridge_callables(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(em_mod, "require_juliacall_main", lambda: object())

        loaded = em_mod._load_julia_fns()

        assert callable(loaded["de"])
        assert callable(loaded["mi"])
        assert callable(loaded["nn"])

    @pytest.mark.native_runtime
    @pytest.mark.parametrize("seed", [12, 31])
    def test_rust_active_uses_native_delay_and_dimension_wrappers(
        self, seed: int
    ) -> None:
        """Rust's native searches follow the Python law and own automatic calls."""
        if em_mod.ACTIVE_BACKEND != "rust":
            pytest.fail("this native lane requires the installed Rust kernel")
        sig = _signal(seed, t=64)
        before = sig.copy()
        assert em_mod.embedding_backend("optimal_delay") == "rust"
        assert em_mod.embedding_backend("optimal_dimension") == "rust"
        tau = optimal_delay(sig, max_lag=11, n_bins=9, backend="rust")
        assert tau == optimal_delay(sig, max_lag=11, n_bins=9, backend="python")
        assert tau == optimal_delay(sig, max_lag=11, n_bins=9)
        dim = optimal_dimension(
            sig, delay=2, max_dim=6, rtol=12.5, atol=1.5, backend="rust"
        )
        assert dim == optimal_dimension(
            sig, delay=2, max_dim=6, rtol=12.5, atol=1.5, backend="python"
        )
        np.testing.assert_array_equal(sig, before)
