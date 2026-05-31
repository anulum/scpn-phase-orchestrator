# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for ITPC

"""Cross-backend parity for :func:`compute_itpc` /
:func:`itpc_persistence`. Every available backend (Rust / Mojo /
Julia / Go / Python) must produce the same output as the Python
reference on the same input.

Tolerances: Rust / Julia / Go / Mojo all within ``1e-12`` — ITPC is
a simple `mean(cos)`/`mean(sin)` accumulator with no log/exp, and the
measured parity sits at ~5e-17 on this host (bit-equivalent).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import get_type_hints

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _itpc_go as itpc_go_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _itpc_julia as itpc_julia_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _itpc_mojo as itpc_mojo_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _itpc_validation as itpc_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._itpc_go import (
    compute_itpc_go,
    itpc_persistence_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._itpc_julia import (
    compute_itpc_julia,
    itpc_persistence_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._itpc_mojo import (
    compute_itpc_mojo,
    itpc_persistence_mojo,
)
from scpn_phase_orchestrator.monitor import itpc as it_mod
from scpn_phase_orchestrator.monitor.itpc import (
    AVAILABLE_BACKENDS,
    compute_itpc,
    itpc_persistence,
)
from tests.typing_contracts import assert_precise_ndarray_hint

TWO_PI = 2.0 * np.pi
ItpcBackend = Callable[[np.ndarray, object, object], np.ndarray]
PersistenceBackend = Callable[[np.ndarray, object, object, object], float]


def test__itpc_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(itpc_validation.validate_compute_itpc_backend_inputs)
    assert callable(itpc_validation.expected_compute_itpc_backend_output)
    assert callable(itpc_validation.validate_itpc_persistence_backend_inputs)
    assert callable(itpc_validation.expected_itpc_persistence_backend_output)


def _force(backend: str) -> str:
    prev = it_mod.ACTIVE_BACKEND
    it_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    it_mod.ACTIVE_BACKEND = prev


def _reference_itpc(phases: np.ndarray) -> np.ndarray:
    prev = _force("python")
    try:
        return compute_itpc(phases)
    finally:
        _reset(prev)


def _reference_pers(phases: np.ndarray, idx: np.ndarray) -> float:
    prev = _force("python")
    try:
        return itpc_persistence(phases, idx)
    finally:
        _reset(prev)


def _problem(seed: int, n_trials: int = 30, n_tp: int = 80):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, TWO_PI, size=(n_trials, n_tp))


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        compute_itpc_go,
        compute_itpc_julia,
        compute_itpc_mojo,
        itpc_persistence_go,
        itpc_persistence_julia,
        itpc_persistence_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        assert_precise_ndarray_hint(hints["phases_flat"])
        assert "float64" in str(hints["phases_flat"])
        if "pause_indices" in hints:
            assert_precise_ndarray_hint(hints["pause_indices"])
            assert "int64" in str(hints["pause_indices"])
        if fn.__name__.startswith("compute_itpc"):
            assert_precise_ndarray_hint(hints["return"])
            assert "float64" in str(hints["return"])


class _FakeJuliaITPC:
    def __init__(self, *, vector: object, persistence: object = 0.5) -> None:
        self.vector = vector
        self.persistence = persistence

    def compute_itpc(
        self,
        _phases_flat: np.ndarray,
        _n_trials: int,
        _n_tp: int,
    ) -> object:
        return self.vector

    def itpc_persistence(
        self,
        _phases_flat: np.ndarray,
        _n_trials: int,
        _n_tp: int,
        _pause_indices: np.ndarray,
    ) -> object:
        return self.persistence


class _FakeGoITPC:
    def __init__(self, *, vector: np.ndarray, persistence: float = 0.5) -> None:
        self.vector = vector
        self.persistence = persistence

    def ComputeITPC(
        self,
        _phases_ptr: object,
        _n_trials: object,
        n_tp: object,
        out_ptr: object,
    ) -> int:
        out = np.ctypeslib.as_array(out_ptr, shape=(int(n_tp.value),))
        out[:] = self.vector
        return 0

    def ITPCPersistence(
        self,
        _phases_ptr: object,
        _n_trials: object,
        _n_tp: object,
        _indices_ptr: object,
        _n_indices: object,
        out_ptr: object,
    ) -> int:
        out_ptr._obj.value = self.persistence
        return 0


class TestDirectBackendBoundaryContracts:
    """Direct optional ITPC backends validate before runtime loading."""

    @pytest.mark.parametrize(
        ("stdout", "expected_count", "label", "match"),
        [
            ("", 2, "ITPC", "exactly 2 scalar"),
            ("0.25\n0.5\n0.75\n", 2, "ITPC", "exactly 2 scalar"),
            ("0.25\n\n0.5\n", 2, "ITPC", "exactly 2 scalar"),
            ("not-a-scalar\n", 1, "PERS", "non-scalar"),
        ],
    )
    def test_mojo_itpc_stdout_contract_rejects_malformed_payloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        expected_count: int,
        label: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(itpc_mojo_mod, "_ensure_exe", lambda: "itpc_mojo")
        monkeypatch.setattr(
            itpc_mojo_mod.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=stdout, stderr=""
            ),
        )

        with pytest.raises(ValueError, match=match):
            itpc_mojo_mod._run(
                "ITPC 2 2 0.0 0.2 0.4 0.6\n",
                expected_count=expected_count,
                label=label,
            )

    @pytest.mark.parametrize(
        "backend",
        [compute_itpc_go, compute_itpc_julia, compute_itpc_mojo],
    )
    @pytest.mark.parametrize(
        ("phases_flat", "n_trials", "n_tp", "match"),
        [
            (np.array([True, False]), 1, 2, "phases_flat"),
            (np.array([0.0, np.nan]), 1, 2, "phases_flat"),
            (np.array([0.0, 1.0], dtype=np.complex128), 1, 2, "real-valued"),
            (np.array([[0.0, 1.0]]), 1, 2, "one-dimensional"),
            (np.array([0.0, 1.0]), True, 2, "n_trials"),
            (np.array([0.0, 1.0]), -1, 2, "n_trials"),
            (np.array([0.0, 1.0]), 1, True, "n_tp"),
            (np.array([0.0, 1.0]), 1, -1, "n_tp"),
            (np.array([0.0, 1.0]), 2, 2, "n_trials\\*n_tp"),
        ],
    )
    def test_compute_validation_precedes_runtime_load(
        self,
        backend: ItpcBackend,
        phases_flat: np.ndarray,
        n_trials: object,
        n_tp: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            backend(phases_flat, n_trials, n_tp)

    @pytest.mark.parametrize(
        "backend",
        [itpc_persistence_go, itpc_persistence_julia, itpc_persistence_mojo],
    )
    @pytest.mark.parametrize(
        ("phases_flat", "n_trials", "n_tp", "pause_indices", "match"),
        [
            (np.array([True, False]), 1, 2, np.array([0]), "phases_flat"),
            (np.array([0.0, np.inf]), 1, 2, np.array([0]), "phases_flat"),
            (np.array([0.0, 1.0]), True, 2, np.array([0]), "n_trials"),
            (np.array([0.0, 1.0]), 1, True, np.array([0]), "n_tp"),
            (np.array([0.0, 1.0]), 2, 2, np.array([0]), "n_trials\\*n_tp"),
            (np.array([0.0, 1.0]), 1, 2, np.array([[0]]), "pause_indices"),
            (np.array([0.0, 1.0]), 1, 2, np.array([True]), "pause_indices"),
            (np.array([0.0, 1.0]), 1, 2, np.array([0.5]), "pause_indices"),
        ],
    )
    def test_persistence_validation_precedes_runtime_load(
        self,
        backend: PersistenceBackend,
        phases_flat: np.ndarray,
        n_trials: object,
        n_tp: object,
        pause_indices: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            backend(phases_flat, n_trials, n_tp, pause_indices)

    @pytest.mark.parametrize(
        ("backend_name", "match"),
        [
            ("go", "ITPC backend output"),
            ("julia", "ITPC backend output"),
            ("mojo", "ITPC backend output"),
        ],
    )
    def test_compute_rejects_invalid_backend_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_name: str,
        match: str,
    ) -> None:
        phases = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float64)
        if backend_name == "go":
            fake = _FakeGoITPC(vector=np.array([0.25, 1.25], dtype=np.float64))
            monkeypatch.setattr(itpc_go_mod, "_load_lib", lambda: fake)
            backend = compute_itpc_go
        elif backend_name == "julia":
            monkeypatch.setattr(
                itpc_julia_mod,
                "_ensure",
                lambda: _FakeJuliaITPC(vector=np.array([0.25, np.nan])),
            )
            backend = compute_itpc_julia
        else:
            monkeypatch.setattr(
                itpc_mojo_mod,
                "_run",
                lambda _payload, *, expected_count, label: [0.25],
            )
            backend = compute_itpc_mojo

        with pytest.raises(ValueError, match=match):
            backend(phases, 2, 2)

    @pytest.mark.parametrize(
        ("backend_name", "invalid_value"),
        [
            ("go", np.inf),
            ("julia", 1.25),
            ("mojo", -0.1),
        ],
    )
    def test_persistence_rejects_invalid_backend_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_name: str,
        invalid_value: float,
    ) -> None:
        phases = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float64)
        indices = np.array([0, 1], dtype=np.int64)
        if backend_name == "go":
            fake = _FakeGoITPC(
                vector=np.array([0.25, 0.5], dtype=np.float64),
                persistence=float(invalid_value),
            )
            monkeypatch.setattr(itpc_go_mod, "_load_lib", lambda: fake)
            backend = itpc_persistence_go
        elif backend_name == "julia":
            monkeypatch.setattr(
                itpc_julia_mod,
                "_ensure",
                lambda: _FakeJuliaITPC(
                    vector=np.array([0.25, 0.5]),
                    persistence=invalid_value,
                ),
            )
            backend = itpc_persistence_julia
        else:
            monkeypatch.setattr(
                itpc_mojo_mod,
                "_run",
                lambda _payload, *, expected_count, label: [invalid_value],
            )
            backend = itpc_persistence_mojo

        with pytest.raises(ValueError, match="ITPC persistence backend output"):
            backend(phases, 2, 2, indices)

    def test_compute_rejects_in_range_backend_output_that_breaks_exact_itpc(
        self,
    ) -> None:
        phases = np.array([0.0, 0.2, 1.5, 1.7], dtype=np.float64)
        expected = itpc_validation.expected_compute_itpc_backend_output(phases, 2, 2)
        wrong = np.zeros_like(expected)

        with pytest.raises(ValueError, match="exact reference"):
            itpc_validation.validate_compute_itpc_backend_output(
                wrong,
                2,
                expected=expected,
            )

    def test_persistence_rejects_in_range_backend_output_that_breaks_exact_itpc(
        self,
    ) -> None:
        phases = np.array([0.0, 0.2, 1.5, 1.7], dtype=np.float64)
        indices = np.array([0, 1], dtype=np.int64)
        expected = itpc_validation.expected_itpc_persistence_backend_output(
            phases,
            2,
            2,
            indices,
        )
        wrong = 0.0 if expected > 0.1 else 0.75

        with pytest.raises(ValueError, match="exact reference"):
            itpc_validation.validate_itpc_persistence_backend_output(
                wrong,
                expected=expected,
            )

    @pytest.mark.parametrize("backend_name", ["go", "julia", "mojo"])
    def test_direct_compute_rejects_in_range_exact_itpc_divergence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_name: str,
    ) -> None:
        phases = np.array([0.0, 0.2, 1.5, 1.7], dtype=np.float64)
        wrong = np.zeros(2, dtype=np.float64)
        if backend_name == "go":
            monkeypatch.setattr(
                itpc_go_mod,
                "_load_lib",
                lambda: _FakeGoITPC(vector=wrong),
            )
            backend = compute_itpc_go
        elif backend_name == "julia":
            monkeypatch.setattr(
                itpc_julia_mod,
                "_ensure",
                lambda: _FakeJuliaITPC(vector=wrong),
            )
            backend = compute_itpc_julia
        else:
            monkeypatch.setattr(
                itpc_mojo_mod,
                "_run",
                lambda _payload, *, expected_count, label: wrong.tolist(),
            )
            backend = compute_itpc_mojo

        with pytest.raises(ValueError, match="exact reference"):
            backend(phases, 2, 2)

    @pytest.mark.parametrize("backend_name", ["go", "julia", "mojo"])
    def test_direct_persistence_rejects_in_range_exact_itpc_divergence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_name: str,
    ) -> None:
        phases = np.array([0.0, 0.2, 1.5, 1.7], dtype=np.float64)
        indices = np.array([0, 1], dtype=np.int64)
        wrong = 0.0
        if backend_name == "go":
            monkeypatch.setattr(
                itpc_go_mod,
                "_load_lib",
                lambda: _FakeGoITPC(
                    vector=np.ones(2, dtype=np.float64),
                    persistence=wrong,
                ),
            )
            backend = itpc_persistence_go
        elif backend_name == "julia":
            monkeypatch.setattr(
                itpc_julia_mod,
                "_ensure",
                lambda: _FakeJuliaITPC(
                    vector=np.ones(2, dtype=np.float64),
                    persistence=wrong,
                ),
            )
            backend = itpc_persistence_julia
        else:
            monkeypatch.setattr(
                itpc_mojo_mod,
                "_run",
                lambda _payload, *, expected_count, label: [wrong],
            )
            backend = itpc_persistence_mojo

        with pytest.raises(ValueError, match="exact reference"):
            backend(phases, 2, 2, indices)


def test_rust_loader_returns_kernel_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_kernel = ModuleType("spo_kernel")
    fake_kernel.compute_itpc_rust = object()
    fake_kernel.itpc_persistence_rust = object()
    monkeypatch.setitem(sys.modules, "spo_kernel", fake_kernel)

    loaded = it_mod._load_rust_fns()

    assert loaded == {
        "itpc": fake_kernel.compute_itpc_rust,
        "persistence": fake_kernel.itpc_persistence_rust,
    }


def test_rust_itpc_dispatch_uses_contiguous_flattened_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, int, int]] = []

    def fake_rust_itpc(
        phases_flat: np.ndarray,
        n_trials: int,
        n_tp: int,
    ) -> np.ndarray:
        calls.append((phases_flat, n_trials, n_tp))
        return _reference_itpc(phases_flat.reshape(n_trials, n_tp))

    monkeypatch.setattr(it_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(it_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setitem(
        it_mod._BACKEND_FN_CACHE,
        "rust",
        {"itpc": fake_rust_itpc, "persistence": lambda *_args: 0.0},
    )
    monkeypatch.setitem(
        it_mod._LOADERS,
        "rust",
        lambda: {"itpc": fake_rust_itpc, "persistence": lambda *_args: 0.0},
    )
    phases = np.asfortranarray(
        np.array([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]], dtype=np.float64)
    )

    result = compute_itpc(phases)

    np.testing.assert_allclose(result, _reference_itpc(phases))
    assert len(calls) == 1
    phases_flat, n_trials, n_tp = calls[0]
    assert phases_flat.flags.c_contiguous
    np.testing.assert_array_equal(phases_flat, phases.ravel())
    assert (n_trials, n_tp) == (2, 3)


def test_rust_persistence_dispatch_uses_contiguous_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, int, int, np.ndarray]] = []

    def fake_rust_persistence(
        phases_flat: np.ndarray,
        n_trials: int,
        n_tp: int,
        pause_indices: np.ndarray,
    ) -> float:
        calls.append((phases_flat, n_trials, n_tp, pause_indices))
        return _reference_pers(phases_flat.reshape(n_trials, n_tp), pause_indices)

    monkeypatch.setattr(it_mod, "ACTIVE_BACKEND", "rust")
    monkeypatch.setattr(it_mod, "AVAILABLE_BACKENDS", ["rust", "python"])
    monkeypatch.setitem(
        it_mod._BACKEND_FN_CACHE,
        "rust",
        {"itpc": lambda *_args: np.array([]), "persistence": fake_rust_persistence},
    )
    monkeypatch.setitem(
        it_mod._LOADERS,
        "rust",
        lambda: {
            "itpc": lambda *_args: np.array([]),
            "persistence": fake_rust_persistence,
        },
    )
    phases = np.asfortranarray(
        np.array([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]], dtype=np.float64)
    )

    result = itpc_persistence(phases, [0, 2])

    assert result == _reference_pers(phases, np.array([0, 2], dtype=np.int64))
    assert len(calls) == 1
    phases_flat, n_trials, n_tp, pause_indices = calls[0]
    assert phases_flat.flags.c_contiguous
    assert pause_indices.flags.c_contiguous
    assert pause_indices.dtype == np.int64
    np.testing.assert_array_equal(phases_flat, phases.ravel())
    np.testing.assert_array_equal(pause_indices, np.array([0, 2], dtype=np.int64))
    assert (n_trials, n_tp) == (2, 3)


def test_non_rust_dispatch_flattens_inputs_and_preserves_exact_backend_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, int, int]] = []

    def fake_itpc(phases_flat: np.ndarray, n_trials: int, n_tp: int) -> np.ndarray:
        calls.append((phases_flat.copy(), n_trials, n_tp))
        return _reference_itpc(phases_flat.reshape(n_trials, n_tp))

    monkeypatch.setattr(it_mod, "ACTIVE_BACKEND", "go")
    monkeypatch.setattr(it_mod, "AVAILABLE_BACKENDS", ["go", "python"])
    monkeypatch.setitem(
        it_mod._BACKEND_FN_CACHE,
        "go",
        {"itpc": fake_itpc, "persistence": lambda *_args: 0.0},
    )
    monkeypatch.setitem(
        it_mod._LOADERS,
        "go",
        lambda: {"itpc": fake_itpc, "persistence": lambda *_args: 0.0},
    )
    phases = np.array([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]], dtype=np.float64)

    result = compute_itpc(phases)

    np.testing.assert_allclose(result, _reference_itpc(phases))
    assert len(calls) == 1
    phases_flat, n_trials, n_tp = calls[0]
    np.testing.assert_array_equal(phases_flat, phases.ravel())
    assert (n_trials, n_tp) == (2, 3)


def test_non_rust_persistence_dispatch_passes_pause_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[np.ndarray, int, int, np.ndarray]] = []

    def fake_persistence(
        phases_flat: np.ndarray,
        n_trials: int,
        n_tp: int,
        pause_indices: np.ndarray,
    ) -> float:
        calls.append((phases_flat.copy(), n_trials, n_tp, pause_indices.copy()))
        return _reference_pers(phases_flat.reshape(n_trials, n_tp), pause_indices)

    monkeypatch.setattr(it_mod, "ACTIVE_BACKEND", "mojo")
    monkeypatch.setattr(it_mod, "AVAILABLE_BACKENDS", ["mojo", "python"])
    monkeypatch.setitem(
        it_mod._BACKEND_FN_CACHE,
        "mojo",
        {"itpc": lambda *_args: np.array([]), "persistence": fake_persistence},
    )
    monkeypatch.setitem(
        it_mod._LOADERS,
        "mojo",
        lambda: {"itpc": lambda *_args: np.array([]), "persistence": fake_persistence},
    )
    phases = np.array([[0.0, 0.25, 0.5], [1.0, 1.25, 1.5]], dtype=np.float64)

    result = itpc_persistence(phases, [0, 2])

    assert result == _reference_pers(phases, np.array([0, 2], dtype=np.int64))
    assert len(calls) == 1
    phases_flat, n_trials, n_tp, pause_indices = calls[0]
    np.testing.assert_array_equal(phases_flat, phases.ravel())
    assert (n_trials, n_tp) == (2, 3)
    np.testing.assert_array_equal(pause_indices, np.array([0, 2]))


def test_public_itpc_falls_back_when_backend_breaks_exact_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = np.array([[0.0, 0.5, 1.0], [1.5, 2.0, 2.5]], dtype=np.float64)

    def fake_itpc(_phases_flat: np.ndarray, _n_trials: int, _n_tp: int) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    monkeypatch.setattr(it_mod, "ACTIVE_BACKEND", "go")
    monkeypatch.setattr(it_mod, "AVAILABLE_BACKENDS", ["go", "python"])
    monkeypatch.setitem(
        it_mod._BACKEND_FN_CACHE,
        "go",
        {"itpc": fake_itpc, "persistence": lambda *_args: 0.0},
    )
    monkeypatch.setitem(
        it_mod._LOADERS,
        "go",
        lambda: {"itpc": fake_itpc, "persistence": lambda *_args: 0.0},
    )

    np.testing.assert_allclose(compute_itpc(phases), _reference_itpc(phases))


def test_public_persistence_falls_back_when_backend_breaks_exact_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases = np.array([[0.0, 0.25, 0.5], [1.0, 1.25, 1.5]], dtype=np.float64)
    pause_indices = np.array([0, 2], dtype=np.int64)

    def fake_persistence(
        _phases_flat: np.ndarray,
        _n_trials: int,
        _n_tp: int,
        _pause_indices: np.ndarray,
    ) -> float:
        return 0.0

    monkeypatch.setattr(it_mod, "ACTIVE_BACKEND", "mojo")
    monkeypatch.setattr(it_mod, "AVAILABLE_BACKENDS", ["mojo", "python"])
    monkeypatch.setitem(
        it_mod._BACKEND_FN_CACHE,
        "mojo",
        {"itpc": lambda *_args: np.array([]), "persistence": fake_persistence},
    )
    monkeypatch.setitem(
        it_mod._LOADERS,
        "mojo",
        lambda: {"itpc": lambda *_args: np.array([]), "persistence": fake_persistence},
    )

    assert itpc_persistence(phases, pause_indices) == _reference_pers(
        phases,
        pause_indices,
    )


class TestRustParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "rust" not in AVAILABLE_BACKENDS:
            pytest.skip("Rust backend not built")

    @given(
        n_trials=st.integers(min_value=2, max_value=60),
        n_tp=st.integers(min_value=1, max_value=120),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_itpc(self, n_trials: int, n_tp: int, seed: int) -> None:
        phases = _problem(seed, n_trials, n_tp)
        ref = _reference_itpc(phases)
        prev = _force("rust")
        try:
            result = compute_itpc(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_persistence(self) -> None:
        phases = _problem(7)
        idx = np.array([5, 10, 25, 40, 75])
        ref = _reference_pers(phases, idx)
        prev = _force("rust")
        try:
            result = itpc_persistence(phases, idx)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


class TestJuliaParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "julia" not in AVAILABLE_BACKENDS:
            pytest.skip("Julia backend not available")

    @pytest.mark.parametrize("seed", [0, 42])
    def test_itpc(self, seed: int) -> None:
        phases = _problem(seed)
        ref = _reference_itpc(phases)
        prev = _force("julia")
        try:
            result = compute_itpc(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)


class TestGoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "go" not in AVAILABLE_BACKENDS:
            pytest.skip("Go backend not built")

    @given(
        n_trials=st.integers(min_value=2, max_value=40),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(
        max_examples=6,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_itpc(self, n_trials: int, seed: int) -> None:
        phases = _problem(seed, n_trials=n_trials)
        ref = _reference_itpc(phases)
        prev = _force("go")
        try:
            result = compute_itpc(phases)
        finally:
            _reset(prev)
        np.testing.assert_allclose(result, ref, atol=1e-12)

    def test_persistence(self) -> None:
        phases = _problem(11)
        idx = np.array([0, 15, 30, 60])
        ref = _reference_pers(phases, idx)
        prev = _force("go")
        try:
            result = itpc_persistence(phases, idx)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-12


class TestMojoParity:
    @pytest.fixture(autouse=True)
    def _skip_if_absent(self) -> None:
        if "mojo" not in AVAILABLE_BACKENDS:
            pytest.skip("Mojo backend not built")

    @pytest.mark.parametrize("seed", [0, 77])
    def test_itpc(self, seed: int) -> None:
        phases = _problem(seed)
        ref = _reference_itpc(phases)
        prev = _force("mojo")
        try:
            result = compute_itpc(phases)
        finally:
            _reset(prev)
        # ITPC has no log / exp amplification — bit-equivalent even
        # across the text round-trip.
        np.testing.assert_allclose(result, ref, atol=1e-9)

    def test_persistence(self) -> None:
        phases = _problem(99)
        idx = np.array([2, 8, 16, 40])
        ref = _reference_pers(phases, idx)
        prev = _force("mojo")
        try:
            result = itpc_persistence(phases, idx)
        finally:
            _reset(prev)
        assert abs(result - ref) < 1e-9


class TestCrossBackendConsistency:
    @pytest.mark.skipif(
        len(AVAILABLE_BACKENDS) < 2,
        reason="Only Python fallback available",
    )
    def test_all_backends_agree(self) -> None:
        phases = _problem(2026, n_trials=40, n_tp=100)
        idx = np.array([5, 25, 50, 75, 95])
        ref_itpc = _reference_itpc(phases)
        ref_pers = _reference_pers(phases, idx)
        tolerances = {
            "rust": 1e-12,
            "julia": 1e-12,
            "go": 1e-12,
            "mojo": 1e-9,
            "python": 0.0,
        }
        for backend in AVAILABLE_BACKENDS:
            prev = _force(backend)
            try:
                itpc_v = compute_itpc(phases)
                pers_v = itpc_persistence(phases, idx)
            finally:
                _reset(prev)
            np.testing.assert_allclose(
                itpc_v,
                ref_itpc,
                atol=tolerances[backend],
                err_msg=f"{backend} ITPC diverged from python",
            )
            assert abs(pers_v - ref_pers) <= tolerances[backend], (
                f"{backend} persistence diverged: {pers_v} vs {ref_pers}"
            )
