# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for transfer entropy

"""Per-backend parity tests for ``monitor/transfer_entropy.py``."""

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _te_validation as te_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._te_go import (
    phase_te_go,
    te_matrix_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._te_julia import (
    phase_te_julia,
    te_matrix_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._te_mojo import (
    phase_te_mojo,
    te_matrix_mojo,
)
from scpn_phase_orchestrator.monitor import transfer_entropy as te_mod
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    phase_transfer_entropy,
    transfer_entropy_matrix,
)
from tests.transfer_entropy_oracle import oracle_phase_te, oracle_te_matrix
from tests.typing_contracts import assert_precise_ndarray_hint

TWO_PI = 2.0 * np.pi


def test__te_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(te_validation.validate_phase_te_backend_inputs)
    assert callable(te_validation.expected_phase_te_backend_output)
    assert callable(te_validation.validate_te_matrix_backend_inputs)
    assert callable(te_validation.expected_te_matrix_backend_output)


def _force(backend: str) -> str:
    prev = te_mod.ACTIVE_BACKEND
    te_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    te_mod.ACTIVE_BACKEND = prev


def _reference_te(src: np.ndarray, tgt: np.ndarray, n_bins: int) -> float:
    prev = _force("python")
    try:
        return phase_transfer_entropy(src, tgt, n_bins)
    finally:
        _reset(prev)


def _reference_matrix(series: np.ndarray, n_bins: int) -> np.ndarray:
    prev = _force("python")
    try:
        return transfer_entropy_matrix(series, n_bins)
    finally:
        _reset(prev)


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        phase_te_go,
        te_matrix_go,
        phase_te_julia,
        te_matrix_julia,
        phase_te_mojo,
        te_matrix_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        checked_hints = [
            value
            for key, value in hints.items()
            if key in {"source", "target", "phase_series"}
        ]
        if fn.__name__.startswith("te_matrix"):
            checked_hints.append(hints["return"])
        for hint in checked_hints:
            assert_precise_ndarray_hint(hint)
            assert "float64" in str(hint)


def test_backend_resolution_records_first_available_backend(monkeypatch) -> None:
    calls: list[str] = []

    def unavailable(name: str):
        def _loader() -> dict[str, object]:
            calls.append(name)
            raise ImportError(name)

        return _loader

    def rust_loader() -> dict[str, object]:
        calls.append("rust")
        return {
            "phase_te": lambda _src, _tgt, _bins: 0.125,
            "te_matrix": lambda _flat, n_osc, _n_time, _bins: np.eye(n_osc).ravel(),
        }

    # Resolution caches what it loads; isolate the cache so the stand-in
    # kernels cannot leak into later tests that name the real Rust owner.
    monkeypatch.setattr(te_mod, "_BACKEND_CACHE", {})
    monkeypatch.setitem(te_mod._LOADERS, "rust", rust_loader)
    monkeypatch.setitem(te_mod._LOADERS, "mojo", unavailable("mojo"))
    monkeypatch.setitem(te_mod._LOADERS, "julia", unavailable("julia"))
    monkeypatch.setitem(te_mod._LOADERS, "go", unavailable("go"))

    active, available = te_mod._resolve_backends()

    assert active == "rust"
    assert available == ["rust", "python"]
    assert calls == ["rust", "mojo", "julia", "go"]


@pytest.mark.native_runtime
def test_rust_loader_exposes_phase_and_matrix_kernels() -> None:
    """The compiled Rust loader's kernels match the counting oracle."""
    if "rust" not in te_mod.AVAILABLE_BACKENDS:
        pytest.skip("rust unavailable")
    kernels = te_mod._load_rust_fns()
    rng = np.random.default_rng(131)
    series = np.cumsum(rng.normal(0.4, 0.6, (3, 120)), axis=1)
    phase_te = kernels["phase_te"]
    te_matrix = kernels["te_matrix"]
    assert callable(phase_te) and callable(te_matrix)
    assert float(phase_te(series[0], series[1], 5)) == pytest.approx(
        oracle_phase_te(series[0].tolist(), series[1].tolist(), 5), abs=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(te_matrix(series.ravel(), 3, 120, 5)).reshape(3, 3),
        oracle_te_matrix(series.tolist(), 5),
        rtol=0.0,
        atol=1e-12,
    )


def test_julia_loader_exposes_direct_bridge_callables(monkeypatch) -> None:
    monkeypatch.setattr(te_mod, "require_juliacall_main", lambda: None)

    kernels = te_mod._load_julia_fns()

    assert kernels["phase_te"] is phase_te_julia
    assert kernels["te_matrix"] is te_matrix_julia


@pytest.mark.native_runtime
@pytest.mark.parametrize("owner", ["python", "rust", "mojo", "julia", "go"])
def test_dispatch_calls_active_backend_with_contiguous_arrays(owner: str) -> None:
    """Strided inputs reach every named owner and still match the oracle."""
    if owner not in te_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"{owner} unavailable")
    rng = np.random.default_rng(137)
    wide = np.cumsum(rng.normal(0.3, 0.5, (3, 200)), axis=1)
    src, tgt = wide[0, ::2], wide[1, 1::2]
    series = wide[:, ::2]
    assert not src.flags.c_contiguous and not series.flags.c_contiguous
    atol = 1e-9 if owner == "mojo" else 1e-12

    assert phase_transfer_entropy(src, tgt, n_bins=7, backend=owner) == pytest.approx(
        oracle_phase_te(src.tolist(), tgt.tolist(), 7), abs=atol
    )
    np.testing.assert_allclose(
        transfer_entropy_matrix(series, n_bins=5, backend=owner),
        oracle_te_matrix(series.tolist(), 5),
        rtol=0.0,
        atol=atol,
    )


def test_public_transfer_entropy_rejects_exact_contract_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.array([0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1, 4.0])
    target = np.array([0.1, 0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1])

    def fake_phase_te(_src: np.ndarray, _tgt: np.ndarray, _n_bins: int) -> float:
        return 0.0

    kernels = {"phase_te": fake_phase_te, "te_matrix": lambda *_args: np.zeros(4)}
    monkeypatch.setattr(te_mod, "ACTIVE_BACKEND", "go")
    monkeypatch.setattr(te_mod, "AVAILABLE_BACKENDS", ["go", "python"])
    monkeypatch.setitem(te_mod._BACKEND_CACHE, "go", kernels)
    monkeypatch.setitem(te_mod._LOADERS, "go", lambda: kernels)

    with pytest.raises(ValueError, match="exact reference"):
        phase_transfer_entropy(source, target, 2)


def test_public_transfer_entropy_matrix_rejects_exact_contract_divergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.array([0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1, 4.0])
    target = np.array([0.1, 0.1, 0.1, 4.0, 0.1, 4.0, 4.0, 0.1])
    series = np.vstack([source, target])

    def fake_matrix(
        _flat: np.ndarray,
        _n_osc: int,
        _n_time: int,
        _n_bins: int,
    ) -> np.ndarray:
        return np.zeros(4, dtype=np.float64)

    kernels = {"phase_te": lambda *_args: 0.0, "te_matrix": fake_matrix}
    monkeypatch.setattr(te_mod, "ACTIVE_BACKEND", "go")
    monkeypatch.setattr(te_mod, "AVAILABLE_BACKENDS", ["go", "python"])
    monkeypatch.setitem(te_mod._BACKEND_CACHE, "go", kernels)
    monkeypatch.setitem(te_mod._LOADERS, "go", lambda: kernels)

    with pytest.raises(ValueError, match="exact reference|diverged"):
        transfer_entropy_matrix(series, 2)
