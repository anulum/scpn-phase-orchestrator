# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov dispatch contract guards

"""Module-specific contracts for Lyapunov backend dispatch and validation."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _lyapunov_go as lyapunov_go_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _lyapunov_julia as lyapunov_julia_mod,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _lyapunov_mojo as lyapunov_mojo_mod,
)
from scpn_phase_orchestrator.monitor import lyapunov as lyapunov_mod

FloatArray: TypeAlias = NDArray[np.float64]


class _ObjectProbe:
    """Array-like value that rejects object-dtype boolean probing only."""

    def __array__(self, dtype: object = None) -> FloatArray:
        """Return numeric data except for object-dtype probing."""
        if dtype is object or dtype == np.dtype(object):
            raise TypeError("object probing disabled")
        return np.array([0.0, 0.1], dtype=np.float64)


def _matrix() -> FloatArray:
    """Return a valid two-node zero-diagonal coupling matrix."""
    return np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)


def _alpha() -> FloatArray:
    """Return a valid two-node phase-lag matrix."""
    return np.zeros((2, 2), dtype=np.float64)


def _sorted_backend(
    _phases: FloatArray,
    _omegas: FloatArray,
    _knm: FloatArray,
    _alpha_matrix: FloatArray,
    _dt: float,
    _n_steps: int,
    _qr_interval: int,
    _zeta: float,
    _psi: float,
) -> FloatArray:
    """Return a sorted deterministic backend spectrum."""
    return np.array([0.25, -0.5], dtype=np.float64)


def _raising_backend(
    _phases: FloatArray,
    _omegas: FloatArray,
    _knm: FloatArray,
    _alpha_matrix: FloatArray,
    _dt: float,
    _n_steps: int,
    _qr_interval: int,
    _zeta: float,
    _psi: float,
) -> FloatArray:
    """Simulate an optional backend failure after dispatch."""
    raise RuntimeError("optional backend unavailable")


def _spectrum_with_backend(
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_backend: str,
    backend: lyapunov_mod.LyapunovBackendFn,
) -> FloatArray:
    """Evaluate ``lyapunov_spectrum`` with a monkeypatched backend."""
    monkeypatch.setattr(lyapunov_mod, "ACTIVE_BACKEND", active_backend)
    monkeypatch.setattr(lyapunov_mod, "_dispatch", lambda: backend)
    return lyapunov_mod.lyapunov_spectrum(
        np.array([0.0, 0.1], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        _matrix(),
        _alpha(),
        dt=0.01,
        n_steps=2,
        qr_interval=1,
    )


def test_optional_backend_loader_returns_mojo_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Mojo loader returns the backend callable after executable probing."""

    def _fake_ensure_exe() -> str:
        return "lyapunov_mojo"

    monkeypatch.setattr(lyapunov_mojo_mod, "_ensure_exe", _fake_ensure_exe)
    monkeypatch.setattr(
        lyapunov_mojo_mod,
        "lyapunov_spectrum_mojo",
        _sorted_backend,
    )

    assert lyapunov_mod._load_mojo_fn() is _sorted_backend


def test_optional_backend_loader_returns_julia_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Julia loader returns the backend callable when juliacall imports."""
    monkeypatch.setitem(sys.modules, "juliacall", ModuleType("juliacall"))
    monkeypatch.setattr(
        lyapunov_julia_mod,
        "lyapunov_spectrum_julia",
        _sorted_backend,
    )

    assert lyapunov_mod._load_julia_fn() is _sorted_backend


def test_optional_backend_loader_returns_go_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Go loader returns the backend callable after library probing."""

    def _fake_load_lib() -> object:
        return object()

    monkeypatch.setattr(lyapunov_go_mod, "_load_lib", _fake_load_lib)
    monkeypatch.setattr(lyapunov_go_mod, "lyapunov_spectrum_go", _sorted_backend)

    assert lyapunov_mod._load_go_fn() is _sorted_backend


def test_dispatch_returns_none_when_no_backend_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch falls back to ``None`` when no backend and no Python slot remain."""
    previous_backend = lyapunov_mod.ACTIVE_BACKEND
    previous_available = list(lyapunov_mod.AVAILABLE_BACKENDS)
    previous_loader = lyapunov_mod._LOADERS["go"]

    def _missing_go() -> lyapunov_mod.LyapunovBackendFn:
        raise ImportError("go backend unavailable")

    lyapunov_mod.ACTIVE_BACKEND = "go"
    lyapunov_mod.AVAILABLE_BACKENDS = ["go"]
    lyapunov_mod._BACKEND_CACHE.clear()
    monkeypatch.setitem(lyapunov_mod._LOADERS, "go", _missing_go)
    try:
        backend = lyapunov_mod._dispatch()
    finally:
        lyapunov_mod.ACTIVE_BACKEND = previous_backend
        lyapunov_mod.AVAILABLE_BACKENDS = previous_available
        monkeypatch.setitem(lyapunov_mod._LOADERS, "go", previous_loader)
        lyapunov_mod._BACKEND_CACHE.clear()

    assert backend is None


def test_resolve_backends_keeps_python_when_optional_loaders_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend resolution ignores unavailable optional loaders."""
    previous_loaders = dict(lyapunov_mod._LOADERS)

    def _missing_backend() -> lyapunov_mod.LyapunovBackendFn:
        raise ImportError("optional backend unavailable")

    for backend_name in previous_loaders:
        monkeypatch.setitem(lyapunov_mod._LOADERS, backend_name, _missing_backend)
    try:
        active_backend, available_backends = lyapunov_mod._resolve_backends()
    finally:
        lyapunov_mod._LOADERS.update(previous_loaders)
        lyapunov_mod._BACKEND_CACHE.clear()

    assert active_backend == "python"
    assert available_backends == ["python"]


def test_vector_validation_handles_failed_boolean_alias_probe() -> None:
    """Public vector validation still accepts numeric arrays if probing fails."""
    guard = lyapunov_mod.LyapunovGuard()

    state = guard.evaluate(_ObjectProbe(), _matrix())

    assert isinstance(state.V, float)


def test_vector_validation_rejects_multidimensional_phase_vector() -> None:
    """Public Lyapunov guard rejects non-vector phase payloads."""
    guard = lyapunov_mod.LyapunovGuard()

    with pytest.raises(ValueError, match="phases must be one-dimensional"):
        guard.evaluate(np.array([[0.0, 0.1]], dtype=np.float64), np.zeros((2, 2)))


def test_matrix_validation_rejects_non_numeric_payload() -> None:
    """Public Lyapunov guard rejects non-numeric coupling matrices."""
    guard = lyapunov_mod.LyapunovGuard()

    with pytest.raises(ValueError, match="knm must be a finite matrix"):
        guard.evaluate(
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([["bad", "payload"], ["data", "0.0"]], dtype=object),
        )


def test_lyapunov_spectrum_rejects_omega_shape_mismatch() -> None:
    """The spectrum API requires natural frequencies to match phase shape."""
    with pytest.raises(ValueError, match="omegas shape"):
        lyapunov_mod.lyapunov_spectrum(
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            _matrix(),
            _alpha(),
            n_steps=1,
        )


def test_lyapunov_spectrum_rejects_empty_phase_vector() -> None:
    """The spectrum API requires at least one oscillator."""
    with pytest.raises(ValueError, match="at least one oscillator"):
        lyapunov_mod.lyapunov_spectrum(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            n_steps=1,
        )


def test_lyapunov_spectrum_rejects_invalid_integer_parameters() -> None:
    """Integer-like spectrum parameters reject booleans and low values."""
    with pytest.raises(ValueError, match="n_steps must be an integer"):
        lyapunov_mod.lyapunov_spectrum(
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            _matrix(),
            _alpha(),
            n_steps=True,
        )

    with pytest.raises(ValueError, match="qr_interval must be >="):
        lyapunov_mod.lyapunov_spectrum(
            np.array([0.0, 0.1], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
            _matrix(),
            _alpha(),
            n_steps=1,
            qr_interval=0,
        )


def test_python_spectrum_covers_driven_jacobian_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero driver exercises the forced Kuramoto Jacobian branch."""
    monkeypatch.setattr(lyapunov_mod, "_dispatch", lambda: None)

    spectrum = lyapunov_mod.lyapunov_spectrum(
        np.array([0.0, 0.1], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        _matrix(),
        _alpha(),
        dt=0.01,
        n_steps=2,
        qr_interval=1,
        zeta=0.2,
        psi=0.4,
    )

    assert spectrum.shape == (2,)
    assert np.all(np.isfinite(spectrum))


def test_non_rust_backend_success_path_validates_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Rust backend dispatch validates and returns sorted spectra."""
    spectrum = _spectrum_with_backend(
        monkeypatch,
        active_backend="go",
        backend=_sorted_backend,
    )

    np.testing.assert_allclose(spectrum, np.array([0.25, -0.5], dtype=np.float64))


def test_non_rust_backend_failure_falls_back_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Rust backend dispatch falls back to the Python reference on failure."""
    spectrum = _spectrum_with_backend(
        monkeypatch,
        active_backend="go",
        backend=_raising_backend,
    )

    assert spectrum.shape == (2,)
    assert np.all(np.isfinite(spectrum))


def test_backend_output_validation_rejects_non_numeric_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend output must be numeric before shape and monotonicity checks."""

    def _bad_backend(
        _phases: FloatArray,
        _omegas: FloatArray,
        _knm: FloatArray,
        _alpha_matrix: FloatArray,
        _dt: float,
        _n_steps: int,
        _qr_interval: int,
        _zeta: float,
        _psi: float,
    ) -> object:
        return ["bad", "payload"]

    with pytest.raises(ValueError, match="output must be numeric"):
        _spectrum_with_backend(
            monkeypatch,
            active_backend="rust",
            backend=_bad_backend,
        )


def test_backend_output_validation_rejects_wrong_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend output length must match the oscillator count."""

    def _bad_backend(
        _phases: FloatArray,
        _omegas: FloatArray,
        _knm: FloatArray,
        _alpha_matrix: FloatArray,
        _dt: float,
        _n_steps: int,
        _qr_interval: int,
        _zeta: float,
        _psi: float,
    ) -> FloatArray:
        return np.array([0.0, -0.1, -0.2], dtype=np.float64)

    with pytest.raises(ValueError, match="shape"):
        _spectrum_with_backend(
            monkeypatch,
            active_backend="rust",
            backend=_bad_backend,
        )


def test_backend_output_validation_rejects_unsorted_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend output must be sorted descending like the reference spectrum."""

    def _bad_backend(
        _phases: FloatArray,
        _omegas: FloatArray,
        _knm: FloatArray,
        _alpha_matrix: FloatArray,
        _dt: float,
        _n_steps: int,
        _qr_interval: int,
        _zeta: float,
        _psi: float,
    ) -> FloatArray:
        return np.array([-0.5, 0.25], dtype=np.float64)

    with pytest.raises(ValueError, match="sorted descending"):
        _spectrum_with_backend(
            monkeypatch,
            active_backend="rust",
            backend=_bad_backend,
        )
