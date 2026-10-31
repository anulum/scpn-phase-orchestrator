# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov dispatch contract guards

"""Module-specific contracts for Lyapunov backend dispatch and validation."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import lyapunov as lyapunov_mod

FloatArray: TypeAlias = NDArray[np.float64]
# A backend whose output is under test, so it may return a malformed payload.
_ProbeBackendFn: TypeAlias = Callable[
    [
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        float,
        int,
        int,
        float,
        float,
    ],
    object,
]


class _ObjectProbe:
    """Array-like value that rejects object-dtype boolean probing only."""

    def __array__(self, dtype: object = None) -> FloatArray:
        """Return numeric data except for object-dtype probing."""
        if dtype is object or dtype == np.dtype(object):
            raise TypeError("object probing disabled")
        return np.array([0.0, 0.1], dtype=np.float64)


class _ArrayFails:
    """Array-like value that rejects all NumPy array conversion attempts."""

    def __array__(self, dtype: object = None) -> FloatArray:
        """Raise a conversion error for defensive alias-helper paths."""
        raise TypeError("array conversion rejected")


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
    backend: _ProbeBackendFn,
) -> FloatArray:
    """Evaluate ``lyapunov_spectrum`` with a monkeypatched backend."""
    monkeypatch.setattr(lyapunov_mod, "ACTIVE_BACKEND", active_backend)
    monkeypatch.setattr(
        lyapunov_mod, "_dispatch", lambda _backend="auto": (active_backend, backend)
    )
    return lyapunov_mod.lyapunov_spectrum(
        np.array([0.0, 0.1], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        _matrix(),
        _alpha(),
        dt=0.01,
        n_steps=2,
        qr_interval=1,
    )


def _synchronised_spectrum(n: int, k: float, dt: float) -> list[float]:
    """Return the RK4 spectrum of an all-to-all network at sync (see test_lyapunov)."""
    z = -n * k * dt
    return [0.0] + [math.log(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24) / dt] * (n - 1)


def _assert_loader_follows_the_law(owner: str, loader: object) -> None:
    """Run the kernel a real loader returns on the analytic synchronised case."""
    if owner not in lyapunov_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"{owner} unavailable")
    kernel = cast("Callable[[], lyapunov_mod.LyapunovBackendFn]", loader)()
    n, k, dt = 3, 0.4, 0.01
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    spec = kernel(
        np.full(n, 0.7), np.full(n, 1.3), knm, np.zeros((n, n)), dt, 200, 5, 0.0, 0.0
    )
    np.testing.assert_allclose(
        spec, _synchronised_spectrum(n, k, dt), rtol=1e-9, atol=1e-12
    )


@pytest.mark.native_runtime
def test_optional_backend_loader_returns_mojo_callable() -> None:
    """The real Mojo loader returns a kernel with the analytic sync spectrum."""
    _assert_loader_follows_the_law("mojo", lyapunov_mod._load_mojo_fn)


def test_complex_alias_helper_handles_failure_and_complex_dtype() -> None:
    """Public alias detection handles failed probes and complex dtypes."""
    assert lyapunov_mod._contains_complex_alias(_ArrayFails()) is False
    assert (
        lyapunov_mod._contains_complex_alias(
            np.array([1.0 + 0.0j], dtype=np.complex128)
        )
        is True
    )


@pytest.mark.native_runtime
def test_optional_backend_loader_returns_julia_callable() -> None:
    """The real Julia loader returns a kernel with the analytic sync spectrum."""
    _assert_loader_follows_the_law("julia", lyapunov_mod._load_julia_fn)


@pytest.mark.native_runtime
def test_optional_backend_loader_returns_go_callable() -> None:
    """The real Go loader returns a kernel with the analytic sync spectrum."""
    _assert_loader_follows_the_law("go", lyapunov_mod._load_go_fn)


@pytest.mark.native_runtime
@pytest.mark.parametrize("owner", ["python", "rust", "mojo", "julia", "go"])
def test_each_named_owner_serves_the_public_spectrum(owner: str) -> None:
    """``lyapunov_spectrum(backend=owner)`` runs that owner on the analytic case."""
    if owner not in lyapunov_mod.AVAILABLE_BACKENDS:
        pytest.skip(f"{owner} unavailable")
    assert lyapunov_mod.lyapunov_backend(owner) == owner
    n, k, dt = 4, 0.3, 0.02
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    spec = lyapunov_mod.lyapunov_spectrum(
        np.full(n, 2.0),
        np.full(n, -0.4),
        knm,
        np.zeros((n, n)),
        dt=dt,
        n_steps=150,
        qr_interval=3,
        backend=owner,
    )
    np.testing.assert_allclose(
        spec, _synchronised_spectrum(n, k, dt), rtol=1e-9, atol=1e-12
    )


def test_named_owner_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent or failing named owners and unknown names refuse."""
    monkeypatch.setattr(lyapunov_mod, "_BACKEND_CACHE", {})
    monkeypatch.setitem(lyapunov_mod._LOADERS, "go", _missing_backend)
    with pytest.raises(RuntimeError, match="Lyapunov backend 'go' unavailable"):
        lyapunov_mod.lyapunov_backend("go")
    monkeypatch.setitem(lyapunov_mod._LOADERS, "go", lambda: _raising_backend)
    lyapunov_mod._BACKEND_CACHE.clear()
    with pytest.raises(RuntimeError, match="Lyapunov backend 'go' failed"):
        lyapunov_mod.lyapunov_spectrum(
            np.array([0.0, 0.1]),
            np.ones(2),
            _matrix(),
            _alpha(),
            n_steps=2,
            qr_interval=1,
            backend="go",
        )
    with pytest.raises(ValueError, match="unknown Lyapunov backend"):
        lyapunov_mod.lyapunov_spectrum(
            np.zeros(2), np.ones(2), _matrix(), _alpha(), backend="cuda"
        )


def _missing_backend() -> lyapunov_mod.LyapunovBackendFn:
    """Fail like an absent shared library."""
    raise ImportError("go backend unavailable")


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

    assert backend == ("python", None)


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


@pytest.mark.parametrize(
    ("field", "payload", "match"),
    [
        (
            "phases_init",
            np.array(["0.0", "0.1"]),
            "phases_init must contain only real numbers",
        ),
        (
            "omegas",
            np.array(["1.0", "1.1"]),
            "omegas must contain only real numbers",
        ),
        (
            "knm",
            np.array([["0.0", "0.4"], ["0.4", "0.0"]]),
            "knm must contain only real numbers",
        ),
        (
            "alpha",
            np.array([["0.0", "0.0"], ["0.0", "0.0"]]),
            "alpha must contain only real numbers",
        ),
    ],
)
def test_lyapunov_spectrum_rejects_numeric_string_array_inputs(
    field: str,
    payload: object,
    match: str,
) -> None:
    """The public spectrum API rejects numeric strings before backend dispatch."""
    kwargs: dict[str, object] = {
        "phases_init": np.array([0.0, 0.1], dtype=np.float64),
        "omegas": np.array([1.0, 1.0], dtype=np.float64),
        "knm": _matrix(),
        "alpha": _alpha(),
        "n_steps": 1,
    }
    kwargs[field] = payload

    # Alias as Callable[..., FloatArray]: kwargs carries a deliberately invalid value.
    spectrum: Callable[..., FloatArray] = lyapunov_mod.lyapunov_spectrum
    with pytest.raises(ValueError, match=match):
        spectrum(**kwargs)


@pytest.mark.parametrize(
    ("field", "payload", "match"),
    [
        (
            "phases_init",
            np.array([0.0 + 0.0j, 0.1 + 0.0j], dtype=np.complex128),
            "phases_init must be a finite real-valued vector",
        ),
        (
            "knm",
            np.array([[0.0, 0.4 + 0.0j], [0.4, 0.0]], dtype=np.complex128),
            "knm must be a finite real-valued matrix",
        ),
    ],
)
def test_lyapunov_spectrum_rejects_complex_array_inputs(
    field: str,
    payload: object,
    match: str,
) -> None:
    """The public spectrum API rejects complex arrays before float coercion."""
    kwargs: dict[str, object] = {
        "phases_init": np.array([0.0, 0.1], dtype=np.float64),
        "omegas": np.array([1.0, 1.0], dtype=np.float64),
        "knm": _matrix(),
        "alpha": _alpha(),
        "n_steps": 1,
    }
    kwargs[field] = payload

    # Alias as Callable[..., FloatArray]: kwargs carries a deliberately invalid value.
    spectrum: Callable[..., FloatArray] = lyapunov_mod.lyapunov_spectrum
    with pytest.raises(ValueError, match=match):
        spectrum(**kwargs)


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


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n_steps", np.bool_(True), "n_steps must be an integer"),
        ("qr_interval", np.bool_(True), "qr_interval must be an integer"),
        ("zeta", np.bool_(True), "zeta must be a finite real"),
        ("psi", np.bool_(True), "psi must be a finite real"),
    ],
)
def test_lyapunov_spectrum_rejects_numpy_boolean_scalar_aliases(
    field: str,
    value: object,
    match: str,
) -> None:
    """The public spectrum API rejects NumPy boolean scalar aliases."""
    kwargs: dict[str, object] = {
        "phases_init": np.array([0.0, 0.1], dtype=np.float64),
        "omegas": np.array([1.0, 1.0], dtype=np.float64),
        "knm": _matrix(),
        "alpha": _alpha(),
        "n_steps": 1,
        "qr_interval": 1,
        "zeta": 0.0,
        "psi": 0.0,
    }
    kwargs[field] = value

    # Alias as Callable[..., FloatArray]: kwargs carries a deliberately invalid value.
    spectrum: Callable[..., FloatArray] = lyapunov_mod.lyapunov_spectrum
    with pytest.raises(ValueError, match=match):
        spectrum(**kwargs)


def test_python_spectrum_covers_driven_jacobian_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero driver exercises the forced Kuramoto Jacobian branch."""
    monkeypatch.setattr(
        lyapunov_mod, "_dispatch", lambda _backend="auto": ("python", None)
    )

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


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ([True, False], "must not contain boolean"),
        (["1.0", "0.0"], "must contain only real numbers"),
        ([1.0 + 0.0j, 0.0 + 0.0j], "must be real-valued"),
    ],
)
def test_backend_output_validation_rejects_lossy_alias_payloads(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    match: str,
) -> None:
    """Backend spectra reject lossy aliases before float coercion."""

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
        return value

    with pytest.raises(ValueError, match=match):
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
