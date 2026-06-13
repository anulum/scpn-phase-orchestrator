# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Thermodynamic entropy production rate

"""Overdamped-Kuramoto thermodynamic dissipation rate with a 5-backend
fallback chain per ``feedback_module_standard_attnres.md``.

    Σ = Σ_i (dθ_i/dt)² · dt
    dθ_i/dt = ω_i + (α / N) · Σ_j K_ij · sin(θ_j − θ_i)

Zero at frequency-locked fixed points; positive otherwise.
Reference: Acebrón et al. 2005, Rev. Mod. Phys. 77:137–185.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "entropy_production_rate",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., float]:
    from spo_kernel import entropy_production_rate as _rust_ep

    def _rust(
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: float,
        dt: float,
    ) -> float:
        return float(
            _rust_ep(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                float(alpha),
                float(dt),
            )
        )

    return cast("Callable[..., float]", _rust)


def _load_mojo_fn() -> Callable[..., float]:
    from ..experimental.accelerators.monitor._entropy_prod_mojo import (
        _ensure_exe,
        entropy_production_rate_mojo,
    )

    _ensure_exe()
    return entropy_production_rate_mojo


def _load_julia_fn() -> Callable[..., float]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._entropy_prod_julia import (
        entropy_production_rate_julia,
    )

    return entropy_production_rate_julia


def _load_go_fn() -> Callable[..., float]:
    from ..experimental.accelerators.monitor._entropy_prod_go import (
        _load_lib,
        entropy_production_rate_go,
    )

    _load_lib()
    return entropy_production_rate_go


_LOADERS: dict[str, Callable[[], Callable[..., float]]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, Callable[..., float]] = {}


def _load_backend(name: str) -> Callable[..., float]:
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    _BACKEND_CACHE.clear()
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch() -> Callable[..., float] | None:
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    deduped: list[str] = []
    for backend in ordered_backends:
        if backend in deduped:
            continue
        deduped.append(backend)
    for backend in deduped:
        if backend == "python":
            return None
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
    return None


def _contains_boolean_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _contains_complex_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in array.flat)


def _has_complex_payload(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return _contains_complex_alias(value)
    return bool(np.iscomplexobj(array) or _contains_complex_alias(value))


def _validate_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or _contains_boolean_alias(value):
        raise ValueError(f"{name} must not be a boolean value")
    if _has_complex_payload(value):
        raise ValueError(f"{name} must be a finite real-valued scalar")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _has_complex_payload(value):
        raise ValueError(f"{name} must contain real-valued samples")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional float array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} shape {array.shape} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_matrix(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, int],
) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _has_complex_payload(value):
        raise ValueError(f"{name} must contain real-valued couplings")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a two-dimensional float array") from exc
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_entropy_rate(value: object, *, name: str = "entropy rate") -> float:
    rate = _validate_finite_float(value, name=name)
    if rate < -1e-12:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return max(rate, 0.0)


def entropy_production_rate(
    phases: object,
    omegas: object,
    knm: object,
    alpha: object,
    dt: object,
) -> float:
    """Thermodynamic dissipation rate ``Σ (dθ/dt)² · dt``.

    ``dθ_i/dt = ω_i + (α / N) Σ_j K_ij sin(θ_j − θ_i)``. Zero at
    frequency-locked fixed points; positive otherwise.

    Acebrón et al. 2005, Rev. Mod. Phys. **77**:137–185.

    Args:
        phases: ``(N,)`` instantaneous phases in radians.
        omegas: ``(N,)`` natural frequencies.
        knm: ``(N, N)`` coupling matrix.
        alpha: global coupling strength.
        dt: integration timestep for the ``· dt`` factor.

    Returns:
        Non-negative dissipation scalar.
    """
    phases = _validate_vector(phases, name="phases")
    n = int(phases.size)
    omegas = _validate_vector(omegas, name="omegas")
    if omegas.shape != phases.shape:
        raise ValueError(f"omegas shape {omegas.shape} does not match {phases.shape}")
    knm = _validate_matrix(knm, name="knm", expected_shape=(n, n))
    alpha = _validate_finite_float(alpha, name="alpha")
    dt = _validate_finite_float(dt, name="dt")
    if dt < 0.0:
        raise ValueError(f"dt must be non-negative, got {dt!r}")
    if n == 0 or dt == 0.0:
        return 0.0
    backend_fn = _dispatch()
    if backend_fn is not None:
        try:
            backend_rate = backend_fn(phases, omegas, knm, alpha, dt)
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None
        else:
            return _validate_entropy_rate(
                backend_rate,
                name="backend entropy rate",
            )

    diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    coupling = np.sum(knm * np.sin(diff), axis=1)
    dtheta_dt = omegas + (alpha / n) * coupling
    return _validate_entropy_rate(np.sum(dtheta_dt**2) * dt)
