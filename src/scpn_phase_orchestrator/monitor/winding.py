# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase winding number tracker

"""Cumulative winding-number tracker with a 5-backend fallback
chain per ``feedback_module_standard_attnres.md``.

``w_i = floor(Σ_t wrap(Δθ_{i,t}) / 2π)`` where ``wrap(x) ∈ (−π, π]``.
Counts how many full ``2π`` rotations each oscillator completes
across a phase history; positive = counterclockwise, negative =
clockwise.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "winding_numbers",
    "winding_vector",
]

TWO_PI = 2.0 * np.pi


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., IntArray]:
    from spo_kernel import winding_numbers as _rust_wind

    def _rust(phases_flat: FloatArray, t: int, n: int) -> IntArray:
        # Rust FFI takes the flat array and infers T from length.
        return np.asarray(_rust_wind(phases_flat, int(n)), dtype=np.int64)

    return cast("Callable[..., IntArray]", _rust)


def _load_mojo_fn() -> Callable[..., IntArray]:
    from ..experimental.accelerators.monitor._winding_mojo import (
        _ensure_exe,
        winding_numbers_mojo,
    )

    _ensure_exe()
    return winding_numbers_mojo


def _load_julia_fn() -> Callable[..., IntArray]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._winding_julia import (
        winding_numbers_julia,
    )

    return winding_numbers_julia


def _load_go_fn() -> Callable[..., IntArray]:
    from ..experimental.accelerators.monitor._winding_go import (
        _load_lib,
        winding_numbers_go,
    )

    _load_lib()
    return winding_numbers_go


_LOADERS: dict[str, Callable[[], Callable[..., IntArray]]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, Callable[..., IntArray]] = {}


def _load_backend(name: str) -> Callable[..., IntArray]:
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
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch() -> Callable[..., IntArray] | None:
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


def _validate_phase_history(phases_history: object) -> FloatArray:
    raw = np.asarray(phases_history)
    if _contains_boolean_alias(phases_history):
        raise ValueError("phases_history must not contain boolean values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases_history must be a numeric array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError("phases_history must contain only finite values")
    if array.ndim > 2:
        raise ValueError(f"phases_history must be 1D or 2D, got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_backend_winding(value: object, *, n: int, t: int) -> IntArray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("backend winding output must be array-like") from exc
    if array.shape != (n,):
        raise ValueError(f"backend winding output shape {array.shape} must be ({n},)")
    try:
        numeric = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("backend winding output must be numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("backend winding output must contain only finite values")
    if not np.all(np.equal(numeric, np.floor(numeric))):
        raise ValueError("backend winding output must contain integer values")
    if np.any(np.abs(numeric) > t):
        raise ValueError("backend winding output exceeds trajectory length bound")
    return np.ascontiguousarray(numeric.astype(np.int64), dtype=np.int64)


def winding_numbers(phases_history: FloatArray) -> IntArray:
    """Cumulative winding number of each oscillator over a trajectory.

    ``w_i = floor(Σ_t wrap(Δθ_{i,t}) / 2π)`` with
    ``wrap(x) ∈ (−π, π]``.

    Args:
        phases_history: ``(T, N)`` phases in radians.

    Returns:
        ``(N,)`` int64 array of winding numbers.
    """
    phases_history = _validate_phase_history(phases_history)
    if phases_history.ndim != 2 or phases_history.shape[0] < 2:
        n = phases_history.shape[-1] if phases_history.ndim == 2 else 0
        return np.zeros(n, dtype=np.int64)

    t, n = int(phases_history.shape[0]), int(phases_history.shape[1])
    flat: FloatArray = np.ascontiguousarray(phases_history.ravel(), dtype=np.float64)

    backend_fn = _dispatch()
    if backend_fn is not None:
        try:
            return _validate_backend_winding(backend_fn(flat, t, n), n=n, t=t)
        except Exception:
            n = int(n)

    # Unwrap via cumulative phase increments to handle wrap-around correctly.
    dtheta = np.diff(phases_history, axis=0)
    dtheta_wrapped = (dtheta + np.pi) % TWO_PI - np.pi
    cumulative = np.sum(dtheta_wrapped, axis=0)
    return cast("IntArray", np.floor(cumulative / TWO_PI).astype(np.int64))


def winding_vector(phases_history: FloatArray) -> IntArray:
    """N-dimensional integer classification vector from winding numbers.

    Alias for :func:`winding_numbers`; topologically distinct
    trajectories map to distinct integer-lattice points.
    """
    return winding_numbers(phases_history)
