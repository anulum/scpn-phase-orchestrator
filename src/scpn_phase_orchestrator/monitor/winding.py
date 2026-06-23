# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase winding number tracker

"""Cumulative winding-number tracker with a 5-backend fallback chain.

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
    """Load the Rust winding-number backend callable."""
    from spo_kernel import winding_numbers as _rust_wind

    def _rust(phases_flat: FloatArray, t: int, n: int) -> IntArray:
        # Rust FFI takes the flat array and infers T from length.
        """Call the Rust winding-number kernel with contiguous float arrays."""
        return cast("IntArray", np.asarray(_rust_wind(phases_flat, int(n))))

    return cast("Callable[..., IntArray]", _rust)


def _load_mojo_fn() -> Callable[..., IntArray]:
    """Load the Mojo winding-number backend callable."""
    from ..experimental.accelerators.monitor._winding_mojo import (
        _ensure_exe,
        winding_numbers_mojo,
    )

    _ensure_exe()
    return winding_numbers_mojo


def _load_julia_fn() -> Callable[..., IntArray]:
    """Load the Julia winding-number backend callable."""
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._winding_julia import (
        winding_numbers_julia,
    )

    return winding_numbers_julia


def _load_go_fn() -> Callable[..., IntArray]:
    """Load the Go winding-number backend callable."""
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
    """Load and cache the named backend callable."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve the active and available backends, fastest-first."""
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


def _dispatch() -> Callable[..., IntArray] | None:
    """Return the fastest available backend callable, or ``None`` for Python."""
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
    """Return whether the value contains any boolean alias."""
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in array.flat)


def _has_complex_payload(value: object) -> bool:
    """Return whether the value carries a complex-number payload."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return _contains_complex_alias(value)
    return bool(np.iscomplexobj(array) or _contains_complex_alias(value))


def _validate_phase_history(phases_history: object) -> FloatArray:
    """Return the phase history as a validated 2-D finite array, else raise."""
    raw = np.asarray(phases_history)
    if _contains_boolean_alias(phases_history):
        raise ValueError("phases_history must not contain boolean values")
    if _has_complex_payload(phases_history):
        raise ValueError("phases_history must contain real-valued samples")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases_history must be a numeric array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError("phases_history must contain only finite values")
    if array.ndim > 2:
        raise ValueError(f"phases_history must be 1D or 2D, got shape {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float64)


def _winding_reference(phases_history: FloatArray) -> IntArray:
    """Return the reference cumulative winding numbers (NumPy floor)."""
    dtheta = np.diff(phases_history, axis=0)
    # Wrap each increment into the half-open interval (-π, π] so an exact
    # forward half-turn (+π) counts forward rather than aliasing to -π.
    dtheta_wrapped = np.pi - (np.pi - dtheta) % TWO_PI
    cumulative = np.sum(dtheta_wrapped, axis=0)
    return cast("IntArray", np.floor(cumulative / TWO_PI).astype(np.int64))


def _validate_backend_winding(
    value: object,
    *,
    n: int,
    t: int,
    expected: IntArray | None = None,
) -> IntArray:
    """Return backend winding numbers matching the reference, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError("backend winding output must not contain boolean values")
    if _has_complex_payload(value):
        raise ValueError("backend winding output must be real-valued")
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
    max_abs_winding = int(np.ceil(max(t - 1, 0) / 2.0))
    if np.any(np.abs(numeric) > max_abs_winding):
        raise ValueError("backend winding output exceeds wrapped-increment bound")
    winding = np.ascontiguousarray(numeric.astype(np.int64), dtype=np.int64)
    if expected is not None:
        reference = np.asarray(expected, dtype=np.int64)
        if reference.shape != winding.shape:
            raise ValueError("exact winding reference shape must match backend output")
        if not np.array_equal(winding, reference):
            raise ValueError(
                "backend winding output diverged from exact winding reference"
            )
    return winding


def winding_numbers(phases_history: FloatArray) -> IntArray:
    """Cumulative winding number of each oscillator over a trajectory.

    ``w_i = floor(Σ_t wrap(Δθ_{i,t}) / 2π)`` with
    ``wrap(x) ∈ (−π, π]``.

    Parameters
    ----------
    phases_history : FloatArray
        ``(T, N)`` phases in radians.

    Returns
    -------
    IntArray
        ``(N,)`` int64 array of winding numbers.
    """
    phases_history = _validate_phase_history(phases_history)
    if phases_history.ndim != 2 or phases_history.shape[0] < 2:
        n = phases_history.shape[-1] if phases_history.ndim == 2 else 0
        return np.zeros(n, dtype=np.int64)

    t, n = int(phases_history.shape[0]), int(phases_history.shape[1])
    flat: FloatArray = np.ascontiguousarray(phases_history.ravel(), dtype=np.float64)
    expected = _winding_reference(phases_history)

    backend_fn = _dispatch()
    if backend_fn is not None:
        try:
            return _validate_backend_winding(
                backend_fn(flat, t, n),
                n=n,
                t=t,
                expected=expected,
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            n = int(n)

    return _validate_backend_winding(expected, n=n, t=t, expected=expected)


def winding_vector(phases_history: FloatArray) -> IntArray:
    """N-dimensional integer classification vector from winding numbers.

    Alias for :func:`winding_numbers`; topologically distinct
    trajectories map to distinct integer-lattice points.

    Parameters
    ----------
    phases_history : FloatArray
        Phase history, shape ``(T, N)``.

    Returns
    -------
    IntArray
        The integer winding classification vector.
    """
    return winding_numbers(phases_history)
