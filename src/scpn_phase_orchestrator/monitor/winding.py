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
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "winding_numbers",
    "winding_vector",
]

TWO_PI = 2.0 * np.pi


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., NDArray]:
    from spo_kernel import winding_numbers as _rust_wind

    def _rust(phases_flat: NDArray, t: int, n: int) -> NDArray:
        # Rust FFI takes the flat array and infers T from length.
        return np.asarray(_rust_wind(phases_flat, int(n)), dtype=np.int64)

    return cast("Callable[..., NDArray]", _rust)


def _load_mojo_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._winding_mojo import (
        _ensure_exe,
        winding_numbers_mojo,
    )

    _ensure_exe()
    return winding_numbers_mojo


def _load_julia_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.monitor._winding_julia import (
        winding_numbers_julia,
    )

    return winding_numbers_julia


def _load_go_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._winding_go import (
        winding_numbers_go,
    )

    return winding_numbers_go


_LOADERS: dict[str, Callable[[], Callable[..., NDArray]]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}


def _resolve_backends() -> tuple[str, list[str]]:
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _LOADERS[name]()
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch() -> Callable[..., NDArray] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


def winding_numbers(phases_history: NDArray) -> NDArray:
    """Cumulative winding number of each oscillator over a trajectory.

    ``w_i = floor(Σ_t wrap(Δθ_{i,t}) / 2π)`` with
    ``wrap(x) ∈ (−π, π]``.

    Args:
        phases_history: ``(T, N)`` phases in radians.

    Returns:
        ``(N,)`` int64 array of winding numbers.
    """
    if phases_history.ndim != 2 or phases_history.shape[0] < 2:
        n = phases_history.shape[-1] if phases_history.ndim == 2 else 0
        return np.zeros(n, dtype=np.int64)

    t, n = int(phases_history.shape[0]), int(phases_history.shape[1])
    flat = np.ascontiguousarray(phases_history.ravel(), dtype=np.float64)

    backend_fn = _dispatch()
    if backend_fn is not None:
        return np.asarray(backend_fn(flat, t, n), dtype=np.int64)

    # Unwrap via cumulative phase increments to handle wrap-around correctly.
    dtheta = np.diff(phases_history, axis=0)
    dtheta_wrapped = (dtheta + np.pi) % TWO_PI - np.pi
    cumulative = np.sum(dtheta_wrapped, axis=0)
    return cast("NDArray", np.floor(cumulative / TWO_PI).astype(np.int64))


def winding_vector(phases_history: NDArray) -> NDArray:
    """N-dimensional integer classification vector from winding numbers.

    Alias for :func:`winding_numbers`; topologically distinct
    trajectories map to distinct integer-lattice points.
    """
    return winding_numbers(phases_history)
