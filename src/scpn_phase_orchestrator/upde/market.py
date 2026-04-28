# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Financial market Kuramoto regime detection

"""Kuramoto-based financial market synchronisation analysis with a
5-backend fallback chain per ``feedback_module_standard_attnres.md``.

Extracts instantaneous phase from price / return time series via the
Hilbert transform (``scipy.signal.hilbert`` — FFT-based, stays
Python-side because the Rust/Go/Mojo backends do not ship an FFT),
then dispatches the two post-processing compute kernels:

* ``market_order_parameter(phases)`` — ``R(t) = |⟨exp(iθ)⟩_N|`` at
  every timestep. ``O(T · N)``.
* ``market_plv(phases, window)`` — rolling phase-locking-value
  matrix between assets, ``O((T − W + 1) · N² · W)`` with a sincos
  precompute that eliminates trig from the inner loop.

The ``detect_regimes`` classifier and ``sync_warning`` crossing
detector are O(T) masking / comparison operations; they stay pure
NumPy. ``R(t) → 1`` preceded Black Monday 1987 and the 2008
crash (arXiv:1109.1167; CEUR-WS Vol-915).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "detect_regimes",
    "extract_phase",
    "market_order_parameter",
    "market_plv",
    "sync_warning",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> tuple[Callable[..., NDArray], Callable[..., NDArray]]:
    from spo_kernel import market_order_parameter_rust, market_plv_rust

    def _rust_op(phases_flat: NDArray, t: int, n: int) -> NDArray:
        return np.asarray(
            market_order_parameter_rust(
                np.ascontiguousarray(phases_flat, dtype=np.float64),
                int(t),
                int(n),
            ),
            dtype=np.float64,
        )

    def _rust_plv(
        phases_flat: NDArray,
        t: int,
        n: int,
        window: int,
    ) -> NDArray:
        return np.asarray(
            market_plv_rust(
                np.ascontiguousarray(phases_flat, dtype=np.float64),
                int(t),
                int(n),
                int(window),
            ),
            dtype=np.float64,
        )

    return _rust_op, _rust_plv


def _load_mojo_fn() -> tuple[Callable[..., NDArray], Callable[..., NDArray]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._market_mojo import (
        _ensure_exe,
        market_order_parameter_mojo,
        market_plv_mojo,
    )

    _ensure_exe()
    return market_order_parameter_mojo, market_plv_mojo


def _load_julia_fn() -> tuple[Callable[..., NDArray], Callable[..., NDArray]]:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.upde._market_julia import (
        market_order_parameter_julia,
        market_plv_julia,
    )

    return market_order_parameter_julia, market_plv_julia


def _load_go_fn() -> tuple[Callable[..., NDArray], Callable[..., NDArray]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._market_go import (
        _load_lib,
        market_order_parameter_go,
        market_plv_go,
    )

    _load_lib()
    return market_order_parameter_go, market_plv_go


_LOADERS: dict[
    str, Callable[[], tuple[Callable[..., NDArray], Callable[..., NDArray]]]
] = {
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


def _dispatch() -> tuple[Callable[..., NDArray], Callable[..., NDArray]] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


def extract_phase(series: NDArray) -> NDArray:
    """Extract instantaneous phase from a time series via the
    Hilbert transform. Shape-preserving; output in ``[0, 2π)``.

    Stays Python-side because the transform is FFT-based
    (``scipy.signal.hilbert``) and the compiled backends do not
    ship an FFT library.
    """
    analytic = hilbert(series, axis=0)
    phase: NDArray = np.angle(analytic) % (2.0 * np.pi)
    return phase


def _python_market_order_parameter(
    phases_flat: NDArray,
    t: int,
    n: int,
) -> NDArray:
    if t == 0 or n == 0:
        return np.empty(0, dtype=np.float64)
    phases = phases_flat.reshape(t, n)
    z = np.exp(1j * phases)
    R: NDArray = np.abs(np.mean(z, axis=1))
    return np.ascontiguousarray(R, dtype=np.float64)


def market_order_parameter(phases: NDArray) -> NDArray:
    """Kuramoto order parameter ``R(t)`` across ``N`` assets at
    every timestep."""
    phases = np.asarray(phases, dtype=np.float64)
    if phases.ndim != 2:
        raise ValueError(f"phases must be (T, N), got {phases.shape}")
    T, N = phases.shape
    flat = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    dispatched = _dispatch()
    if dispatched is not None:
        op_fn, _ = dispatched
        return np.asarray(op_fn(flat, T, N), dtype=np.float64)
    return _python_market_order_parameter(flat, T, N)


def _python_market_plv(
    phases_flat: NDArray,
    t: int,
    n: int,
    window: int,
) -> NDArray:
    if t < window or n == 0 or window == 0:
        return np.empty(0, dtype=np.float64)
    phases = phases_flat.reshape(t, n)
    n_windows = t - window + 1
    out = np.empty(n_windows * n * n, dtype=np.float64)
    inv_w = 1.0 / window
    for w in range(n_windows):
        chunk = phases[w : w + window]  # (window, n)
        s = np.sin(chunk)
        c = np.cos(chunk)
        # cos(θ_j − θ_i) = c_j·c_i + s_j·s_i → sum over window
        # sin(θ_j − θ_i) = s_j·c_i − c_j·s_i → sum over window
        sum_cos = (c.T @ c + s.T @ s) * inv_w
        sum_sin = (s.T @ c - c.T @ s) * inv_w
        plv_mat = np.sqrt(sum_cos * sum_cos + sum_sin * sum_sin)
        out[w * n * n : (w + 1) * n * n] = plv_mat.ravel()
    return out


def market_plv(phases: NDArray, window: int = 50) -> NDArray:
    """Rolling phase-locking-value matrix between assets.

    Returns shape ``(T − window + 1, N, N)``.
    """
    phases = np.asarray(phases, dtype=np.float64)
    if phases.ndim != 2:
        raise ValueError(f"phases must be (T, N), got {phases.shape}")
    T, N = phases.shape
    flat = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    dispatched = _dispatch()
    if dispatched is not None:
        _, plv_fn = dispatched
        result_flat = np.asarray(plv_fn(flat, T, N, int(window)), dtype=np.float64)
    else:
        result_flat = _python_market_plv(flat, T, N, int(window))
    if result_flat.size == 0:
        return result_flat.reshape(0, N, N)
    n_windows = T - int(window) + 1
    return result_flat.reshape(n_windows, N, N)


def detect_regimes(
    R: NDArray,
    sync_threshold: float = 0.7,
    desync_threshold: float = 0.3,
) -> NDArray:
    """Classify market synchronisation regimes from ``R(t)``.

    Returns ``int32`` labels: 0 = desynchronised, 1 = transition,
    2 = synchronised. O(T) masking; no multi-language port needed.
    """
    R = np.asarray(R, dtype=np.float64)
    try:
        from spo_kernel import detect_regimes_rust as _rust_regimes

        flat = np.ascontiguousarray(R.ravel())
        return np.asarray(_rust_regimes(flat, sync_threshold, desync_threshold))
    except ImportError:
        pass
    regimes = np.ones(len(R), dtype=np.int32)
    mask_sync = sync_threshold <= R
    mask_desync = desync_threshold >= R
    regimes[mask_sync] = 2
    regimes[mask_desync] = 0
    return regimes


def sync_warning(
    R: NDArray,
    threshold: float = 0.7,
    lookback: int = 10,
) -> NDArray:
    """Detect synchronisation warning signals — timesteps where
    the smoothed ``R`` crosses the threshold from below."""
    if lookback > 1:
        kernel = np.ones(lookback) / lookback
        R_smooth = np.convolve(R, kernel, mode="same")
    else:
        R_smooth = R
    warnings = np.zeros(len(R), dtype=bool)
    for t in range(1, len(R)):
        if R_smooth[t] >= threshold and R_smooth[t - 1] < threshold:
            warnings[t] = True
    return warnings
