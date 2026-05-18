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
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int32]

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

_MarketFn = Callable[..., FloatArray]


def _load_rust_fn() -> tuple[_MarketFn, _MarketFn]:
    from spo_kernel import market_order_parameter_rust, market_plv_rust

    def _rust_op(phases_flat: FloatArray, t: int, n: int) -> FloatArray:
        return np.asarray(
            market_order_parameter_rust(
                np.ascontiguousarray(phases_flat, dtype=np.float64),
                int(t),
                int(n),
            ),
            dtype=np.float64,
        )

    def _rust_plv(
        phases_flat: FloatArray,
        t: int,
        n: int,
        window: int,
    ) -> FloatArray:
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


def _load_mojo_fn() -> tuple[_MarketFn, _MarketFn]:
    # pragma: no cover — toolchain
    from ..experimental.accelerators.upde._market_mojo import (
        _ensure_exe,
        market_order_parameter_mojo,
        market_plv_mojo,
    )

    _ensure_exe()
    return market_order_parameter_mojo, market_plv_mojo


def _load_julia_fn() -> tuple[_MarketFn, _MarketFn]:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401

    from ..experimental.accelerators.upde._market_julia import (
        market_order_parameter_julia,
        market_plv_julia,
    )

    return market_order_parameter_julia, market_plv_julia


def _load_go_fn() -> tuple[_MarketFn, _MarketFn]:
    # pragma: no cover — toolchain
    from ..experimental.accelerators.upde._market_go import (
        _load_lib,
        market_order_parameter_go,
        market_plv_go,
    )

    _load_lib()
    return market_order_parameter_go, market_plv_go


_LOADERS: dict[str, Callable[[], tuple[_MarketFn, _MarketFn]]] = {
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


def _dispatch() -> tuple[_MarketFn, _MarketFn] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    return coerced


def _validate_series(value: object) -> FloatArray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = "series must be a finite one- or two-dimensional array"
        raise ValueError(msg) from exc
    if arr.ndim not in (1, 2):
        raise ValueError(f"series shape {arr.shape} must be one- or two-dimensional")
    if arr.shape[0] < 1:
        raise ValueError("series must contain at least one timestep")
    if arr.ndim == 2 and arr.shape[1] < 1:
        raise ValueError("series must contain at least one channel")
    if not np.all(np.isfinite(arr)):
        raise ValueError("series must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_phase_matrix(value: object) -> FloatArray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a finite (T, N) array") from exc
    if arr.ndim != 2:
        raise ValueError(f"phases must be (T, N), got {arr.shape}")
    if arr.shape[1] < 1:
        raise ValueError("phases must contain at least one asset")
    if not np.all(np.isfinite(arr)):
        raise ValueError("phases must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_signal_vector(value: object, *, name: str) -> FloatArray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional array") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} shape {arr.shape} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def extract_phase(series: FloatArray) -> FloatArray:
    """Extract instantaneous phase from a time series via the
    Hilbert transform. Shape-preserving; output in ``[0, 2π)``.

    Stays Python-side because the transform is FFT-based
    (``scipy.signal.hilbert``) and the compiled backends do not
    ship an FFT library.
    """
    series = _validate_series(series)
    analytic = hilbert(series, axis=0)
    phase: FloatArray = np.angle(analytic) % (2.0 * np.pi)
    return phase


def _python_market_order_parameter(
    phases_flat: FloatArray,
    t: int,
    n: int,
) -> FloatArray:
    if t == 0 or n == 0:
        return np.empty(0, dtype=np.float64)
    phases = phases_flat.reshape(t, n)
    z = np.exp(1j * phases)
    R: FloatArray = np.abs(np.mean(z, axis=1))
    return np.ascontiguousarray(R, dtype=np.float64)


def market_order_parameter(phases: FloatArray) -> FloatArray:
    """Kuramoto order parameter ``R(t)`` across ``N`` assets at
    every timestep."""
    phases = _validate_phase_matrix(phases)
    T, N = phases.shape
    flat = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    dispatched = _dispatch()
    if dispatched is not None:
        op_fn, _ = dispatched
        return np.asarray(op_fn(flat, T, N), dtype=np.float64)
    return _python_market_order_parameter(flat, T, N)


def _python_market_plv(
    phases_flat: FloatArray,
    t: int,
    n: int,
    window: int,
) -> FloatArray:
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


def market_plv(phases: FloatArray, window: int = 50) -> FloatArray:
    """Rolling phase-locking-value matrix between assets.

    Returns shape ``(T − window + 1, N, N)``.
    """
    phases = _validate_phase_matrix(phases)
    window = _validate_positive_int(window, name="window")
    T, N = phases.shape
    flat = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    dispatched = _dispatch()
    if dispatched is not None:
        _, plv_fn = dispatched
        result_flat = np.asarray(plv_fn(flat, T, N, window), dtype=np.float64)
    else:
        result_flat = _python_market_plv(flat, T, N, window)
    if result_flat.size == 0:
        return result_flat.reshape(0, N, N)
    n_windows = T - window + 1
    return result_flat.reshape(n_windows, N, N)


def detect_regimes(
    R: FloatArray,
    sync_threshold: float = 0.7,
    desync_threshold: float = 0.3,
) -> IntArray:
    """Classify market synchronisation regimes from ``R(t)``.

    Returns ``int32`` labels: 0 = desynchronised, 1 = transition,
    2 = synchronised. O(T) masking; no multi-language port needed.
    """
    R = _validate_signal_vector(R, name="R")
    sync_threshold = _validate_finite_float(
        sync_threshold,
        name="sync_threshold",
    )
    desync_threshold = _validate_finite_float(
        desync_threshold,
        name="desync_threshold",
    )
    if sync_threshold < desync_threshold:
        raise ValueError(
            "sync_threshold must be greater than or equal to desync_threshold",
        )
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
    R: FloatArray,
    threshold: float = 0.7,
    lookback: int = 10,
) -> BoolArray:
    """Detect synchronisation warning signals — timesteps where
    the smoothed ``R`` crosses the threshold from below."""
    R = _validate_signal_vector(R, name="R")
    threshold = _validate_finite_float(threshold, name="threshold")
    lookback = _validate_positive_int(lookback, name="lookback")
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
