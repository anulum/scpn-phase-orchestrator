# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Financial market Kuramoto regime detection

"""Kuramoto-based financial market synchronisation analysis.

Exposes a 5-backend fallback chain.

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

from scpn_phase_orchestrator.experimental.accelerators.upde._market_validation import (
    validate_market_order_output,
    validate_market_plv_output,
)
from scpn_phase_orchestrator.upde._julia_runtime import require_juliacall_main

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
    """Load the Rust market backend callable."""
    from spo_kernel import market_order_parameter_rust, market_plv_rust

    def _rust_op(phases_flat: FloatArray, t: int, n: int) -> FloatArray:
        """Call the Rust market order-parameter kernel."""
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
        """Call the Rust market phase-locking-value kernel."""
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
    """Load the Mojo market backend callable."""
    from ..experimental.accelerators.upde._market_mojo import (
        _ensure_exe,
        market_order_parameter_mojo,
        market_plv_mojo,
    )

    _ensure_exe()
    return market_order_parameter_mojo, market_plv_mojo


def _load_julia_fn() -> tuple[_MarketFn, _MarketFn]:
    # pragma: no cover — toolchain
    """Load the Julia market backend callable."""
    require_juliacall_main()

    from ..experimental.accelerators.upde._market_julia import (
        market_order_parameter_julia,
        market_plv_julia,
    )

    return market_order_parameter_julia, market_plv_julia


def _load_go_fn() -> tuple[_MarketFn, _MarketFn]:
    # pragma: no cover — toolchain
    """Load the Go market backend callable."""
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
_BACKEND_CACHE: dict[str, tuple[_MarketFn, _MarketFn]] = {}


def _load_backend(name: str) -> tuple[_MarketFn, _MarketFn]:
    """Load and cache the named backend callable."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve the active and available backends, fastest-first."""
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


def _dispatch() -> tuple[_MarketFn, _MarketFn] | None:
    """Return the fastest available backend callable, or ``None`` for Python."""
    if ACTIVE_BACKEND == "python":
        return None
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    seen: set[str] = set()
    for backend in ordered_backends:
        if backend in seen:
            continue
        seen.add(backend)
        if backend == "python":
            continue
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
    return None


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    return coerced


def _validate_series(value: object) -> FloatArray:
    """Return the price/return series as a validated finite array, else raise."""
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
    """Return the phase matrix as a validated 2-D finite array, else raise."""
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
    """Return the signal as a validated 1-D finite array, else raise."""
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
    """Extract instantaneous phase from a time series via the Hilbert transform.

    Stays Python-side because the transform is FFT-based
    (``scipy.signal.hilbert``) and the compiled backends do not
    ship an FFT library.

    Parameters
    ----------
    series : FloatArray
        Real-valued time series, shape ``(T,)``.

    Returns
    -------
    FloatArray
        The instantaneous phase of the series in ``[0, 2π)``.
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
    """Return the reference market order parameter (NumPy floor)."""
    phases = phases_flat.reshape(t, n)
    z = np.exp(1j * phases)
    R: FloatArray = np.abs(np.mean(z, axis=1))
    return np.ascontiguousarray(R, dtype=np.float64)


def market_order_parameter(phases: FloatArray) -> FloatArray:
    """Return the Kuramoto order parameter ``R(t)`` across ``N`` assets.

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.

    Returns
    -------
    FloatArray
        The Kuramoto order parameter time series ``R(t)``.
    """
    phases = _validate_phase_matrix(phases)
    T, N = phases.shape
    if T == 0:
        return np.empty(0, dtype=np.float64)
    flat = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    dispatched = _dispatch()
    if dispatched is not None:
        op_fn, _ = dispatched
        return validate_market_order_output(op_fn(flat, T, N), t=T)
    return _python_market_order_parameter(flat, T, N)


def _python_market_plv(
    phases_flat: FloatArray,
    t: int,
    n: int,
    window: int,
) -> FloatArray:
    """Return the reference market phase-locking value (NumPy floor)."""
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
    """Compute the rolling phase-locking-value matrix between assets.

    Returns shape ``(T − window + 1, N, N)``.

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.
    window : int
        Sliding-window length in samples.

    Returns
    -------
    FloatArray
        The rolling phase-locking-value matrices, shape ``(T − window + 1, N, N)``.
    """
    phases = _validate_phase_matrix(phases)
    window = _validate_positive_int(window, name="window")
    T, N = phases.shape
    if window > T or N == 0:
        return np.empty((0, N, N), dtype=np.float64)
    flat = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    dispatched = _dispatch()
    if dispatched is not None:
        _, plv_fn = dispatched
        result_flat = validate_market_plv_output(
            plv_fn(flat, T, N, window),
            t=T,
            n=N,
            window=window,
        )
    else:
        result_flat = _python_market_plv(flat, T, N, window)
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

    Parameters
    ----------
    R : FloatArray
        Order-parameter time series ``R(t)``, shape ``(T,)``.
    sync_threshold : float
        Order parameter above which the market is classed as synchronised.
    desync_threshold : float
        Order parameter below which the market is classed as desynchronised.

    Returns
    -------
    IntArray
        The per-timestep market regime labels.

    Raises
    ------
    ValueError
        If the thresholds are inconsistent or ``R`` is not 1-D.
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
    """Detect synchronisation warnings where smoothed ``R`` crosses up.

    Parameters
    ----------
    R : FloatArray
        Order-parameter time series ``R(t)``, shape ``(T,)``.
    threshold : float
        Decision threshold.
    lookback : int
        Number of past samples smoothed over before the crossing test.

    Returns
    -------
    BoolArray
        A per-timestep boolean mask of synchronisation warnings.
    """
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
