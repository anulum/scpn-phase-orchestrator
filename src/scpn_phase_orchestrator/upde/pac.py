# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-amplitude coupling

"""Tort 2010 phase-amplitude coupling with 5-backend fallback chain.

Follows ``feedback_module_standard_attnres.md``:

* ``modulation_index`` — scalar Tort 2010 MI on a single
  (θ_low, a_high) pair of time series.
* ``pac_matrix`` — ``(N, N)`` pairwise MI matrix over ``N``
  oscillator phase / amplitude channels.
* ``pac_gate`` — pure-Python boolean gate on an MI value (no
  backend dispatch needed — trivial comparison).

All compute kernels are available in Rust, Mojo, Julia, Go, Python.
``AVAILABLE_BACKENDS`` reports detected backends in canonical fallback order,
while ``ACTIVE_BACKEND`` is selected by a small import-time hot-path probe so
slow external wrappers do not displace the faster local path.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from numbers import Integral, Real
from time import perf_counter
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "modulation_index",
    "pac_matrix",
    "pac_gate",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import pac_matrix_compute, pac_modulation_index

    return {
        "modulation_index": pac_modulation_index,
        "pac_matrix": pac_matrix_compute,
    }


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    from ..experimental.accelerators.upde._pac_mojo import (
        _ensure_exe,
        modulation_index_mojo,
        pac_matrix_mojo,
    )

    _ensure_exe()
    return {
        "modulation_index": modulation_index_mojo,
        "pac_matrix": pac_matrix_mojo,
    }


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    import juliacall  # noqa: F401

    from ..experimental.accelerators.upde._pac_julia import (
        modulation_index_julia,
        pac_matrix_julia,
    )

    return {
        "modulation_index": modulation_index_julia,
        "pac_matrix": pac_matrix_julia,
    }


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    from ..experimental.accelerators.upde._pac_go import (
        _load_lib,
        modulation_index_go,
        pac_matrix_go,
    )

    _load_lib()
    return {
        "modulation_index": modulation_index_go,
        "pac_matrix": pac_matrix_go,
    }


_LOADERS: dict[str, Callable[[], dict[str, object]]] = {
    "rust": _load_rust_fns,
    "mojo": _load_mojo_fns,
    "julia": _load_julia_fns,
    "go": _load_go_fns,
}
_BACKEND_CACHE: dict[str, dict[str, object]] = {}


def _validate_n_bins(n_bins: object) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral):
        raise ValueError("n_bins must be an integer >= 2")
    bins = int(n_bins)
    if bins < 2:
        return bins
    return bins


def _validate_signal(name: str, value: FloatArray) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_history(
    name: str,
    value: FloatArray,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != 2:
        raise ValueError("phases_history and amplitudes_history must be 2-D")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _validate_mi_value(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return float(np.clip(parsed, 0.0, 1.0))


def _load_backend(name: str) -> dict[str, object]:
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _modulation_index_python(
    theta_low: FloatArray,
    amp_high: FloatArray,
    n_bins: int,
) -> float:
    n = min(theta_low.size, amp_high.size)
    theta = theta_low[:n] % (2.0 * np.pi)
    amp = amp_high[:n]
    bin_edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    bin_indices = np.digitize(theta, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    sums = np.bincount(bin_indices, weights=amp, minlength=n_bins)
    counts = np.bincount(bin_indices, minlength=n_bins)
    mean_amp = np.divide(
        sums,
        counts,
        out=np.zeros(n_bins, dtype=np.float64),
        where=counts > 0,
    )
    total = mean_amp.sum()
    if total <= 0.0:
        return 0.0
    p = mean_amp / total
    log_n = np.log(n_bins)
    if log_n < 1e-15:
        return 0.0
    positive = p > 0.0
    kl = float(np.sum(p[positive] * np.log(p[positive] * n_bins)))
    mi = kl / log_n
    return float(np.clip(mi, 0.0, 1.0))


def _resolve_backends() -> tuple[str, list[str]]:
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    active = min(available, key=_modulation_index_probe_seconds)
    return active, available


def _modulation_index_probe_seconds(name: str) -> float:
    theta = np.linspace(0.0, 2.0 * np.pi, 1000, dtype=np.float64)
    amp = np.abs(np.sin(theta)) + 0.1
    start = perf_counter()
    try:
        if name == "python":
            _modulation_index_python(theta, amp, 18)
        else:
            fn = cast(
                "Callable[[FloatArray, FloatArray, int], float]",
                _load_backend(name)["modulation_index"],
            )
            fn(theta, amp, 18)
    except (ImportError, RuntimeError, OSError, KeyError):
        return float("inf")
    return perf_counter() - start


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch(fn_name: str) -> object:
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    seen: set[str] = set()
    for backend in ordered_backends:
        if backend in seen:
            continue
        seen.add(backend)
        if backend == "python":
            return None
        try:
            fn = _load_backend(backend).get(fn_name)
        except (ImportError, RuntimeError, OSError):
            continue
        if fn is None:
            continue
        return fn
    return None


def modulation_index(
    theta_low: FloatArray, amp_high: FloatArray, n_bins: int = 18
) -> float:
    """Phase-amplitude coupling via Tort et al. 2010, J. Neurophysiol.

    Bins amplitude by phase, computes KL divergence from uniform,
    returns the modulation index normalised to ``[0, 1]`` by
    ``log(n_bins)``.
    """
    n_bins = _validate_n_bins(n_bins)
    theta_low = _validate_signal("theta_low", theta_low)
    amp_high = _validate_signal("amp_high", amp_high)
    if n_bins < 2:
        return 0.0
    if theta_low.size == 0 or amp_high.size == 0:
        return 0.0

    backend_fn = _dispatch("modulation_index")
    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, FloatArray, int], float]",
            backend_fn,
        )
        return _validate_mi_value(
            "modulation_index backend",
            fn(
                np.ascontiguousarray(theta_low, dtype=np.float64),
                np.ascontiguousarray(amp_high, dtype=np.float64),
                n_bins,
            ),
        )

    return _validate_mi_value(
        "modulation_index",
        _modulation_index_python(theta_low, amp_high, n_bins),
    )


def pac_matrix(
    phases_history: FloatArray,
    amplitudes_history: FloatArray,
    n_bins: int = 18,
) -> FloatArray:
    """``(N, N)`` PAC matrix. Entry ``[i, j]`` is
    ``MI(phase_i, amplitude_j)`` over the ``T`` timesteps.

    Args:
        phases_history: ``(T, N)`` phase time series.
        amplitudes_history: ``(T, N)`` amplitude time series.
        n_bins: number of phase bins.
    """
    n_bins = _validate_n_bins(n_bins)
    phases_history = _validate_history("phases_history", phases_history)
    amplitudes_history = _validate_history("amplitudes_history", amplitudes_history)
    t, n = phases_history.shape
    if n_bins < 2:
        return np.zeros((n, n), dtype=np.float64)
    if t == 0 or n == 0:
        return np.zeros((n, n), dtype=np.float64)
    if amplitudes_history.shape != (t, n):
        raise ValueError("phases and amplitudes must have the same shape")

    backend_fn = _dispatch("pac_matrix")
    if backend_fn is not None:
        fn = cast(
            ("Callable[[FloatArray, FloatArray, int, int, int], FloatArray]"),
            backend_fn,
        )
        flat = fn(
            np.ascontiguousarray(phases_history.ravel(order="C"), dtype=np.float64),
            np.ascontiguousarray(amplitudes_history.ravel(order="C"), dtype=np.float64),
            t,
            n,
            n_bins,
        )
        matrix = np.asarray(flat, dtype=np.float64).ravel(order="C")
        if matrix.size != n * n:
            raise ValueError(
                "pac_matrix backend must return n*n values in C-order layout"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("pac_matrix backend must return finite values")
        return np.clip(matrix.reshape((n, n), order="C"), 0.0, 1.0)

    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            result[i, j] = modulation_index(
                phases_history[:, i], amplitudes_history[:, j], n_bins
            )
    return result


def pac_gate(pac_value: float, threshold: float = 0.3) -> bool:
    """Binary gate: ``True`` when PAC exceeds ``threshold``.

    Pure-Python helper; no dispatcher — the comparison is trivial.
    """
    pac_value = _validate_finite_real("pac_value", pac_value)
    threshold = _validate_finite_real("threshold", threshold)
    return pac_value >= threshold
