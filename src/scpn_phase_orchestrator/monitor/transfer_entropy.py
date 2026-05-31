# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase Transfer Entropy

"""Phase transfer entropy via binned histograms with a 5-backend
fallback chain per ``feedback_module_standard_attnres.md``.

Two compute kernels:

* ``phase_transfer_entropy`` — scalar ``TE(X → Y)`` on a pair of
  equal-length phase series.
* ``transfer_entropy_matrix`` — ``(N, N)`` pairwise TE matrix over
  ``N`` oscillator trajectories.

Estimator: 1-step Markov-order conditional entropy difference

    TE(X → Y) = H(Y_{t+1} | Y_t) − H(Y_{t+1} | Y_t, X_t)

with phases wrapped to ``[0, 2π)`` and binned into ``n_bins``
equal-width intervals. Higher TE indicates stronger directional
coupling from source to target.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "phase_transfer_entropy",
    "transfer_entropy_matrix",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import (
        phase_transfer_entropy_rust,
        transfer_entropy_matrix_rust,
    )

    return {
        "phase_te": phase_transfer_entropy_rust,
        "te_matrix": transfer_entropy_matrix_rust,
    }


def _load_mojo_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._te_mojo import (
        _ensure_exe,
        phase_te_mojo,
        te_matrix_mojo,
    )

    _ensure_exe()
    return {"phase_te": phase_te_mojo, "te_matrix": te_matrix_mojo}


def _load_julia_fns() -> dict[str, object]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._te_julia import (
        phase_te_julia,
        te_matrix_julia,
    )

    return {"phase_te": phase_te_julia, "te_matrix": te_matrix_julia}


def _load_go_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._te_go import (
        _load_lib,
        phase_te_go,
        te_matrix_go,
    )

    _load_lib()
    return {"phase_te": phase_te_go, "te_matrix": te_matrix_go}


_LOADERS: dict[str, Callable[[], dict[str, object]]] = {
    "rust": _load_rust_fns,
    "mojo": _load_mojo_fns,
    "julia": _load_julia_fns,
    "go": _load_go_fns,
}
_BACKEND_CACHE: dict[str, dict[str, object]] = {}


def _load_backend(name: str) -> dict[str, object]:
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


def _dispatch(fn_name: str) -> object:
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
            fn = _load_backend(backend).get(fn_name)
        except (ImportError, RuntimeError, OSError):
            continue
        if fn is None:
            continue
        return fn
    return None


def _contains_boolean_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _validate_phase_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real-valued 1-D phase vector")
    try:
        phases = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 1-D phase vector") from exc
    if phases.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {phases.shape}")
    if not np.all(np.isfinite(phases)):
        raise ValueError(f"{name} must contain only finite values")
    return phases


def _validate_phase_series(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real-valued 2-D phase series")
    try:
        series = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 2-D phase series") from exc
    if series.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (oscillators, timesteps), got shape {series.shape}"
        )
    if series.shape[0] == 0 or series.shape[1] == 0:
        raise ValueError(
            f"{name} must contain at least one oscillator and one timestep, "
            f"got shape {series.shape}"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError(f"{name} must contain only finite values")
    return series


def _validate_n_bins(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("n_bins must be an integer greater than or equal to 2")
    if value < 2:
        raise ValueError("n_bins must be greater than or equal to 2")
    return int(value)


def _validate_te_scalar(
    value: object,
    *,
    name: str = "transfer entropy",
    max_entropy: float | None = None,
) -> float:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not be a boolean value")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative scalar") from exc
    if not np.isfinite(result) or result < -1e-12:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    if max_entropy is not None and result > max_entropy + 1e-12:
        raise ValueError(f"{name} must not exceed log(n_bins)")
    return max(result, 0.0)


def _validate_te_matrix(
    value: object,
    *,
    n_osc: int,
    max_entropy: float,
    expected: FloatArray | None = None,
    atol: float = 1e-12,
) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("transfer entropy matrix must not contain boolean values")
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("transfer entropy matrix must be numeric") from exc
    if matrix.size != n_osc * n_osc:
        raise ValueError(
            "transfer entropy matrix size must match oscillator count squared"
        )
    matrix = matrix.reshape(n_osc, n_osc)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transfer entropy matrix must contain only finite values")
    if np.any(matrix < -1e-12):
        raise ValueError("transfer entropy matrix must be non-negative")
    if np.any(matrix > max_entropy + 1e-12):
        raise ValueError("transfer entropy matrix entries must not exceed log(n_bins)")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("transfer entropy matrix diagonal must be zero")
    matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    if expected is not None:
        reference = np.asarray(expected, dtype=np.float64).reshape(n_osc, n_osc)
        if not np.allclose(matrix, reference, rtol=0.0, atol=atol):
            raise ValueError("transfer entropy matrix diverged from exact reference")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _conditional_entropy(
    target: IntArray, condition: IntArray, n_cond_bins: int
) -> float:
    """``H(target | condition)`` via histogram."""
    n = len(target)
    h = 0.0
    for c in range(n_cond_bins):
        mask = condition == c
        count = int(np.sum(mask))
        if count < 2:
            continue
        vals = target[mask]
        _, counts = np.unique(vals, return_counts=True)
        probs = counts / count
        h -= (count / n) * float(np.sum(probs * np.log(probs + 1e-30)))
    return h


def _phase_te_reference(
    source_values: FloatArray,
    target_values: FloatArray,
    bin_count: int,
) -> float:
    n_samples = min(len(source_values), len(target_values))
    if n_samples < 3:
        return 0.0
    source_values = source_values[:n_samples]
    target_values = target_values[:n_samples]
    n = n_samples - 1
    bins = np.linspace(0, 2 * np.pi, bin_count + 1)
    src_binned: IntArray = np.clip(
        np.digitize(source_values[:n] % (2 * np.pi), bins) - 1,
        0,
        bin_count - 1,
    ).astype(np.int64)
    tgt_binned: IntArray = np.clip(
        np.digitize(target_values[:n] % (2 * np.pi), bins) - 1,
        0,
        bin_count - 1,
    ).astype(np.int64)
    tgt_next: IntArray = np.clip(
        np.digitize(target_values[1 : n + 1] % (2 * np.pi), bins) - 1,
        0,
        bin_count - 1,
    ).astype(np.int64)
    h_y_yt = _conditional_entropy(tgt_next, tgt_binned, bin_count)
    joint_cond: IntArray = (tgt_binned * bin_count + src_binned).astype(np.int64)
    h_y_yt_x = _conditional_entropy(tgt_next, joint_cond, bin_count * bin_count)
    return _validate_te_scalar(
        h_y_yt - h_y_yt_x,
        max_entropy=float(np.log(bin_count)),
    )


def _te_matrix_reference(series: FloatArray, bin_count: int) -> FloatArray:
    n_osc, _n_time = series.shape
    te: FloatArray = np.zeros((n_osc, n_osc), dtype=np.float64)
    for i in range(n_osc):
        for j in range(n_osc):
            if i != j:
                te[i, j] = _phase_te_reference(series[i], series[j], bin_count)
    return te


def phase_transfer_entropy(
    source: FloatArray, target: FloatArray, n_bins: int = 16
) -> float:
    """Transfer entropy ``TE(X → Y)`` on binned phase series."""
    bin_count = _validate_n_bins(n_bins)
    source_values = _validate_phase_vector(source, name="source")
    target_values = _validate_phase_vector(target, name="target")
    n_samples = min(len(source_values), len(target_values))
    if n_samples < 3:
        return 0.0
    source_values = source_values[:n_samples]
    target_values = target_values[:n_samples]
    expected = _phase_te_reference(source_values, target_values, bin_count)
    backend_fn = _dispatch("phase_te")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, FloatArray, int], float]", backend_fn)
        try:
            result = fn(
                np.ascontiguousarray(source_values, dtype=np.float64),
                np.ascontiguousarray(target_values, dtype=np.float64),
                bin_count,
            )
            result = _validate_te_scalar(
                result,
                name="backend transfer entropy",
                max_entropy=float(np.log(bin_count)),
            )
            if not np.isclose(
                result,
                expected,
                rtol=0.0,
                atol=1e-9 if ACTIVE_BACKEND == "mojo" else 1e-12,
            ):
                raise ValueError(
                    "backend transfer entropy diverged from exact reference"
                )
            return result
        except Exception:
            bin_count = int(bin_count)

    return expected


def transfer_entropy_matrix(phase_series: FloatArray, n_bins: int = 16) -> FloatArray:
    """Pairwise TE matrix; entry ``[i, j] = TE(i → j)`` for all
    oscillator pairs with zero diagonal."""
    bin_count = _validate_n_bins(n_bins)
    series = _validate_phase_series(phase_series, name="phase_series")
    n_osc, n_time = series.shape
    expected = _te_matrix_reference(series, bin_count)
    backend_fn = _dispatch("te_matrix")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, int, int, int], FloatArray]", backend_fn)
        try:
            flat = fn(
                np.ascontiguousarray(series.ravel(), dtype=np.float64),
                n_osc,
                n_time,
                bin_count,
            )
            return _validate_te_matrix(
                flat,
                n_osc=n_osc,
                max_entropy=float(np.log(bin_count)),
                expected=expected,
                atol=1e-9 if ACTIVE_BACKEND == "mojo" else 1e-12,
            )
        except Exception:
            n_time = int(n_time)

    return expected
