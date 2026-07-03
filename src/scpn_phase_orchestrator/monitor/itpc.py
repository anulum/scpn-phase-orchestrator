# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Inter-Trial Phase Coherence

"""Lachaux 1999 inter-trial phase coherence with a 5-backend fallback chain.

Two kernels:

* :func:`compute_itpc` — ITPC across trials at each time point.
* :func:`itpc_persistence` — mean ITPC at stimulus-pause indices.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor._julia_runtime import require_juliacall_main

FloatArray: TypeAlias = NDArray[np.float64]
ArrayPayload: TypeAlias = NDArray[np.generic]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "compute_itpc",
    "itpc_persistence",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    """Load the Rust ITPC backend callables."""
    from spo_kernel import (
        compute_itpc_rust,
        itpc_persistence_rust,
    )

    return {
        "itpc": compute_itpc_rust,
        "persistence": itpc_persistence_rust,
    }


def _load_mojo_fns() -> dict[str, object]:
    """Load the Mojo ITPC backend callables."""
    from ..experimental.accelerators.monitor._itpc_mojo import (
        _ensure_exe,
        compute_itpc_mojo,
        itpc_persistence_mojo,
    )

    _ensure_exe()
    return {"itpc": compute_itpc_mojo, "persistence": itpc_persistence_mojo}


def _load_julia_fns() -> dict[str, object]:
    """Load the Julia ITPC backend callables."""
    require_juliacall_main()
    from ..experimental.accelerators.monitor._itpc_julia import (
        compute_itpc_julia,
        itpc_persistence_julia,
    )

    return {"itpc": compute_itpc_julia, "persistence": itpc_persistence_julia}


def _load_go_fns() -> dict[str, object]:
    """Load the Go ITPC backend callables."""
    from ..experimental.accelerators.monitor._itpc_go import (
        _load_lib,
        compute_itpc_go,
        itpc_persistence_go,
    )

    _load_lib()
    return {"itpc": compute_itpc_go, "persistence": itpc_persistence_go}


_LOADERS: dict[str, Callable[[], dict[str, object]]] = {
    "rust": _load_rust_fns,
    "mojo": _load_mojo_fns,
    "julia": _load_julia_fns,
    "go": _load_go_fns,
}
_BACKEND_FN_CACHE: dict[str, dict[str, object]] = {}


def _load_backend(name: str) -> dict[str, object]:
    """Load and cache the named backend callables."""
    cached = _BACKEND_FN_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_FN_CACHE[name] = loaded
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


def _dispatch(fn_name: str) -> object | None:
    """Return the fastest available backend callables, or ``None`` for Python."""
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    deduped: list[str] = []
    for backend in ordered_backends:
        if backend in deduped:
            continue
        deduped.append(backend)

    for backend in deduped:
        if backend == "python":
            return None
        loader = _LOADERS.get(backend)
        if loader is None:
            continue
        try:
            backend_cache = _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        fn = backend_cache.get(fn_name)
        if fn is None:
            continue
        return fn
    return None


def _contains_boolean_alias(raw: ArrayPayload) -> bool:
    """Return whether the array contains any boolean alias."""
    if raw.dtype == np.bool_:
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in raw.flat)


def _contains_complex_alias(raw: ArrayPayload) -> bool:
    """Return whether the array contains any complex-number alias."""
    if np.iscomplexobj(raw):
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, complex | np.complexfloating) for value in raw.flat)


def _is_string_like(value: object) -> bool:
    """Return whether ``value`` is a Python or NumPy string scalar."""
    return isinstance(value, (str, bytes, np.str_, np.bytes_))


def _is_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` is a string scalar parsable as a float."""
    if not _is_string_like(value):
        return False
    try:
        float(cast("str | bytes", value))
    except (TypeError, ValueError):
        return False
    return True


def _contains_numeric_string_alias(raw: ArrayPayload) -> bool:
    """Return whether the array contains numeric string aliases."""
    if raw.dtype.kind not in {"O", "S", "U"}:
        return False
    saw_string = False
    for value in raw.astype(object, copy=False).flat:
        if not _is_string_like(value):
            continue
        saw_string = True
        if not _is_numeric_string_alias(value):
            return False
    return saw_string


def _validate_phases_trials(phases_trials: object) -> FloatArray:
    """Return the per-trial phase array as a validated 2-D finite array, else raise."""
    raw = np.asarray(phases_trials)
    if _contains_boolean_alias(raw):
        raise ValueError("phases_trials must not contain boolean values")
    if _contains_complex_alias(raw):
        raise ValueError("phases_trials must contain real-valued phase samples")
    if _contains_numeric_string_alias(raw):
        raise ValueError("phases_trials must not contain numeric-string aliases")
    try:
        phases = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases_trials must be a finite 1D or 2D float array") from exc
    if phases.ndim not in {1, 2}:
        raise ValueError(f"phases_trials must be 1D or 2D, got shape {phases.shape}")
    if not np.all(np.isfinite(phases)):
        raise ValueError("phases_trials must contain only finite values")
    return np.ascontiguousarray(phases, dtype=np.float64)


def _validate_pause_indices(pause_indices: object) -> IntArray:
    """Return the validated pause/window indices, else raise ``ValueError``."""
    raw = np.asarray(pause_indices, dtype=object)
    if raw.ndim != 1:
        raise ValueError("pause_indices must be a one-dimensional integer array")
    if raw.size == 0:
        return np.zeros(0, dtype=np.int64)
    flat = raw.ravel()
    if not all(
        isinstance(value, Integral) and not isinstance(value, bool) for value in flat
    ):
        raise ValueError("pause_indices must contain only integer indices")
    return np.ascontiguousarray(flat, dtype=np.int64)


def _compute_itpc_reference(phases: FloatArray) -> FloatArray:
    """Return the reference inter-trial phase coherence series (NumPy floor)."""
    if phases.ndim == 1:
        return np.array([1.0], dtype=np.float64)
    if phases.shape[0] == 0:
        return np.zeros(phases.shape[1], dtype=np.float64)
    return np.ascontiguousarray(
        np.abs(np.mean(np.exp(1j * phases), axis=0)),
        dtype=np.float64,
    )


def _validate_itpc_values(
    value: object,
    *,
    n_timepoints: int,
    expected: FloatArray | None = None,
    atol: float = 1e-12,
) -> FloatArray:
    """Return backend ITPC values matching the reference in [0, 1], else raise."""
    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError("ITPC output must not contain boolean values")
    if _contains_complex_alias(raw):
        raise ValueError("ITPC output must contain real values")
    if _contains_numeric_string_alias(raw):
        raise ValueError("ITPC output must not contain numeric-string aliases")
    try:
        itpc = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("ITPC output must be numeric") from exc
    if itpc.shape != (n_timepoints,):
        raise ValueError(
            f"ITPC output shape {itpc.shape} does not match ({n_timepoints},)"
        )
    if not np.all(np.isfinite(itpc)):
        raise ValueError("ITPC output must contain only finite values")
    tolerance = 1e-12
    if np.any(itpc < -tolerance) or np.any(itpc > 1.0 + tolerance):
        raise ValueError("ITPC output must lie in [0, 1]")
    clipped = np.ascontiguousarray(np.clip(itpc, 0.0, 1.0), dtype=np.float64)
    if expected is not None:
        reference = np.asarray(expected, dtype=np.float64)
        if reference.shape != clipped.shape:
            raise ValueError("ITPC exact reference shape must match backend output")
        if not np.allclose(clipped, reference, rtol=0.0, atol=atol):
            raise ValueError("ITPC backend output diverged from exact reference")
    return clipped


def _validate_persistence_value(
    value: object,
    *,
    expected: float | None = None,
    atol: float = 1e-12,
) -> float:
    """Return the ITPC persistence value as a float in [0, 1], else raise."""
    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError("ITPC persistence output must not contain boolean values")
    if _contains_complex_alias(raw):
        raise ValueError("ITPC persistence output must contain real values")
    if _contains_numeric_string_alias(raw):
        raise ValueError(
            "ITPC persistence output must not contain numeric-string aliases"
        )
    try:
        scalar = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("ITPC persistence output must be numeric") from exc
    if scalar.shape != ():
        raise ValueError("ITPC persistence output must be scalar")
    score = float(scalar)
    if not np.isfinite(score):
        raise ValueError("ITPC persistence output must be finite")
    tolerance = 1e-12
    if score < -tolerance or score > 1.0 + tolerance:
        raise ValueError("ITPC persistence output must lie in [0, 1]")
    clipped = min(1.0, max(0.0, score))
    if expected is not None and not np.isclose(
        clipped,
        float(expected),
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError(
            "ITPC persistence backend output diverged from exact reference"
        )
    return clipped


def compute_itpc(phases_trials: object) -> FloatArray:
    """Inter-Trial Phase Coherence at each time point.

    ``ITPC = |mean(exp(i·θ))|`` across trials (Lachaux et al. 1999).

    Parameters
    ----------
    phases_trials : object
        shape ``(n_trials, n_timepoints)`` — phases in radians. A 1-D input is treated
        as a single trial.

    Returns
    -------
    FloatArray
        ``(n_timepoints,)`` array of ITPC values in ``[0, 1]``.
    """
    phases = _validate_phases_trials(phases_trials)
    if phases.ndim == 1:
        return np.array([1.0], dtype=np.float64)
    if phases.shape[0] == 0:
        return np.array([], dtype=np.float64)
    n_trials, n_tp = phases.shape
    expected = _compute_itpc_reference(phases)

    backend_fn = _dispatch("itpc")
    if backend_fn is not None:
        try:
            if ACTIVE_BACKEND == "rust":
                fn_rust = cast(
                    "Callable[[FloatArray, int, int], FloatArray]",
                    backend_fn,
                )
                flat = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
                return _validate_itpc_values(
                    fn_rust(flat, n_trials, n_tp),
                    n_timepoints=n_tp,
                    expected=expected,
                )
            fn = cast("Callable[[FloatArray, int, int], FloatArray]", backend_fn)
            return _validate_itpc_values(
                fn(phases.ravel(), int(n_trials), int(n_tp)),
                n_timepoints=n_tp,
                expected=expected,
                atol=1e-9 if ACTIVE_BACKEND == "mojo" else 1e-12,
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    return _validate_itpc_values(expected, n_timepoints=n_tp, expected=expected)


def itpc_persistence(
    phases_trials: object,
    pause_indices: object,
) -> float:
    """Mean ITPC at stimulus-pause indices.

    Distinguishes true neural entrainment from evoked response: if ITPC
    remains high after the driving stimulus stops, oscillators have
    genuinely phase-locked. If it drops immediately, the response was
    merely evoked.

    Parameters
    ----------
    phases_trials : object
        ``(n_trials, n_timepoints)`` phases in radians.
    pause_indices : object
        time-point indices falling within / after a pause.

    Returns
    -------
    float
        Mean ITPC across ``pause_indices``. ``0.0`` if empty.
    """
    phases = _validate_phases_trials(phases_trials)
    pause_idx = _validate_pause_indices(pause_indices)
    if pause_idx.size == 0:
        return 0.0

    if phases.ndim == 1:
        phases = phases.reshape(1, -1)
    n_trials, n_tp = phases.shape
    itpc_full = _compute_itpc_reference(phases)
    valid = pause_idx[(pause_idx >= 0) & (pause_idx < itpc_full.size)]
    expected = 0.0 if valid.size == 0 else float(np.mean(itpc_full[valid]))

    backend_fn = _dispatch("persistence")
    if backend_fn is not None:
        try:
            if ACTIVE_BACKEND == "rust":
                fn_rust = cast(
                    "Callable[[FloatArray, int, int, IntArray], float]",
                    backend_fn,
                )
                return _validate_persistence_value(
                    fn_rust(
                        np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                        n_trials,
                        n_tp,
                        np.ascontiguousarray(pause_idx, dtype=np.int64),
                    ),
                    expected=expected,
                )
            fn = cast("Callable[[FloatArray, int, int, IntArray], float]", backend_fn)
            return _validate_persistence_value(
                fn(phases.ravel(), int(n_trials), int(n_tp), pause_idx),
                expected=expected,
                atol=1e-9 if ACTIVE_BACKEND == "mojo" else 1e-12,
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    return _validate_persistence_value(expected, expected=expected)
