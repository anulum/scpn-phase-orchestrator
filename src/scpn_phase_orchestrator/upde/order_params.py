# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order parameter computation

"""Kuramoto order parameter family with 5-backend fallback chain.

Follows the AttnRes-level module standard
(``feedback_module_standard_attnres.md``):

* ``compute_order_parameter`` — R and mean phase ψ.
* ``compute_plv`` — phase-locking value between two equal-length
  phase series.
* ``compute_layer_coherence`` — R restricted to a layer.

Each kernel is available in five languages — Rust, Mojo, Julia, Go,
Python. ``AVAILABLE_BACKENDS`` reports detected backends in canonical
fallback order, while ``ACTIVE_BACKEND`` is selected by a small import-time
hot-path probe so slow external wrappers do not displace the faster local
path.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "compute_order_parameter",
    "compute_plv",
    "compute_layer_coherence",
]


# ---------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import (
        compute_layer_coherence_rust,
        order_parameter,
        plv,
    )

    return {
        "order_parameter": order_parameter,
        "plv": plv,
        "layer_coherence": compute_layer_coherence_rust,
    }


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    from ..experimental.accelerators.upde._order_params_mojo import (
        _ensure_exe,
        layer_coherence_mojo,
        order_parameter_mojo,
        plv_mojo,
    )

    _ensure_exe()
    return {
        "order_parameter": order_parameter_mojo,
        "plv": plv_mojo,
        "layer_coherence": layer_coherence_mojo,
    }


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    import juliacall  # noqa: F401

    from ..experimental.accelerators.upde._order_params_julia import (
        layer_coherence_julia,
        order_parameter_julia,
        plv_julia,
    )

    return {
        "order_parameter": order_parameter_julia,
        "plv": plv_julia,
        "layer_coherence": layer_coherence_julia,
    }


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    from ..experimental.accelerators.upde._order_params_go import (
        _load_lib,
        layer_coherence_go,
        order_parameter_go,
        plv_go,
    )

    _load_lib()
    return {
        "order_parameter": order_parameter_go,
        "plv": plv_go,
        "layer_coherence": layer_coherence_go,
    }


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


def _contains_boolean_alias(value: object) -> bool:
    try:
        values = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in values.flat)


def _unit_interval(value: float) -> float:
    """Preserve the physical bound of coherence magnitudes."""
    if not np.isfinite(value):
        raise ValueError("coherence magnitude must be finite")
    if value < -1e-12 or value > 1.0 + 1e-12:
        raise ValueError("coherence magnitude must lie in [0, 1]")
    return float(np.clip(value, 0.0, 1.0))


def _mean_phase(value: float) -> float:
    """Preserve the physical phase domain for mean phase outputs."""
    if not np.isfinite(value):
        raise ValueError("mean phase must be finite")
    return float(value % TWO_PI)


def _validate_phases(name: str, phases: FloatArray) -> FloatArray:
    raw = np.asarray(phases)
    if _contains_boolean_alias(phases):
        raise ValueError(f"{name} must not contain boolean values")
    try:
        values = raw.astype(np.float64, copy=True).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _layer_indices(layer_mask: BoolArray | IntArray, n_phases: int) -> IntArray:
    mask = np.asarray(layer_mask)
    if mask.dtype == bool:
        flattened = mask.ravel()
        if flattened.size != n_phases:
            raise ValueError("layer_mask boolean length must match phases length")
        return np.flatnonzero(flattened).astype(np.int64)
    if _contains_boolean_alias(layer_mask):
        raise ValueError("layer_mask indices must not contain boolean values")
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("layer_mask indices must be integers")
    try:
        indices = mask.astype(np.int64, copy=True).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError("layer_mask indices must be integers") from exc
    if indices.size > 0 and (np.any(indices < 0) or np.any(indices >= n_phases)):
        raise ValueError("layer_mask indices must reference existing oscillators")
    if np.unique(indices).size != indices.size:
        raise ValueError("layer_mask indices must not repeat oscillators")
    return indices


def _python_order_parameter(phases: FloatArray) -> tuple[float, float]:
    with np.errstate(invalid="ignore"):
        z = np.mean(np.exp(1j * phases))
    return _unit_interval(float(np.abs(z))), _mean_phase(float(np.angle(z)))


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
    active = min(available, key=_order_parameter_probe_seconds)
    return active, available


def _order_parameter_probe_seconds(name: str) -> float:
    phases = np.linspace(0.0, TWO_PI, 256, dtype=np.float64)
    start = perf_counter()
    try:
        if name == "python":
            _python_order_parameter(phases)
        else:
            fn = cast(
                "Callable[[FloatArray], tuple[float, float]]",
                _load_backend(name)["order_parameter"],
            )
            fn(phases)
    except (ImportError, RuntimeError, OSError, KeyError):
        return float("inf")
    return perf_counter() - start


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()
_HAS_RUST = ACTIVE_BACKEND == "rust"


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
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        if fn is None:
            continue
        return fn
    return None


# ---------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------


def compute_order_parameter(phases: FloatArray) -> tuple[float, float]:
    """Kuramoto global order parameter ``(R, ψ)``.

    ``R = |mean(exp(i · θ))|``;
    ``ψ = arg(mean(exp(i · θ))) mod 2π``.
    """
    phases = _validate_phases("phases", phases)
    if phases.size == 0:
        return (0.0, 0.0)
    backend_fn = _dispatch("order_parameter")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray], tuple[float, float]]", backend_fn)
        p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
        r, psi = fn(p)
        return _unit_interval(float(r)), _mean_phase(float(psi))

    return _python_order_parameter(phases)


def compute_plv(phases_a: FloatArray, phases_b: FloatArray) -> float:
    """Phase-locking value between two equal-length phase series.

    PLV = ``|mean(exp(i · (φ_a − φ_b)))|`` over samples.
    """
    phases_a = _validate_phases("phases_a", phases_a)
    phases_b = _validate_phases("phases_b", phases_b)
    if phases_a.size != phases_b.size:
        raise ValueError(
            f"PLV requires equal-length arrays, got {phases_a.size} vs {phases_b.size}"
        )
    if phases_a.size == 0:
        return 0.0
    backend_fn = _dispatch("plv")
    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, FloatArray], float]",
            backend_fn,
        )
        a = np.ascontiguousarray(phases_a.ravel(), dtype=np.float64)
        b = np.ascontiguousarray(phases_b.ravel(), dtype=np.float64)
        return _unit_interval(float(fn(a, b)))

    return _unit_interval(float(np.abs(np.mean(np.exp(1j * (phases_a - phases_b))))))


def compute_layer_coherence(
    phases: FloatArray, layer_mask: BoolArray | IntArray
) -> float:
    """Order parameter R for the subset of oscillators selected by
    ``layer_mask`` (boolean mask *or* integer index array)."""
    phases = _validate_phases("phases", phases)
    if phases.size == 0:
        return 0.0
    indices = _layer_indices(layer_mask, phases.size)
    if indices.size == 0:
        return 0.0
    backend_fn = _dispatch("layer_coherence")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, IntArray], float]", backend_fn)
        p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
        return _unit_interval(float(fn(p, indices)))

    sub = phases[indices]
    if sub.size == 0:
        return 0.0
    z = np.mean(np.exp(1j * sub))
    return _unit_interval(float(np.abs(z)))
