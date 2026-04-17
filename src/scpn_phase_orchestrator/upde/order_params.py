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
Python — resolved fastest-first at import time through
``ACTIVE_BACKEND`` / ``AVAILABLE_BACKENDS``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

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
    from scpn_phase_orchestrator.upde._order_params_mojo import (
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
    import juliacall  # type: ignore[import-untyped]  # noqa: F401

    from scpn_phase_orchestrator.upde._order_params_julia import (
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
    from scpn_phase_orchestrator.upde._order_params_go import (
        layer_coherence_go,
        order_parameter_go,
        plv_go,
    )

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


def _dispatch(fn_name: str) -> object:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()[fn_name]


# ---------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------


def compute_order_parameter(phases: NDArray) -> tuple[float, float]:
    """Kuramoto global order parameter ``(R, ψ)``.

    ``R = |mean(exp(i · θ))|``;
    ``ψ = arg(mean(exp(i · θ))) mod 2π``.
    """
    if phases.size == 0:
        return (0.0, 0.0)
    backend_fn = _dispatch("order_parameter")
    if backend_fn is not None:
        fn = cast(
            "Callable[[NDArray], tuple[float, float]]", backend_fn
        )
        p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
        r, psi = fn(p)
        return float(r), float(psi)

    with np.errstate(invalid="ignore"):
        z = np.mean(np.exp(1j * phases))
    return float(np.abs(z)), float(np.angle(z) % TWO_PI)


def compute_plv(phases_a: NDArray, phases_b: NDArray) -> float:
    """Phase-locking value between two equal-length phase series.

    PLV = ``|mean(exp(i · (φ_a − φ_b)))|`` over samples.
    """
    if phases_a.size == 0 or phases_b.size == 0:
        return 0.0
    if phases_a.size != phases_b.size:
        raise ValueError(
            f"PLV requires equal-length arrays, got "
            f"{phases_a.size} vs {phases_b.size}"
        )
    backend_fn = _dispatch("plv")
    if backend_fn is not None:
        fn = cast(
            "Callable[[NDArray, NDArray], float]", backend_fn
        )
        a = np.ascontiguousarray(phases_a.ravel(), dtype=np.float64)
        b = np.ascontiguousarray(phases_b.ravel(), dtype=np.float64)
        return float(fn(a, b))

    return float(
        np.abs(np.mean(np.exp(1j * (phases_a - phases_b))))
    )


def compute_layer_coherence(
    phases: NDArray, layer_mask: NDArray
) -> float:
    """Order parameter R for the subset of oscillators selected by
    ``layer_mask`` (boolean mask *or* integer index array)."""
    mask = np.asarray(layer_mask)
    if mask.dtype == bool:
        indices = np.flatnonzero(mask).astype(np.int64)
    else:
        indices = mask.astype(np.int64)
    if indices.size == 0:
        return 0.0
    backend_fn = _dispatch("layer_coherence")
    if backend_fn is not None:
        fn = cast(
            "Callable[[NDArray, NDArray], float]", backend_fn
        )
        p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
        return float(fn(p, indices))

    sub = phases[indices]
    if sub.size == 0:
        return 0.0
    z = np.mean(np.exp(1j * sub))
    return float(np.abs(z))
