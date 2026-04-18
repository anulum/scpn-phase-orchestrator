# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Thermodynamic entropy production rate

"""Overdamped-Kuramoto thermodynamic dissipation rate with a 5-backend
fallback chain per ``feedback_module_standard_attnres.md``.

    Σ = Σ_i (dθ_i/dt)² · dt
    dθ_i/dt = ω_i + (α / N) · Σ_j K_ij · sin(θ_j − θ_i)

Zero at frequency-locked fixed points; positive otherwise.
Reference: Acebrón et al. 2005, Rev. Mod. Phys. 77:137–185.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "entropy_production_rate",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., float]:
    from spo_kernel import entropy_production_rate as _rust_ep

    def _rust(
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        alpha: float,
        dt: float,
    ) -> float:
        return float(
            _rust_ep(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                float(alpha),
                float(dt),
            )
        )

    return cast("Callable[..., float]", _rust)


def _load_mojo_fn() -> Callable[..., float]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._entropy_prod_mojo import (
        _ensure_exe,
        entropy_production_rate_mojo,
    )

    _ensure_exe()
    return entropy_production_rate_mojo


def _load_julia_fn() -> Callable[..., float]:  # pragma: no cover — toolchain
    import juliacall  # type: ignore[import-untyped]  # noqa: F401

    from scpn_phase_orchestrator.monitor._entropy_prod_julia import (
        entropy_production_rate_julia,
    )

    return entropy_production_rate_julia


def _load_go_fn() -> Callable[..., float]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._entropy_prod_go import (
        entropy_production_rate_go,
    )

    return entropy_production_rate_go


_LOADERS: dict[str, Callable[[], Callable[..., float]]] = {
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


def _dispatch() -> Callable[..., float] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


def entropy_production_rate(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: float,
    dt: float,
) -> float:
    """Thermodynamic dissipation rate ``Σ (dθ/dt)² · dt``.

    ``dθ_i/dt = ω_i + (α / N) Σ_j K_ij sin(θ_j − θ_i)``. Zero at
    frequency-locked fixed points; positive otherwise.

    Acebrón et al. 2005, Rev. Mod. Phys. **77**:137–185.

    Args:
        phases: ``(N,)`` instantaneous phases in radians.
        omegas: ``(N,)`` natural frequencies.
        knm: ``(N, N)`` coupling matrix.
        alpha: global coupling strength.
        dt: integration timestep for the ``· dt`` factor.

    Returns:
        Non-negative dissipation scalar.
    """
    n = int(phases.size)
    if n == 0 or dt <= 0.0:
        return 0.0
    backend_fn = _dispatch()
    if backend_fn is not None:
        return float(
            backend_fn(phases, omegas, knm, float(alpha), float(dt))
        )

    diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    coupling = np.sum(knm * np.sin(diff), axis=1)
    dtheta_dt = omegas + (alpha / n) * coupling
    return float(np.sum(dtheta_dt**2) * dt)
