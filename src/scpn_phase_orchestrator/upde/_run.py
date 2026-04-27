# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE batched-run dispatcher

"""Five-backend dispatcher for the UPDE batched integrator.

Exports:

* :data:`ACTIVE_BACKEND` / :data:`AVAILABLE_BACKENDS` — resolved at
  import time with the fastest-first preference
  (Rust → Mojo → Julia → Go → Python).
* :func:`upde_run` — stateless entry point that any caller can use
  directly; :class:`scpn_phase_orchestrator.upde.engine.UPDEEngine`
  also routes through here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._ref_kernel import upde_run_python

__all__ = ["ACTIVE_BACKEND", "AVAILABLE_BACKENDS", "upde_run"]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., NDArray]:
    from spo_kernel import PyUPDEStepper  # noqa: F401

    def _rust_run(
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        alpha: NDArray,
        zeta: float,
        psi: float,
        dt: float,
        n_steps: int,
        method: str,
        n_substeps: int,
        atol: float,
        rtol: float,
    ) -> NDArray:
        n = int(phases.size)
        stepper = PyUPDEStepper(
            n, dt, method, n_substeps=n_substeps, atol=atol, rtol=rtol
        )
        return np.asarray(
            stepper.run(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                float(zeta),
                float(psi),
                np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
                int(n_steps),
            ),
            dtype=np.float64,
        )

    return cast("Callable[..., NDArray]", _rust_run)


def _load_mojo_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._engine_mojo import (
        _ensure_exe,
        upde_run_mojo,
    )

    _ensure_exe()
    return upde_run_mojo


def _load_julia_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    import juliacall  # type: ignore[import-untyped]  # noqa: F401

    from scpn_phase_orchestrator.upde._engine_julia import upde_run_julia

    return upde_run_julia


def _load_go_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._engine_go import upde_run_go

    return upde_run_go


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


def upde_run(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
    method: str = "euler",
    n_substeps: int = 1,
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> NDArray:
    """Stateless batched UPDE integrator.

    Dispatches to the first available backend per the SPO chain
    (Rust → Mojo → Julia → Go → Python). Every backend runs the
    same algorithm: ``method ∈ {"euler", "rk4", "rk45"}`` with
    ``n_substeps`` applied to the fixed-step methods; RK45 is
    adaptive with ``(atol, rtol)`` tolerances. Phases are wrapped
    to ``[0, 2π)`` after each outer step.
    """
    p = np.ascontiguousarray(phases, dtype=np.float64)
    o = np.ascontiguousarray(omegas, dtype=np.float64)
    k = np.ascontiguousarray(knm, dtype=np.float64)
    a = np.ascontiguousarray(alpha, dtype=np.float64)
    backend_fn = _dispatch()
    if backend_fn is None:
        return upde_run_python(
            p,
            o,
            k,
            a,
            float(zeta),
            float(psi),
            float(dt),
            int(n_steps),
            method,
            int(n_substeps),
            float(atol),
            float(rtol),
        )
    return np.asarray(
        backend_fn(
            p,
            o,
            k,
            a,
            float(zeta),
            float(psi),
            float(dt),
            int(n_steps),
            method,
            int(n_substeps),
            float(atol),
            float(rtol),
        ),
        dtype=np.float64,
    )
