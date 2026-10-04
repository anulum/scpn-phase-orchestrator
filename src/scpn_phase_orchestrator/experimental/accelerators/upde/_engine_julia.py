# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for UPDE engine

"""Julia backend for ``upde/engine.py``'s batched ``run()`` kernel."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)
from scpn_phase_orchestrator.upde._engine_validation import (
    validate_upde_backend_inputs,
    validate_upde_backend_output,
    validate_upde_schedule_backend_inputs,
)

__all__ = ["upde_run_julia", "upde_run_omega_schedule_julia"]
FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "upde_engine.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    """Build or load the backend artifact if it is missing, else raise."""
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    JuliaMain = require_julia_main()

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.UPDEEngineJL
    return _JULIA_MODULE


def _julia_error_type() -> type[BaseException] | None:
    """Return ``juliacall.JuliaError``, or ``None`` when juliacall is absent.

    The type is resolved only after a call has failed. A process without
    juliacall cannot have raised a Julia error, so its exception passes through
    unchanged and the adapter's output validation stays usable there.
    """
    try:
        from juliacall import JuliaError
    except ModuleNotFoundError:
        return None
    return cast("type[BaseException]", JuliaError)


def _invoke(func: Callable[..., object], *args: object) -> object:
    """Invoke the real Julia solver with numeric-domain error translation.

    Parameters
    ----------
    func : Callable[..., object]
        Actual Julia UPDE entry point from the loaded native module.
    *args : object
        Already validated solver arguments.

    Returns
    -------
    object
        Actual Julia result for the owning output validator.

    Raises
    ------
    ValueError
        Native Julia ``DomainError`` indicates divergent numerical computation.
    juliacall.JuliaError
        Other native Julia failures retain their original exception.
    """
    try:
        return func(*args)
    except Exception as exc:
        julia_error = _julia_error_type()
        if julia_error is None or not isinstance(exc, julia_error):
            raise
        main = require_julia_main()
        if bool(main.isa(cast("Any", exc).exception, main.DomainError)):
            raise ValueError(
                "Julia UPDE computation diverged with a domain error"
            ) from exc
        raise


def upde_run_julia(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> FloatArray:
    """Run the admitted UPDE integration through the actual Julia backend.

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.
    omegas : FloatArray
        Natural frequencies in rad/s, shape ``(N,)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : FloatArray
        Finite phase-lag matrix in radians, shape ``(N, N)``; use zeros for no lag.
    zeta : float
        External drive strength ``ζ``.
    psi : float
        External drive reference phase ``Ψ`` in radians.
    dt : float
        Integration step size.
    n_steps : int
        Number of integration steps to run.
    method : str
        Integration method (``euler``, ``rk4``, or ``rk45``).
    n_substeps : int
        Number of inner substeps per outer step.
    atol : float
        Absolute tolerance for the adaptive (rk45) integrator.
    rtol : float
        Relative tolerance for the adaptive (rk45) integrator.

    Returns
    -------
    FloatArray
        The final phases after ``n_steps`` integration steps.

    Notes
    -----
    Native Julia DomainError from numerical divergence is translated to
    ValueError. Other Julia exceptions retain their native identity.
    """
    (
        p,
        o,
        k,
        a,
        zeta_f,
        psi_f,
        dt_f,
        n_steps_i,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validate_upde_backend_inputs(
        phases,
        omegas,
        knm,
        alpha,
        zeta,
        psi,
        dt,
        n_steps,
        method,
        n_substeps,
        atol,
        rtol,
    )
    n = int(p.size)
    if n_steps_i == 0:
        return p.copy()
    jl = _ensure()
    return validate_upde_backend_output(
        _invoke(
            jl.upde_run,
            p,
            o,
            k,
            a,
            n,
            zeta_f,
            psi_f,
            dt_f,
            n_steps_i,
            method_s,
            n_substeps_i,
            atol_f,
            rtol_f,
        ),
        n=n,
    )


def upde_run_omega_schedule_julia(
    phases: FloatArray,
    omega_schedule: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> FloatArray:
    """Run the admitted UPDE integration through the actual Julia backend.

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.
    omega_schedule : FloatArray
        Per-step natural-frequency vectors, shape ``(n_steps, N)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : FloatArray
        Finite phase-lag matrix in radians, shape ``(N, N)``; use zeros for no lag.
    zeta : float
        External drive strength ``ζ``.
    psi : float
        External drive reference phase ``Ψ`` in radians.
    dt : float
        Integration step size.
    method : str
        Integration method (``euler``, ``rk4``, or ``rk45``).
    n_substeps : int
        Number of inner substeps per outer step.
    atol : float
        Absolute tolerance for the adaptive (rk45) integrator.
    rtol : float
        Relative tolerance for the adaptive (rk45) integrator.

    Returns
    -------
    FloatArray
        The final phases after integrating the omega schedule.

    Notes
    -----
    Native Julia DomainError from numerical divergence is translated to
    ValueError. Other Julia exceptions retain their native identity.
    """
    (
        p,
        schedule,
        k,
        a,
        zeta_f,
        psi_f,
        dt_f,
        n_steps_i,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validate_upde_schedule_backend_inputs(
        phases,
        omega_schedule,
        knm,
        alpha,
        zeta,
        psi,
        dt,
        method,
        n_substeps,
        atol,
        rtol,
    )
    n = int(p.size)
    jl = _ensure()
    return validate_upde_backend_output(
        _invoke(
            jl.upde_run_omega_schedule,
            p,
            schedule,
            k,
            a,
            n,
            zeta_f,
            psi_f,
            dt_f,
            n_steps_i,
            method_s,
            n_substeps_i,
            atol_f,
            rtol_f,
        ),
        n=n,
    )
