# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for UPDE engine

"""Julia backend for ``upde/engine.py``'s batched ``run()`` kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_validation import (
    validate_upde_backend_inputs,
    validate_upde_backend_output,
    validate_upde_schedule_backend_inputs,
)

__all__ = ["upde_run_julia", "upde_run_omega_schedule_julia"]
FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "upde_engine.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.UPDEEngineJL
    return _JULIA_MODULE


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
    """Run the core UPDE phase integrator.

    The calculation is delegated to the Julia backend.
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
        np.asarray(
            jl.upde_run(
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
            dtype=np.float64,
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
    """Run UPDE with one frequency vector per outer step in Julia."""

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
        np.asarray(
            jl.upde_run_omega_schedule(
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
            dtype=np.float64,
        ),
        n=n,
    )
