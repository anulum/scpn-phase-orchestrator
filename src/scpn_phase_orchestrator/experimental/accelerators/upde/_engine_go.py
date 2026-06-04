# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for UPDE engine

"""Go backend for ``upde/engine.py`` via ``libupde_engine.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_validation import (
    validate_upde_backend_inputs,
    validate_upde_backend_output,
    validate_upde_schedule_backend_inputs,
)

__all__ = ["upde_run_go", "upde_run_omega_schedule_go"]
FloatArray: TypeAlias = NDArray[np.float64]

_METHOD_IDS = {"euler": 0, "rk4": 1, "rk45": 2}

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libupde_engine.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libupde_engine.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libupde_engine.so "
            f"upde_engine.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.UPDERun.restype = ctypes.c_int
    lib.UPDERun.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # phases (in/out)
        ctypes.POINTER(ctypes.c_double),  # omegas
        ctypes.POINTER(ctypes.c_double),  # knm
        ctypes.POINTER(ctypes.c_double),  # alpha
        ctypes.c_int,  # n
        ctypes.c_double,  # zeta
        ctypes.c_double,  # psi
        ctypes.c_double,  # dt
        ctypes.c_int,  # n_steps
        ctypes.c_int,  # method (0/1/2)
        ctypes.c_int,  # n_substeps
        ctypes.c_double,  # atol
        ctypes.c_double,  # rtol
    ]
    if hasattr(lib, "UPDERunOmegaSchedule"):
        lib.UPDERunOmegaSchedule.restype = ctypes.c_int
        lib.UPDERunOmegaSchedule.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # phases (in/out)
            ctypes.POINTER(ctypes.c_double),  # omega schedule
            ctypes.POINTER(ctypes.c_double),  # knm
            ctypes.POINTER(ctypes.c_double),  # alpha
            ctypes.c_int,  # n
            ctypes.c_double,  # zeta
            ctypes.c_double,  # psi
            ctypes.c_double,  # dt
            ctypes.c_int,  # n_steps
            ctypes.c_int,  # method (0/1/2)
            ctypes.c_int,  # n_substeps
            ctypes.c_double,  # atol
            ctypes.c_double,  # rtol
        ]
    _LIB = lib
    return lib


def upde_run_go(
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

    The calculation is delegated to the Go backend.
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
    lib = _load_lib()
    rc = lib.UPDERun(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_double(zeta_f),
        ctypes.c_double(psi_f),
        ctypes.c_double(dt_f),
        ctypes.c_int(n_steps_i),
        ctypes.c_int(_METHOD_IDS[method_s]),
        ctypes.c_int(n_substeps_i),
        ctypes.c_double(atol_f),
        ctypes.c_double(rtol_f),
    )
    if rc != 0:
        raise ValueError(f"Go UPDERun rc={rc}")
    return validate_upde_backend_output(p, n=n)


def upde_run_omega_schedule_go(
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
    """Run UPDE with one frequency vector per outer step in the Go backend."""

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
    lib = _load_lib()
    if not hasattr(lib, "UPDERunOmegaSchedule"):
        raise ImportError("Go UPDERunOmegaSchedule symbol is not available")
    rc = lib.UPDERunOmegaSchedule(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        schedule.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_double(zeta_f),
        ctypes.c_double(psi_f),
        ctypes.c_double(dt_f),
        ctypes.c_int(n_steps_i),
        ctypes.c_int(_METHOD_IDS[method_s]),
        ctypes.c_int(n_substeps_i),
        ctypes.c_double(atol_f),
        ctypes.c_double(rtol_f),
    )
    if rc != 0:
        raise ValueError(f"Go UPDERunOmegaSchedule rc={rc}")
    return validate_upde_backend_output(p, n=n)
