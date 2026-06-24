# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for inertial stepper

"""Julia backend for ``upde/inertial.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._inertial_validation import (
    validate_inertial_inputs,
    validate_inertial_output,
)

__all__ = ["inertial_step_julia"]

FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "inertial.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    """Build or load the backend artifact if it is missing, else raise."""
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.InertialJL
    return _JULIA_MODULE


def inertial_step_julia(
    theta: FloatArray,
    omega_dot: FloatArray,
    power: FloatArray,
    knm_flat: FloatArray,
    inertia: FloatArray,
    damping: FloatArray,
    n: int,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Advance one inertial Kuramoto step.

    The calculation is delegated to the Julia backend.
    """
    th, od, pw, km, ine, dmp, n_i, dt_f = validate_inertial_inputs(
        theta,
        omega_dot,
        power,
        knm_flat,
        inertia,
        damping,
        n,
        dt,
    )
    jl = _ensure()
    new_theta, new_omega_dot = jl.inertial_step(
        th,
        od,
        pw,
        km,
        ine,
        dmp,
        n_i,
        dt_f,
    )
    return validate_inertial_output(new_theta, new_omega_dot, n=n_i)
