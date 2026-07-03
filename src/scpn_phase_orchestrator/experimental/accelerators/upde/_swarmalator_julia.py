# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for swarmalator stepper

"""Julia backend for ``upde/swarmalator.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)
from scpn_phase_orchestrator.upde._swarmalator_validation import (
    validate_swarmalator_inputs,
    validate_swarmalator_output,
)

__all__ = ["swarmalator_step_julia"]

FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "swarmalator.jl"
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
    _JULIA_MODULE = JuliaMain.SwarmalatorJL
    return _JULIA_MODULE


def swarmalator_step_julia(
    pos: FloatArray,
    phases: FloatArray,
    omegas: FloatArray,
    n: int,
    dim: int,
    a: float,
    b: float,
    j: float,
    k: float,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Advance one swarmalator position-phase step.

    The calculation is delegated to the Julia backend.
    """
    p, ph, om, n_i, dim_i, a_f, b_f, j_f, k_f, dt_f = validate_swarmalator_inputs(
        pos,
        phases,
        omegas,
        n,
        dim,
        a,
        b,
        j,
        k,
        dt,
    )
    jl = _ensure()
    new_pos_flat, new_phases = jl.swarmalator_step(
        np.ascontiguousarray(p.ravel(), dtype=np.float64),
        ph,
        om,
        n_i,
        dim_i,
        a_f,
        b_f,
        j_f,
        k_f,
        dt_f,
    )
    new_pos_array, new_phases_array = validate_swarmalator_output(
        new_pos_flat,
        new_phases,
        n=n_i,
        dim=dim_i,
    )
    return new_pos_array, new_phases_array
