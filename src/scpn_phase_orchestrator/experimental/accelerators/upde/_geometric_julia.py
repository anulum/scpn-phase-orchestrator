# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for torus geometric integrator

"""Julia backend for ``upde/geometric.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._geometric_validation import (
    TWO_PI,
    validate_torus_inputs,
    validate_torus_output,
)

__all__ = ["torus_run_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "geometric.jl"
_JULIA_MODULE: Any | None = None
FloatArray: TypeAlias = NDArray[np.float64]


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.GeometricJL
    return _JULIA_MODULE


def torus_run_julia(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate torus phase dynamics.

    The calculation is delegated to the Julia backend.
    """
    (
        p,
        o,
        k,
        a,
        n_i,
        zeta_f,
        psi_f,
        dt_f,
        n_steps_i,
    ) = validate_torus_inputs(
        phases,
        omegas,
        knm_flat,
        alpha_flat,
        n,
        zeta,
        psi,
        dt,
        n_steps,
    )
    if n_steps_i == 0:
        return p % TWO_PI
    jl = _ensure()
    result = jl.torus_run(
        p,
        o,
        k,
        a,
        n_i,
        zeta_f,
        psi_f,
        dt_f,
        n_steps_i,
    )
    return validate_torus_output(np.asarray(result, dtype=np.float64), n=n_i)
