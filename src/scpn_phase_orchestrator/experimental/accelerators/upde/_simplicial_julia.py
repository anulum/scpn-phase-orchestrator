# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for simplicial Kuramoto

"""Julia backend for ``upde/simplicial.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

from ._simplicial_validation import (
    validate_simplicial_inputs,
    validate_simplicial_output,
)

__all__ = ["simplicial_run_julia"]

FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "simplicial.jl"
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
    _JULIA_MODULE = JuliaMain.SimplicialJL
    return _JULIA_MODULE


def simplicial_run_julia(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    sigma2: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate pairwise-plus-simplicial Kuramoto dynamics.

    The calculation is delegated to the Julia backend.
    """
    phases, omegas, knm_flat, alpha_flat, n, zeta, psi, sigma2, dt, n_steps = (
        validate_simplicial_inputs(
            phases,
            omegas,
            knm_flat,
            alpha_flat,
            n,
            zeta,
            psi,
            sigma2,
            dt,
            n_steps,
        )
    )
    if n_steps == 0:
        return phases.copy()
    jl = _ensure()
    result = jl.simplicial_run(
        phases,
        omegas,
        knm_flat,
        alpha_flat,
        int(n),
        float(zeta),
        float(psi),
        float(sigma2),
        float(dt),
        int(n_steps),
    )
    result_array: FloatArray = np.asarray(result, dtype=np.float64)
    return validate_simplicial_output(result_array, n=n)
