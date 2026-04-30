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

__all__ = ["simplicial_run_julia"]

FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "simplicial.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

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
    jl = _ensure()
    result = jl.simplicial_run(
        np.ascontiguousarray(phases, dtype=np.float64),
        np.ascontiguousarray(omegas, dtype=np.float64),
        np.ascontiguousarray(knm_flat, dtype=np.float64),
        np.ascontiguousarray(alpha_flat, dtype=np.float64),
        int(n),
        float(zeta),
        float(psi),
        float(sigma2),
        float(dt),
        int(n_steps),
    )
    result_array: FloatArray = np.asarray(result, dtype=np.float64)
    return result_array
