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
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["swarmalator_step_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "swarmalator.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.SwarmalatorJL
    return _JULIA_MODULE


def swarmalator_step_julia(
    pos: NDArray,
    phases: NDArray,
    omegas: NDArray,
    n: int,
    dim: int,
    a: float,
    b: float,
    j: float,
    k: float,
    dt: float,
) -> tuple[NDArray, NDArray]:
    jl = _ensure()
    new_pos_flat, new_phases = jl.swarmalator_step(
        np.ascontiguousarray(pos.ravel(), dtype=np.float64),
        np.ascontiguousarray(phases.ravel(), dtype=np.float64),
        np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
        int(n),
        int(dim),
        float(a),
        float(b),
        float(j),
        float(k),
        float(dt),
    )
    return (
        np.asarray(new_pos_flat, dtype=np.float64).reshape(n, dim),
        np.asarray(new_phases, dtype=np.float64),
    )
