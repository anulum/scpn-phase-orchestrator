# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for NPE

"""Julia backend for ``monitor/npe.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["phase_distance_matrix_julia", "compute_npe_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "npe.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.NPE
    return _JULIA_MODULE


def phase_distance_matrix_julia(phases: NDArray) -> NDArray:
    jl = _ensure()
    return np.asarray(
        jl.phase_distance_matrix(
            np.ascontiguousarray(phases.ravel(), dtype=np.float64)
        ),
        dtype=np.float64,
    )


def compute_npe_julia(phases: NDArray, max_radius: float) -> float:
    jl = _ensure()
    return float(
        jl.compute_npe(
            np.ascontiguousarray(phases.ravel(), dtype=np.float64),
            float(max_radius),
        )
    )
