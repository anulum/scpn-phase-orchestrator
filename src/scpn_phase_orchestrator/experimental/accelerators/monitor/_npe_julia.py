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
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._npe_validation import (
    validate_npe_backend_inputs,
    validate_npe_backend_output,
    validate_phase_distance_backend_input,
    validate_phase_distance_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["phase_distance_matrix_julia", "compute_npe_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "npe.jl"
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


def phase_distance_matrix_julia(phases: FloatArray) -> FloatArray:
    """Compute pairwise wrapped phase distances through the Julia backend."""

    p = validate_phase_distance_backend_input(phases)
    jl = _ensure()
    return validate_phase_distance_backend_output(
        jl.phase_distance_matrix(p),
        n_phases=p.size,
    )


def compute_npe_julia(phases: FloatArray, max_radius: float) -> float:
    """Compute normalised phase entropy through the Julia backend."""

    p, radius = validate_npe_backend_inputs(phases, max_radius)
    jl = _ensure()
    return validate_npe_backend_output(
        jl.compute_npe(
            p,
            radius,
        )
    )
