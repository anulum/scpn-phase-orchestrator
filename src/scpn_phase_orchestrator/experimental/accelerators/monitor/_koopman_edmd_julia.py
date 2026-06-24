# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for Koopman EDMD

"""Julia backend for ``monitor/koopman_edmd.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._koopman_edmd_validation import (
    validate_edmd_backend_inputs,
    validate_edmd_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["koopman_edmd_solve_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "koopman_edmd.jl"
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
    _JULIA_MODULE = JuliaMain.KoopmanEdmd
    return _JULIA_MODULE


def koopman_edmd_solve_julia(
    x_lift: FloatArray,
    inputs: FloatArray,
    y_lift: FloatArray,
    states: FloatArray,
    regularisation: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Solve the EDMD-with-control least squares through the Julia backend."""
    dims = validate_edmd_backend_inputs(x_lift, inputs, y_lift, states)
    jl = _ensure()
    a, b, c = jl.koopman_edmd_solve(
        np.ascontiguousarray(x_lift, dtype=np.float64),
        np.ascontiguousarray(inputs, dtype=np.float64),
        np.ascontiguousarray(y_lift, dtype=np.float64),
        np.ascontiguousarray(states, dtype=np.float64),
        float(regularisation),
    )
    return validate_edmd_backend_output(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
        dims,
    )
