# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for recurrence kernels

"""Julia backend for ``monitor/recurrence.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["cross_recurrence_matrix_julia", "recurrence_matrix_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "recurrence.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.RecurrenceJL
    return _JULIA_MODULE


def recurrence_matrix_julia(
    traj_flat: NDArray,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> NDArray:
    jl = _ensure()
    return np.asarray(
        jl.recurrence_matrix(
            np.ascontiguousarray(traj_flat.ravel(), dtype=np.float64),
            int(t),
            int(d),
            float(epsilon),
            bool(angular),
        ),
        dtype=np.uint8,
    )


def cross_recurrence_matrix_julia(
    traj_a_flat: NDArray,
    traj_b_flat: NDArray,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> NDArray:
    jl = _ensure()
    return np.asarray(
        jl.cross_recurrence_matrix(
            np.ascontiguousarray(traj_a_flat.ravel(), dtype=np.float64),
            np.ascontiguousarray(traj_b_flat.ravel(), dtype=np.float64),
            int(t),
            int(d),
            float(epsilon),
            bool(angular),
        ),
        dtype=np.uint8,
    )
