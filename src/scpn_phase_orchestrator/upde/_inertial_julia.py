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
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["inertial_step_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "inertial.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
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
    theta: NDArray,
    omega_dot: NDArray,
    power: NDArray,
    knm_flat: NDArray,
    inertia: NDArray,
    damping: NDArray,
    n: int,
    dt: float,
) -> tuple[NDArray, NDArray]:
    jl = _ensure()
    new_theta, new_omega_dot = jl.inertial_step(
        np.ascontiguousarray(theta, dtype=np.float64),
        np.ascontiguousarray(omega_dot, dtype=np.float64),
        np.ascontiguousarray(power, dtype=np.float64),
        np.ascontiguousarray(knm_flat, dtype=np.float64),
        np.ascontiguousarray(inertia, dtype=np.float64),
        np.ascontiguousarray(damping, dtype=np.float64),
        int(n),
        float(dt),
    )
    return (
        np.asarray(new_theta, dtype=np.float64),
        np.asarray(new_omega_dot, dtype=np.float64),
    )
