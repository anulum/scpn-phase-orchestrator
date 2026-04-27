# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for Poincaré kernels

"""Julia backend for ``monitor/poincare.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["phase_poincare_julia", "poincare_section_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "poincare.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain  # type: ignore[import-untyped]

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.PoincareJL
    return _JULIA_MODULE


def poincare_section_julia(
    traj_flat: NDArray,
    t: int,
    d: int,
    normal: NDArray,
    offset: float,
    direction_id: int,
) -> tuple[NDArray, NDArray, int]:
    jl = _ensure()
    cr, times, n_cr = jl.poincare_section(
        np.ascontiguousarray(traj_flat.ravel(), dtype=np.float64),
        int(t),
        int(d),
        np.ascontiguousarray(normal.ravel(), dtype=np.float64),
        float(offset),
        int(direction_id),
    )
    return (
        np.asarray(cr, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        int(n_cr),
    )


def phase_poincare_julia(
    phases_flat: NDArray,
    t: int,
    n: int,
    oscillator_idx: int,
    section_phase: float,
) -> tuple[NDArray, NDArray, int]:
    jl = _ensure()
    cr, times, n_cr = jl.phase_poincare(
        np.ascontiguousarray(phases_flat.ravel(), dtype=np.float64),
        int(t),
        int(n),
        int(oscillator_idx),
        float(section_phase),
    )
    return (
        np.asarray(cr, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        int(n_cr),
    )
