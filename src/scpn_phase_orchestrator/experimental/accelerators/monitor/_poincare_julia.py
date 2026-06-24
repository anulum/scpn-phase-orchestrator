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
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._poincare_validation import (
    validate_phase_poincare_backend_inputs,
    validate_poincare_backend_outputs,
    validate_poincare_section_backend_inputs,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["phase_poincare_julia", "poincare_section_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "poincare.jl"
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
    _JULIA_MODULE = JuliaMain.PoincareJL
    return _JULIA_MODULE


def poincare_section_julia(
    traj_flat: FloatArray,
    t: int,
    d: int,
    normal: FloatArray,
    offset: float,
    direction_id: int,
) -> tuple[FloatArray, FloatArray, int]:
    """Extract Poincare section crossings through the Julia backend."""
    traj, t, d, nrm, offset, direction_id = validate_poincare_section_backend_inputs(
        traj_flat,
        t,
        d,
        normal,
        offset,
        direction_id,
    )
    jl = _ensure()
    cr, times, n_cr = jl.poincare_section(
        traj,
        t,
        d,
        nrm,
        offset,
        direction_id,
    )
    return validate_poincare_backend_outputs(
        cr,
        times,
        n_cr,
        t=t,
        dim=d,
    )


def phase_poincare_julia(
    phases_flat: FloatArray,
    t: int,
    n: int,
    oscillator_idx: int,
    section_phase: float,
) -> tuple[FloatArray, FloatArray, int]:
    """Compute phase-space Poincare diagnostics through the Julia backend."""
    phases, t, n, oscillator_idx, section_phase = (
        validate_phase_poincare_backend_inputs(
            phases_flat,
            t,
            n,
            oscillator_idx,
            section_phase,
        )
    )
    jl = _ensure()
    cr, times, n_cr = jl.phase_poincare(
        phases,
        t,
        n,
        oscillator_idx,
        section_phase,
    )
    return validate_poincare_backend_outputs(
        cr,
        times,
        n_cr,
        t=t,
        dim=n,
    )
