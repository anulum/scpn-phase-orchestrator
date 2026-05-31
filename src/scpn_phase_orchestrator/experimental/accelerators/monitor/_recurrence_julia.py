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
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._recurrence_validation import (
    expected_recurrence_backend_output,
    validate_cross_recurrence_backend_inputs,
    validate_recurrence_backend_inputs,
    validate_recurrence_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
ByteArray: TypeAlias = NDArray[np.uint8]

__all__ = ["cross_recurrence_matrix_julia", "recurrence_matrix_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "recurrence.jl"
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
    traj_flat: FloatArray,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> ByteArray:
    """Compute the recurrence matrix through the Julia backend."""

    p, t_int, d_int, radius, angular_flag = validate_recurrence_backend_inputs(
        traj_flat,
        t,
        d,
        epsilon,
        angular,
    )
    jl = _ensure()
    return validate_recurrence_backend_output(
        jl.recurrence_matrix(
            p,
            t_int,
            d_int,
            radius,
            angular_flag,
        ),
        t=t_int,
        name="recurrence_matrix",
        expected=expected_recurrence_backend_output(
            p,
            p,
            t=t_int,
            d=d_int,
            epsilon=radius,
            angular=angular_flag,
        ),
    )


def cross_recurrence_matrix_julia(
    traj_a_flat: FloatArray,
    traj_b_flat: FloatArray,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> ByteArray:
    """Compute the cross-recurrence matrix through the Julia backend."""

    (
        a,
        b,
        t_int,
        d_int,
        radius,
        angular_flag,
    ) = validate_cross_recurrence_backend_inputs(
        traj_a_flat,
        traj_b_flat,
        t,
        d,
        epsilon,
        angular,
    )
    jl = _ensure()
    return validate_recurrence_backend_output(
        jl.cross_recurrence_matrix(
            a,
            b,
            t_int,
            d_int,
            radius,
            angular_flag,
        ),
        t=t_int,
        name="cross_recurrence_matrix",
        expected=expected_recurrence_backend_output(
            a,
            b,
            t=t_int,
            d=d_int,
            epsilon=radius,
            angular=angular_flag,
        ),
    )
