# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for dimension kernels

"""Julia backend for ``monitor/dimension.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

from ._dimension_validation import (
    expected_correlation_integral_backend_output,
    expected_kaplan_yorke_backend_output,
    validate_correlation_integral_backend_inputs,
    validate_correlation_integral_backend_output,
    validate_kaplan_yorke_backend_input,
    validate_kaplan_yorke_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = ["correlation_integral_julia", "kaplan_yorke_dimension_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "dimension.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    """Build or load the backend artifact if it is missing, else raise."""
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    JuliaMain = require_julia_main()

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.DimensionJL
    return _JULIA_MODULE


def correlation_integral_julia(
    traj_flat: FloatArray,
    t: int,
    d: int,
    idx_i: IntArray,
    idx_j: IntArray,
    epsilons: FloatArray,
) -> FloatArray:
    """Compute the phase-space correlation integral through the Julia backend."""
    traj, t_int, d_int, ii, jj, eps = validate_correlation_integral_backend_inputs(
        traj_flat,
        t,
        d,
        idx_i,
        idx_j,
        epsilons,
    )
    jl = _ensure()
    expected = expected_correlation_integral_backend_output(
        traj,
        t_int,
        d_int,
        ii,
        jj,
        eps,
    )
    return validate_correlation_integral_backend_output(
        jl.correlation_integral(
            traj,
            t_int,
            d_int,
            ii,
            jj,
            eps,
        ),
        eps,
        expected=expected,
    )


def kaplan_yorke_dimension_julia(lyapunov_exponents: FloatArray) -> float:
    """Estimate the Kaplan-Yorke dimension through the Julia backend."""
    le = validate_kaplan_yorke_backend_input(lyapunov_exponents)
    jl = _ensure()
    expected = expected_kaplan_yorke_backend_output(le)
    return validate_kaplan_yorke_backend_output(
        jl.kaplan_yorke_dimension(
            le,
        ),
        le,
        expected=expected,
    )
