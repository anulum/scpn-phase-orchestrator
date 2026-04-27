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
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["correlation_integral_julia", "kaplan_yorke_dimension_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "dimension.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain  # type: ignore[import-untyped]

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.DimensionJL
    return _JULIA_MODULE


def correlation_integral_julia(
    traj_flat: NDArray,
    t: int,
    d: int,
    idx_i: NDArray,
    idx_j: NDArray,
    epsilons: NDArray,
) -> NDArray:
    jl = _ensure()
    return np.asarray(
        jl.correlation_integral(
            np.ascontiguousarray(traj_flat, dtype=np.float64),
            int(t),
            int(d),
            np.ascontiguousarray(idx_i, dtype=np.int64),
            np.ascontiguousarray(idx_j, dtype=np.int64),
            np.ascontiguousarray(epsilons, dtype=np.float64),
        ),
        dtype=np.float64,
    )


def kaplan_yorke_dimension_julia(lyapunov_exponents: NDArray) -> float:
    jl = _ensure()
    return float(
        jl.kaplan_yorke_dimension(
            np.ascontiguousarray(lyapunov_exponents, dtype=np.float64),
        )
    )
