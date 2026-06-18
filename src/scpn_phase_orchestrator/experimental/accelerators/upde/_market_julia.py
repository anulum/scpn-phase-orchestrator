# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for market PLV / R(t)

"""Julia backend for ``upde/market.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._market_validation import (
    validate_market_order_inputs,
    validate_market_order_output,
    validate_market_plv_inputs,
    validate_market_plv_output,
)

__all__ = ["market_order_parameter_julia", "market_plv_julia"]

FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "market.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.MarketJL
    return _JULIA_MODULE


def market_order_parameter_julia(
    phases_flat: FloatArray,
    t: int,
    n: int,
) -> FloatArray:
    """Compute market phase order parameter.

    The calculation is delegated to the Julia backend.
    """
    p, t_i, n_i = validate_market_order_inputs(phases_flat, t, n)
    jl = _ensure()
    result = jl.market_order_parameter(
        p,
        t_i,
        n_i,
    )
    return validate_market_order_output(result, t=t_i)


def market_plv_julia(
    phases_flat: FloatArray,
    t: int,
    n: int,
    window: int,
) -> FloatArray:
    """Compute market phase-locking value.

    The calculation is delegated to the Julia backend.
    """
    p, t_i, n_i, window_i = validate_market_plv_inputs(
        phases_flat,
        t,
        n,
        window,
    )
    jl = _ensure()
    result = jl.market_plv(
        p,
        t_i,
        n_i,
        window_i,
    )
    return validate_market_plv_output(result, t=t_i, n=n_i, window=window_i)
