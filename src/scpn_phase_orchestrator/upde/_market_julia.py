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
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["market_order_parameter_julia", "market_plv_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "market.jl"
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
    phases_flat: NDArray,
    t: int,
    n: int,
) -> NDArray:
    jl = _ensure()
    result = jl.market_order_parameter(
        np.ascontiguousarray(phases_flat, dtype=np.float64),
        int(t),
        int(n),
    )
    return np.asarray(result, dtype=np.float64)


def market_plv_julia(
    phases_flat: NDArray,
    t: int,
    n: int,
    window: int,
) -> NDArray:
    jl = _ensure()
    result = jl.market_plv(
        np.ascontiguousarray(phases_flat, dtype=np.float64),
        int(t),
        int(n),
        int(window),
    )
    return np.asarray(result, dtype=np.float64)
