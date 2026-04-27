# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for winding numbers

"""Julia backend for ``monitor/winding.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["winding_numbers_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "winding.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain  # type: ignore[import-untyped]

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.WindingJL
    return _JULIA_MODULE


def winding_numbers_julia(
    phases_flat: NDArray,
    t: int,
    n: int,
) -> NDArray:
    jl = _ensure()
    return np.asarray(
        jl.winding_numbers(
            np.ascontiguousarray(phases_flat.ravel(), dtype=np.float64),
            int(t),
            int(n),
        ),
        dtype=np.int64,
    )
