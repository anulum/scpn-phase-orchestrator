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
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ._winding_validation import validate_winding_backend_inputs

__all__ = ["winding_numbers_julia"]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "winding.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.WindingJL
    return _JULIA_MODULE


def winding_numbers_julia(
    phases_flat: FloatArray,
    t: int,
    n: int,
) -> IntArray:
    """Compute oscillator winding numbers through the Julia backend."""

    phases, t, n = validate_winding_backend_inputs(phases_flat, t, n)
    jl = _ensure()
    return cast(
        "IntArray",
        np.asarray(
            jl.winding_numbers(
                phases,
                t,
                n,
            )
        ),
    )
