# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for PAC

"""Julia backend for ``upde/pac.py``. Loads ``juliacall`` lazily."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["modulation_index_julia", "pac_matrix_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "pac.jl"
_JULIA_MODULE: Any | None = None


def _ensure_julia_loaded() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain  # type: ignore[import-untyped]

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.PAC
    return _JULIA_MODULE


def modulation_index_julia(
    theta_low: NDArray, amp_high: NDArray, n_bins: int
) -> float:
    jl = _ensure_julia_loaded()
    return float(
        jl.modulation_index(
            np.ascontiguousarray(theta_low.ravel(), dtype=np.float64),
            np.ascontiguousarray(amp_high.ravel(), dtype=np.float64),
            n_bins,
        )
    )


def pac_matrix_julia(
    phases_flat: NDArray,
    amplitudes_flat: NDArray,
    t: int,
    n: int,
    n_bins: int,
) -> NDArray:
    jl = _ensure_julia_loaded()
    return np.asarray(
        jl.pac_matrix(
            np.ascontiguousarray(phases_flat, dtype=np.float64),
            np.ascontiguousarray(amplitudes_flat, dtype=np.float64),
            t,
            n,
            n_bins,
        ),
        dtype=np.float64,
    )
