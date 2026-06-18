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
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._pac_validation import (
    validate_modulation_index_inputs,
    validate_modulation_index_output,
    validate_pac_matrix_inputs,
    validate_pac_matrix_output,
)

__all__ = ["modulation_index_julia", "pac_matrix_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "pac.jl"
_JULIA_MODULE: Any | None = None

FloatArray: TypeAlias = NDArray[np.float64]


def _ensure_julia_loaded() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.PAC
    return _JULIA_MODULE


def modulation_index_julia(
    theta_low: FloatArray, amp_high: FloatArray, n_bins: int
) -> float:
    """Compute phase-amplitude coupling modulation index.

    The calculation is delegated to the Julia backend.
    """
    theta, amp, bins = validate_modulation_index_inputs(theta_low, amp_high, n_bins)
    if theta.size == 0:
        return 0.0
    jl = _ensure_julia_loaded()
    return validate_modulation_index_output(
        jl.modulation_index(
            theta,
            amp,
            bins,
        ),
    )


def pac_matrix_julia(
    phases_flat: FloatArray,
    amplitudes_flat: FloatArray,
    t: int,
    n: int,
    n_bins: int,
) -> FloatArray:
    """Compute the phase-amplitude coupling matrix.

    The calculation is delegated to the Julia backend.
    """
    phases, amplitudes, t_i, n_i, bins = validate_pac_matrix_inputs(
        phases_flat,
        amplitudes_flat,
        t,
        n,
        n_bins,
    )
    jl = _ensure_julia_loaded()
    result = np.asarray(
        jl.pac_matrix(
            phases,
            amplitudes,
            t_i,
            n_i,
            bins,
        ),
        dtype=np.float64,
    )
    return validate_pac_matrix_output(result, n=n_i)
