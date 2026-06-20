# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for twin-confidence divergence

"""Julia backend for ``monitor/twin_confidence.py`` via ``juliacall``.

Loads ``julia/twin_confidence.jl`` on first call and exposes
``twin_divergence_julia`` with the same signature as the NumPy reference kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._twin_confidence_validation import (
    validate_twin_divergence_backend_inputs,
    validate_twin_divergence_backend_output,
)

__all__ = ["twin_divergence_julia"]
FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "twin_confidence.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.TwinConfidence
    return _JULIA_MODULE


def twin_divergence_julia(
    model_phases: FloatArray,
    observed_phases: FloatArray,
    model_order: FloatArray,
    observed_order: FloatArray,
    n: int,
    w: int,
    n_bins: int,
) -> FloatArray:
    """Compute the twin divergence pair through the Julia backend.

    Parameters
    ----------
    model_phases, observed_phases : FloatArray
        Model and observed phase vectors of length ``n``.
    model_order, observed_order : FloatArray
        Model and observed order-parameter windows of length ``w``.
    n, w, n_bins : int
        Phase count, order-window length, and histogram bin count.

    Returns
    -------
    FloatArray
        Two-element ``[phase_js_divergence, order_wasserstein]`` array.

    Raises
    ------
    ValueError
        If the inputs are invalid or the Julia kernel reports a contract failure.
    """
    (
        model_phases64,
        observed_phases64,
        model_order64,
        observed_order64,
        n_int,
        w_int,
        n_bins_int,
    ) = validate_twin_divergence_backend_inputs(
        model_phases,
        observed_phases,
        model_order,
        observed_order,
        n,
        w,
        n_bins,
    )
    jl = _ensure()
    return validate_twin_divergence_backend_output(
        jl.twin_divergence(
            np.ascontiguousarray(model_phases64, dtype=np.float64),
            np.ascontiguousarray(observed_phases64, dtype=np.float64),
            np.ascontiguousarray(model_order64, dtype=np.float64),
            np.ascontiguousarray(observed_order64, dtype=np.float64),
            n_int,
            w_int,
            n_bins_int,
        )
    )
