# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for transfer entropy

"""Julia backend for ``monitor/transfer_entropy.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._te_validation import (
    expected_phase_te_backend_output,
    expected_te_matrix_backend_output,
    validate_phase_te_backend_inputs,
    validate_te_backend_output,
    validate_te_matrix_backend_inputs,
    validate_te_matrix_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["phase_te_julia", "te_matrix_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "transfer_entropy.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    """Build or load the backend artifact if it is missing, else raise."""
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.TransferEntropy
    return _JULIA_MODULE


def phase_te_julia(source: FloatArray, target: FloatArray, n_bins: int) -> float:
    """Compute pairwise phase transfer entropy through the Julia backend."""
    source, target, n_bins = validate_phase_te_backend_inputs(
        source,
        target,
        n_bins,
    )
    jl = _ensure()
    source_values = np.ascontiguousarray(source.ravel(), dtype=np.float64)
    target_values = np.ascontiguousarray(target.ravel(), dtype=np.float64)
    expected = expected_phase_te_backend_output(source_values, target_values, n_bins)
    return validate_te_backend_output(
        jl.phase_transfer_entropy(
            source_values,
            target_values,
            n_bins,
        ),
        n_bins=n_bins,
        expected=expected,
    )


def te_matrix_julia(
    phase_series: FloatArray,
    n_osc: int,
    n_time: int,
    n_bins: int,
) -> FloatArray:
    """Compute the phase transfer-entropy matrix through the Julia backend."""
    phase_series, n_osc, n_time, n_bins = validate_te_matrix_backend_inputs(
        phase_series,
        n_osc,
        n_time,
        n_bins,
    )
    jl = _ensure()
    series = np.ascontiguousarray(phase_series, dtype=np.float64)
    expected = expected_te_matrix_backend_output(series, n_osc, n_time, n_bins)
    return validate_te_matrix_backend_output(
        jl.transfer_entropy_matrix(
            series,
            n_osc,
            n_time,
            n_bins,
        ),
        n_osc=n_osc,
        n_bins=n_bins,
        expected=expected,
    )
