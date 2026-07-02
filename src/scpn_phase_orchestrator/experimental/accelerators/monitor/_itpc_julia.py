# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for inter-trial phase coherence

"""Julia backend for ``monitor/itpc.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

from ._itpc_validation import (
    expected_compute_itpc_backend_output,
    expected_itpc_persistence_backend_output,
    validate_compute_itpc_backend_inputs,
    validate_compute_itpc_backend_output,
    validate_itpc_persistence_backend_inputs,
    validate_itpc_persistence_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = ["compute_itpc_julia", "itpc_persistence_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "itpc.jl"
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
    _JULIA_MODULE = JuliaMain.ITPC
    return _JULIA_MODULE


def compute_itpc_julia(phases_flat: FloatArray, n_trials: int, n_tp: int) -> FloatArray:
    """Compute inter-trial phase coherence through the Julia backend."""
    phases, n_trials, n_tp = validate_compute_itpc_backend_inputs(
        phases_flat,
        n_trials,
        n_tp,
    )
    if n_trials == 0 or n_tp == 0:
        return np.zeros(n_tp, dtype=np.float64)
    jl = _ensure()
    expected = expected_compute_itpc_backend_output(phases, n_trials, n_tp)
    return validate_compute_itpc_backend_output(
        jl.compute_itpc(
            phases,
            n_trials,
            n_tp,
        ),
        n_tp,
        expected=expected,
    )


def itpc_persistence_julia(
    phases_flat: FloatArray,
    n_trials: int,
    n_tp: int,
    pause_indices: IntArray,
) -> float:
    """Compute inter-trial phase-coherence persistence through the Julia backend."""
    phases, n_trials, n_tp, indices = validate_itpc_persistence_backend_inputs(
        phases_flat,
        n_trials,
        n_tp,
        pause_indices,
    )
    if indices.size == 0 or n_trials == 0 or n_tp == 0:
        return 0.0
    jl = _ensure()
    expected = expected_itpc_persistence_backend_output(
        phases,
        n_trials,
        n_tp,
        indices,
    )
    return validate_itpc_persistence_backend_output(
        jl.itpc_persistence(
            phases,
            n_trials,
            n_tp,
            indices,
        ),
        expected=expected,
    )
