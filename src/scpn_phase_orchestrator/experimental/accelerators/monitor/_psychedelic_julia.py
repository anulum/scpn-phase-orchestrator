# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for psychedelic observables

"""Julia backend for ``monitor/psychedelic.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._psychedelic_validation import (
    validate_psychedelic_backend_inputs,
    validate_psychedelic_entropy_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["entropy_from_phases_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "psychedelic.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.PsychedelicJL
    return _JULIA_MODULE


def entropy_from_phases_julia(phases: FloatArray, n_bins: int) -> float:
    """Compute phase-distribution entropy through the Julia backend."""

    phase_values, bin_count = validate_psychedelic_backend_inputs(phases, n_bins)
    if phase_values.size == 0:
        return 0.0
    jl = _ensure()
    return validate_psychedelic_entropy_backend_output(
        jl.entropy_from_phases(
            phase_values,
            bin_count,
        ),
        bin_count,
    )
