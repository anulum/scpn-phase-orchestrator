# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for envelope kernels

"""Julia backend for ``upde/envelope.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._envelope_validation import (
    validate_envelope_modulation_input,
    validate_envelope_modulation_output,
    validate_extract_envelope_input,
    validate_extract_envelope_output,
)

__all__ = ["envelope_modulation_depth_julia", "extract_envelope_julia"]
FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "envelope.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.EnvelopeJL
    return _JULIA_MODULE


def extract_envelope_julia(amps: FloatArray, window: int) -> FloatArray:
    """Extract the analytic phase envelope.

    The calculation is delegated to the Julia backend.
    """

    a, window_i = validate_extract_envelope_input(amps, window)
    if a.size == 0:
        return np.zeros(0, dtype=np.float64)
    if window_i >= a.size:
        rms = float(np.sqrt(np.mean(a * a)))
        return np.full(a.size, rms, dtype=np.float64)
    jl = _ensure()
    return validate_extract_envelope_output(
        np.asarray(
            jl.extract_envelope(
                a,
                window_i,
            ),
            dtype=np.float64,
        ),
        n=int(a.size),
    )


def envelope_modulation_depth_julia(env: FloatArray) -> float:
    """Compute envelope modulation depth.

    The calculation is delegated to the Julia backend.
    """

    e = validate_envelope_modulation_input(env)
    if e.size == 0:
        return 0.0
    vmax = float(np.max(e))
    vmin = float(np.min(e))
    if vmax + vmin <= 0.0:
        return 0.0
    jl = _ensure()
    return validate_envelope_modulation_output(
        jl.envelope_modulation_depth(
            e,
        )
    )
