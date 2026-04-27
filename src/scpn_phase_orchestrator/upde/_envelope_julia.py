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
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["envelope_modulation_depth_julia", "extract_envelope_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "envelope.jl"
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


def extract_envelope_julia(amps: NDArray, window: int) -> NDArray:
    jl = _ensure()
    return np.asarray(
        jl.extract_envelope(
            np.ascontiguousarray(amps.ravel(), dtype=np.float64),
            int(window),
        ),
        dtype=np.float64,
    )


def envelope_modulation_depth_julia(env: NDArray) -> float:
    jl = _ensure()
    return float(
        jl.envelope_modulation_depth(
            np.ascontiguousarray(env.ravel(), dtype=np.float64),
        )
    )
