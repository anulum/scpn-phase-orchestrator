# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for spectral eigendecomposition

"""Julia backend for ``coupling/spectral.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

from ._spectral_validation import (
    validate_spectral_backend_inputs,
    validate_spectral_backend_output,
)

__all__ = ["spectral_eig_julia"]
FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "spectral.jl"
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
    _JULIA_MODULE = JuliaMain.SpectralJL
    return _JULIA_MODULE


def spectral_eig_julia(
    knm_flat: FloatArray,
    n: int,
) -> tuple[FloatArray, FloatArray]:
    """Compute coupling-spectrum eigenvalues and Fiedler vector with Julia."""
    k, n = validate_spectral_backend_inputs(knm_flat, n)
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy()
    jl = _ensure()
    eigvals, fiedler = jl.spectral_eig(
        k,
        n,
    )
    return validate_spectral_backend_output((eigvals, fiedler), n=n)
