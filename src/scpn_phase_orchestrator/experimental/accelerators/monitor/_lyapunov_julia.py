# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for Lyapunov spectrum

"""Julia backend for ``monitor/lyapunov.py`` via ``juliacall``.

Loads ``julia/lyapunov.jl`` on first call and exposes
``lyapunov_spectrum_julia`` with the same signature as the Python
reference implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

from ._lyapunov_validation import (
    validate_lyapunov_backend_inputs,
    validate_lyapunov_backend_output,
)

__all__ = ["lyapunov_spectrum_julia"]
FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "lyapunov.jl"
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
    _JULIA_MODULE = JuliaMain.LyapunovSpectrum
    return _JULIA_MODULE


def lyapunov_spectrum_julia(
    phases_init: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    dt: float,
    n_steps: int,
    qr_interval: int,
    zeta: float,
    psi: float,
) -> FloatArray:
    """Estimate the Lyapunov spectrum through the Julia backend."""
    (
        phases_init,
        omegas,
        knm,
        alpha,
        dt,
        n_steps,
        qr_interval,
        zeta,
        psi,
    ) = validate_lyapunov_backend_inputs(
        phases_init,
        omegas,
        knm,
        alpha,
        dt,
        n_steps,
        qr_interval,
        zeta,
        psi,
    )
    n = int(phases_init.size)
    jl = _ensure()
    return validate_lyapunov_backend_output(
        jl.lyapunov_spectrum(
            np.ascontiguousarray(phases_init.ravel(), dtype=np.float64),
            np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
            np.ascontiguousarray(knm.ravel(), dtype=np.float64),
            np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
            n,
            float(dt),
            int(n_steps),
            int(qr_interval),
            float(zeta),
            float(psi),
        ),
        n,
    )
