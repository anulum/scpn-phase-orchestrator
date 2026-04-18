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
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["lyapunov_spectrum_julia"]

_JULIA_FILE = (
    Path(__file__).resolve().parents[3] / "julia" / "lyapunov.jl"
)
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain  # type: ignore[import-untyped]

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.LyapunovSpectrum
    return _JULIA_MODULE


def lyapunov_spectrum_julia(
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    dt: float,
    n_steps: int,
    qr_interval: int,
    zeta: float,
    psi: float,
) -> NDArray:
    jl = _ensure()
    n = int(phases_init.size)
    return np.asarray(
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
        dtype=np.float64,
    )
