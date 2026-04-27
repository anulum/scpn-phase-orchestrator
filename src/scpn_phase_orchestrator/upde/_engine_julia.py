# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for UPDE engine

"""Julia backend for ``upde/engine.py``'s batched ``run()`` kernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["upde_run_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "upde_engine.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain  # type: ignore[import-untyped]

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.UPDEEngineJL
    return _JULIA_MODULE


def upde_run_julia(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> NDArray:
    jl = _ensure()
    n = int(phases.size)
    return np.asarray(
        jl.upde_run(
            np.ascontiguousarray(phases.ravel(), dtype=np.float64),
            np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
            np.ascontiguousarray(knm.ravel(), dtype=np.float64),
            np.ascontiguousarray(alpha.ravel(), dtype=np.float64),
            n,
            float(zeta),
            float(psi),
            float(dt),
            int(n_steps),
            str(method),
            int(n_substeps),
            float(atol),
            float(rtol),
        ),
        dtype=np.float64,
    )
