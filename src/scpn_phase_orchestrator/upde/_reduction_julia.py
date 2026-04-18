# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for Ott-Antonsen reduction

"""Julia backend for ``upde/reduction.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["oa_run_julia"]

_JULIA_FILE = (
    Path(__file__).resolve().parents[3] / "julia" / "reduction.jl"
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
    _JULIA_MODULE = JuliaMain.ReductionJL
    return _JULIA_MODULE


def oa_run_julia(
    z_re: float,
    z_im: float,
    omega_0: float,
    delta: float,
    k_coupling: float,
    dt: float,
    n_steps: int,
) -> tuple[float, float, float, float]:
    jl = _ensure()
    result = jl.oa_run(
        float(z_re), float(z_im),
        float(omega_0), float(delta),
        float(k_coupling), float(dt),
        int(n_steps),
    )
    return (float(result[0]), float(result[1]),
            float(result[2]), float(result[3]))
