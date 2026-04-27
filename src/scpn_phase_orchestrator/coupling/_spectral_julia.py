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
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["spectral_eig_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "spectral.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain  # type: ignore[import-untyped]

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.SpectralJL
    return _JULIA_MODULE


def spectral_eig_julia(
    knm_flat: NDArray,
    n: int,
) -> tuple[NDArray, NDArray]:
    jl = _ensure()
    eigvals, fiedler = jl.spectral_eig(
        np.ascontiguousarray(knm_flat, dtype=np.float64),
        int(n),
    )
    return (
        np.asarray(eigvals, dtype=np.float64),
        np.asarray(fiedler, dtype=np.float64),
    )
