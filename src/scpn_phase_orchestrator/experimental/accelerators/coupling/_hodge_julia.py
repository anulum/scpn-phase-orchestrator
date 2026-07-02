# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for Hodge decomposition

"""Julia backend for ``coupling/hodge.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

from ._hodge_validation import (
    validate_hodge_backend_inputs,
    validate_hodge_backend_output,
)

__all__ = ["hodge_decomposition_julia"]
FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "hodge.jl"
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
    _JULIA_MODULE = JuliaMain.HodgeJL
    return _JULIA_MODULE


def hodge_decomposition_julia(
    knm_flat: FloatArray,
    phases: FloatArray,
    n: int,
    edges_flat: NDArray[np.int64],
    n_edges: int,
    tris_flat: NDArray[np.int64],
    n_tris: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute the Hodge gradient, curl, and harmonic flow matrices with Julia."""
    k, p, n, edges, n_edges, tris, n_tris = validate_hodge_backend_inputs(
        knm_flat,
        phases,
        n,
        edges_flat,
        n_edges,
        tris_flat,
        n_tris,
    )
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    jl = _ensure()
    g, c, h = jl.hodge_decomposition(
        k,
        p,
        n,
        edges,
        n_edges,
        tris,
        n_tris,
    )
    return validate_hodge_backend_output((g, c, h), n=n)
