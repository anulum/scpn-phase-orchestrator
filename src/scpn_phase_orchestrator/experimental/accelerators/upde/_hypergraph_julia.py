# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for hypergraph Kuramoto

"""Julia backend for ``upde/hypergraph.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["hypergraph_run_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "hypergraph.jl"
_JULIA_MODULE: Any | None = None
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.HypergraphJL
    return _JULIA_MODULE


def hypergraph_run_julia(
    phases: FloatArray,
    omegas: FloatArray,
    n: int,
    edge_nodes: IntArray,
    edge_offsets: IntArray,
    edge_strengths: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate hypergraph Kuramoto dynamics.

    The calculation is delegated to the Julia backend.
    """

    jl = _ensure()
    result = jl.hypergraph_run(
        np.ascontiguousarray(phases, dtype=np.float64),
        np.ascontiguousarray(omegas, dtype=np.float64),
        int(n),
        np.ascontiguousarray(edge_nodes, dtype=np.int64),
        np.ascontiguousarray(edge_offsets, dtype=np.int64),
        np.ascontiguousarray(edge_strengths, dtype=np.float64),
        np.ascontiguousarray(knm_flat, dtype=np.float64),
        np.ascontiguousarray(alpha_flat, dtype=np.float64),
        float(zeta),
        float(psi),
        float(dt),
        int(n_steps),
    )
    return np.asarray(result, dtype=np.float64)
