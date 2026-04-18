# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for embedding primitives

"""Julia backend for ``monitor/embedding.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "delay_embed_julia",
    "mutual_information_julia",
    "nearest_neighbor_distances_julia",
]

_JULIA_FILE = (
    Path(__file__).resolve().parents[3] / "julia" / "embedding.jl"
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
    _JULIA_MODULE = JuliaMain.EmbeddingJL
    return _JULIA_MODULE


def delay_embed_julia(
    signal: NDArray, delay: int, dimension: int,
) -> NDArray:
    jl = _ensure()
    return np.asarray(
        jl.delay_embed(
            np.ascontiguousarray(signal.ravel(), dtype=np.float64),
            int(delay), int(dimension),
        ),
        dtype=np.float64,
    )


def mutual_information_julia(
    signal: NDArray, lag: int, n_bins: int,
) -> float:
    jl = _ensure()
    return float(
        jl.mutual_information(
            np.ascontiguousarray(signal.ravel(), dtype=np.float64),
            int(lag), int(n_bins),
        )
    )


def nearest_neighbor_distances_julia(
    embedded: NDArray, t: int, m: int,
) -> tuple[NDArray, NDArray]:
    jl = _ensure()
    dist, idx = jl.nearest_neighbor_distances(
        np.ascontiguousarray(embedded.ravel(), dtype=np.float64),
        int(t), int(m),
    )
    return (
        np.asarray(dist, dtype=np.float64),
        np.asarray(idx, dtype=np.int64),
    )
