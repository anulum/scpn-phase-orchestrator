# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for transfer entropy

"""Julia backend for ``monitor/transfer_entropy.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["phase_te_julia", "te_matrix_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "transfer_entropy.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.TransferEntropy
    return _JULIA_MODULE


def phase_te_julia(source: NDArray, target: NDArray, n_bins: int) -> float:
    jl = _ensure()
    return float(
        jl.phase_transfer_entropy(
            np.ascontiguousarray(source.ravel(), dtype=np.float64),
            np.ascontiguousarray(target.ravel(), dtype=np.float64),
            n_bins,
        )
    )


def te_matrix_julia(
    phase_series: NDArray,
    n_osc: int,
    n_time: int,
    n_bins: int,
) -> NDArray:
    jl = _ensure()
    return np.asarray(
        jl.transfer_entropy_matrix(
            np.ascontiguousarray(phase_series, dtype=np.float64),
            n_osc,
            n_time,
            n_bins,
        ),
        dtype=np.float64,
    )
