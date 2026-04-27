# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for order parameters

"""Julia backend for ``upde/order_params.py``.

Loads ``juliacall`` lazily — ``_load_julia()`` in the dispatcher
probes this module at resolve time and raises ``ImportError`` when
the toolchain is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "order_parameter_julia",
    "plv_julia",
    "layer_coherence_julia",
]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "order_params.jl"
_JULIA_MODULE: Any | None = None


def _ensure_julia_loaded() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain  # type: ignore[import-untyped]

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.OrderParams
    return _JULIA_MODULE


def order_parameter_julia(phases: NDArray) -> tuple[float, float]:
    jl = _ensure_julia_loaded()
    r, psi = jl.order_parameter(np.ascontiguousarray(phases.ravel(), dtype=np.float64))
    return float(r), float(psi)


def plv_julia(phases_a: NDArray, phases_b: NDArray) -> float:
    jl = _ensure_julia_loaded()
    return float(
        jl.plv(
            np.ascontiguousarray(phases_a.ravel(), dtype=np.float64),
            np.ascontiguousarray(phases_b.ravel(), dtype=np.float64),
        )
    )


def layer_coherence_julia(phases: NDArray, indices: NDArray) -> float:
    jl = _ensure_julia_loaded()
    return float(
        jl.layer_coherence(
            np.ascontiguousarray(phases.ravel(), dtype=np.float64),
            np.ascontiguousarray(indices.ravel(), dtype=np.int64),
        )
    )
