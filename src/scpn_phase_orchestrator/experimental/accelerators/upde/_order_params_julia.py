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
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._order_params_validation import (
    validate_layer_coherence_inputs,
    validate_order_parameter_inputs,
    validate_order_parameter_output,
    validate_plv_inputs,
    validate_unit_interval_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "order_parameter_julia",
    "plv_julia",
    "layer_coherence_julia",
]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "order_params.jl"
_JULIA_MODULE: Any | None = None


def _ensure_julia_loaded() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.OrderParams
    return _JULIA_MODULE


def order_parameter_julia(phases: FloatArray) -> tuple[float, float]:
    """Compute the Kuramoto order parameter.

    The calculation is delegated to the Julia backend.
    """

    phases64 = validate_order_parameter_inputs(phases)
    if phases64.size == 0:
        return (0.0, 0.0)
    jl = _ensure_julia_loaded()
    r, psi = jl.order_parameter(phases64)
    return validate_order_parameter_output(r, psi)


def plv_julia(phases_a: FloatArray, phases_b: FloatArray) -> float:
    """Compute phase-locking value.

    The calculation is delegated to the Julia backend.
    """

    a64, b64 = validate_plv_inputs(phases_a, phases_b)
    if a64.size == 0:
        return 0.0
    jl = _ensure_julia_loaded()
    return validate_unit_interval_output(jl.plv(a64, b64), name="PLV")


def layer_coherence_julia(phases: FloatArray, indices: IntArray) -> float:
    """Compute layer-wise phase coherence.

    The calculation is delegated to the Julia backend.
    """

    phases64, indices64 = validate_layer_coherence_inputs(phases, indices)
    if indices64.size == 0:
        return 0.0
    jl = _ensure_julia_loaded()
    return validate_unit_interval_output(
        jl.layer_coherence(phases64, indices64),
        name="layer coherence",
    )
