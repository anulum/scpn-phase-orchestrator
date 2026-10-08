# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for chimera local-R kernel

"""Julia backend for ``monitor/chimera.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)
from scpn_phase_orchestrator.monitor._chimera_validation import (
    validate_chimera_backend_inputs,
    validate_chimera_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["local_order_parameter_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "chimera.jl"


class _JuliaChimera(Protocol):
    """Actual Julia module's local-order entry point."""

    def local_order_parameter(
        self, phases: FloatArray, knm_flat: FloatArray, n: int
    ) -> object:
        """Return native output for typed shape/domain validation."""
        ...


_JULIA_MODULE: _JuliaChimera | None = None


def _ensure() -> _JuliaChimera:
    """Load and cache the actual Julia chimera source module, else raise."""
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    JuliaMain = require_julia_main()

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    from juliacall import JuliaError

    try:
        JuliaMain.include(str(_JULIA_FILE))
    except JuliaError as exc:
        raise ImportError(f"Julia chimera source cannot load: {exc}") from exc
    _JULIA_MODULE = cast("_JuliaChimera", JuliaMain.ChimeraJL)
    return _JULIA_MODULE


def local_order_parameter_julia(
    phases: FloatArray,
    knm_flat: FloatArray,
    n: int,
) -> FloatArray:
    """Measure positive non-self adjacency through the actual Julia module.

    Parameters
    ----------
    phases : FloatArray
        Finite real radian phases, N entries; original aliases are refused.
    knm_flat : FloatArray
        Finite row-major N*N coupling; positive off-diagonal entries define
        equally weighted neighbours. Diagonal tolerance is 1e-15.
    n : int
        Plain nonnegative count matching both buffers.

    Returns
    -------
    FloatArray
        N finite local magnitudes in [0,1]; the empty identity requires no
        Julia module execution.

    Raises
    ------
    ImportError
        If the actual Julia runtime or source cannot load.
    ValueError
        If source types, dimensions or numerical domains are invalid.
    """
    phases_vec, knm_vec, n = validate_chimera_backend_inputs(phases, knm_flat, n)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    jl = _ensure()
    return validate_chimera_backend_output(
        jl.local_order_parameter(
            phases_vec,
            knm_vec,
            n,
        ),
        n,
    )
