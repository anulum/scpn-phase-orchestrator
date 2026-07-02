# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for spatial coupling modulation

"""Julia backend for ``coupling/spatial_modulator.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

from ._spatial_modulator_validation import (
    validate_spatial_modulator_inputs,
    validate_spatial_modulator_output,
)

__all__ = ["spatial_modulate_julia"]
FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "spatial_modulator.jl"
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
    _JULIA_MODULE = JuliaMain.SpatialModulatorJL
    return _JULIA_MODULE


def spatial_modulate_julia(
    k_nm_flat: FloatArray,
    positions_flat: FloatArray,
    n: int,
    dim: int,
    k_base: float,
    decay_form_code: int,
    decay_exponent: float,
    decay_length_scale: float,
    epsilon: float,
) -> FloatArray:
    """Compute a spatially modulated coupling matrix with Julia."""
    k, p, n, dim, k_base, form, exponent, length, eps = (
        validate_spatial_modulator_inputs(
            k_nm_flat,
            positions_flat,
            n,
            dim,
            k_base,
            decay_form_code,
            decay_exponent,
            decay_length_scale,
            epsilon,
        )
    )
    jl = _ensure()
    out = jl.spatial_modulate(k, p, n, dim, k_base, form, exponent, length, eps)
    return validate_spatial_modulator_output(out, n=n)
