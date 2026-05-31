# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for psychedelic observables

"""Julia backend for ``monitor/psychedelic.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["entropy_from_phases_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "psychedelic.jl"
_JULIA_MODULE: Any | None = None


def _contains_boolean_alias(value: object) -> bool:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _validated_backend_inputs(phases: object, n_bins: object) -> tuple[FloatArray, int]:
    if _contains_boolean_alias(phases):
        raise ValueError("phases must not contain boolean values")
    raw = np.asarray(phases)
    if np.iscomplexobj(raw):
        raise ValueError("phases must contain real-valued samples")
    try:
        phase_values = raw.astype(np.float64, copy=True).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a finite real-valued vector") from exc
    if not np.all(np.isfinite(phase_values)):
        raise ValueError("phases must contain only finite values")
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, int):
        raise TypeError("n_bins must be an integer greater than or equal to 2")
    bin_count = int(n_bins)
    if bin_count < 2:
        raise ValueError("n_bins must be greater than or equal to 2")
    return np.ascontiguousarray(phase_values, dtype=np.float64), bin_count


def _ensure() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.PsychedelicJL
    return _JULIA_MODULE


def entropy_from_phases_julia(phases: FloatArray, n_bins: int) -> float:
    """Compute phase-distribution entropy through the Julia backend."""

    phase_values, bin_count = _validated_backend_inputs(phases, n_bins)
    jl = _ensure()
    return cast(
        "float",
        jl.entropy_from_phases(
            phase_values,
            bin_count,
        ),
    )
