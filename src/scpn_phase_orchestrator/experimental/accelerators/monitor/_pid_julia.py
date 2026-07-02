# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for partial information decomposition

"""Julia backend for ``monitor/pid.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

from ._pid_validation import (
    validate_pid_backend_inputs,
    validate_pid_scalar_output,
)

__all__ = ["pid_decomposition_julia"]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "pid.jl"
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
    _JULIA_MODULE = JuliaMain.PidJL
    return _JULIA_MODULE


def pid_decomposition_julia(
    phase_history_flat: FloatArray,
    t: int,
    n: int,
    group_a: IntArray,
    group_b: IntArray,
    n_bins: int,
) -> tuple[float, float]:
    """Compute (redundancy, synergy) through the Julia backend."""
    history, t, n, group_a_idx, group_b_idx, bins = validate_pid_backend_inputs(
        phase_history_flat, t, n, group_a, group_b, n_bins
    )
    jl = _ensure()
    red, syn = jl.pid_decomposition(
        history,
        t,
        n,
        group_a_idx,
        group_b_idx,
        bins,
    )
    return (
        validate_pid_scalar_output(red, name="redundancy"),
        validate_pid_scalar_output(syn, name="synergy"),
    )
