# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for OPT-entropy

"""Julia backend for ``monitor/opt_entropy.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._opt_entropy_validation import (
    expected_ordinal_pattern_backend_output,
    expected_transition_entropy_backend_output,
    ordinal_window_count,
    validate_ordinal_pattern_backend_output,
    validate_transition_entropy_backend_inputs,
    validate_transition_entropy_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = ["ordinal_pattern_sequence_julia", "transition_entropy_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "opt_entropy.jl"
_JULIA_MODULE: Any | None = None


def _ensure() -> Any:
    """Build or load the backend artifact if it is missing, else raise."""
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.OptEntropy
    return _JULIA_MODULE


def ordinal_pattern_sequence_julia(
    series: FloatArray,
    dimension: int,
    delay: int,
) -> IntArray:
    """Compute the ordinal-pattern code sequence through the Julia backend."""
    s, d, tau = validate_transition_entropy_backend_inputs(series, dimension, delay)
    jl = _ensure()
    count = ordinal_window_count(int(s.size), d, tau)
    return validate_ordinal_pattern_backend_output(
        np.asarray(jl.ordinal_pattern_sequence(s, d, tau)),
        n_windows=count,
        dimension=d,
        expected=expected_ordinal_pattern_backend_output(s, d, tau),
    )


def transition_entropy_julia(
    series: FloatArray,
    dimension: int,
    delay: int,
) -> float:
    """Compute the normalised transition entropy through the Julia backend."""
    s, d, tau = validate_transition_entropy_backend_inputs(series, dimension, delay)
    jl = _ensure()
    return validate_transition_entropy_backend_output(
        jl.transition_entropy(s, d, tau),
        expected=expected_transition_entropy_backend_output(s, d, tau),
    )
