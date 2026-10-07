# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for steady-state R

"""Julia backend for ``upde/basin_stability.py``."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)
from scpn_phase_orchestrator.upde._basin_stability_validation import (
    validate_basin_stability_inputs,
    validate_basin_stability_output,
)

__all__ = ["steady_state_r_julia"]

FloatArray = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "basin_stability.jl"


class _JuliaBasin(Protocol):
    """Original Julia module's deterministic trial entry point."""

    def steady_state_r(
        self,
        p: FloatArray,
        o: FloatArray,
        k: FloatArray,
        a: FloatArray,
        n: int,
        scale: float,
        dt: float,
        transient: int,
        measure: int,
    ) -> object:
        """Return the actual Julia scalar for public output validation."""
        ...


_JULIA_MODULE: _JuliaBasin | None = None


def _ensure() -> _JuliaBasin:
    """Build or load the backend artifact if it is missing, else raise."""
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    JuliaMain = require_julia_main()

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    try:
        JuliaMain.include(str(_JULIA_FILE))
    except Exception as exc:
        error_class = getattr(importlib.import_module("juliacall"), "JuliaError", None)
        if isinstance(error_class, type) and isinstance(exc, error_class):
            raise ImportError(f"Julia basin source cannot load: {exc}") from exc
        raise
    _JULIA_MODULE = cast("_JuliaBasin", JuliaMain.BasinStabilityJL)
    return _JULIA_MODULE


def steady_state_r_julia(
    phases_init: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    k_scale: float,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> float:
    """Measure one finite-window Kuramoto trial through the original Julia runtime.

    Parameters
    ----------
    phases_init, omegas : numpy.ndarray
        Finite real phases in radians and frequencies in rad/s, N entries.
    knm_flat, alpha_flat : numpy.ndarray
        Row-major N*N target/source rate coupling and radian phase lags.
    n : int
        Positive population count matching every supplied buffer.
    k_scale : float
        Finite coupling multiplier, without implicit population normalization.
    dt : float
        Finite positive timestep in seconds.
    n_transient, n_measure : int
        Nonnegative discarded and post-step measurement counts.

    Returns
    -------
    float
        Mean post-step R in [0,1]; the zero-window identity is zero.

    Raises
    ------
    ImportError
        If a required runtime artifact is unavailable.
    TypeError
        If input or output payloads contain unsupported numerical aliases.
    ValueError
        If shapes, finite domains, metadata or native arithmetic fail.
    """
    (
        p,
        o,
        k,
        a,
        n_i,
        k_scale_f,
        dt_f,
        n_transient_i,
        n_measure_i,
    ) = validate_basin_stability_inputs(
        phases_init,
        omegas,
        knm_flat,
        alpha_flat,
        n,
        k_scale,
        dt,
        n_transient,
        n_measure,
    )
    jl = _ensure()
    try:
        r = jl.steady_state_r(
            p,
            o,
            k,
            a,
            n_i,
            k_scale_f,
            dt_f,
            n_transient_i,
            n_measure_i,
        )
    except Exception as exc:
        error_class = getattr(importlib.import_module("juliacall"), "JuliaError", None)
        if isinstance(error_class, type) and isinstance(exc, error_class):
            raise ValueError(f"Julia basin trial failed: {exc}") from exc
        raise
    return validate_basin_stability_output(r)
