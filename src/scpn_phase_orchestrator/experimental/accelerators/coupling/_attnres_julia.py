# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for multi-head AttnRes

"""Julia backend for the multi-head AttnRes dispatcher.

Loads ``juliacall`` lazily — a plain ``import`` would crash every
Python process that does not have the Julia toolchain installed.
``_load_julia()`` in ``attention_residuals.py`` probes this module
plus ``juliacall`` at resolve time; missing toolchain surfaces as
``ImportError`` and the dispatcher falls through.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling._attnres_validation import (
    validate_attnres_backend_inputs,
    validate_attnres_backend_output,
)
from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

__all__ = ["attnres_modulate_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "attnres.jl"
FloatArray: TypeAlias = NDArray[np.float64]


class _JuliaAttnRes(Protocol):
    """Julia module contract checked by the shared numerical output validator."""

    def attnres_modulate(
        self,
        knm: FloatArray,
        theta: FloatArray,
        w_q: FloatArray,
        w_k: FloatArray,
        w_v: FloatArray,
        w_o: FloatArray,
        n: int,
        n_heads: int,
        block_size: int,
        temperature: float,
        lambda_: float,
    ) -> object:
        """Return Julia's actual row-major coupling values for validation."""
        ...


_JULIA_MODULE: _JuliaAttnRes | None = None


def _ensure_julia_loaded() -> _JuliaAttnRes:
    """Load the Julia backend runtime if not already loaded, else raise."""
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
            raise ImportError(f"Julia attention source cannot load: {exc}") from exc
        raise
    _JULIA_MODULE = cast("_JuliaAttnRes", JuliaMain.AttnRes)
    return _JULIA_MODULE


def attnres_modulate_julia(
    knm_flat: FloatArray,
    theta: FloatArray,
    w_q: FloatArray,
    w_k: FloatArray,
    w_v: FloatArray,
    w_o: FloatArray,
    n: int,
    n_heads: int,
    block_size: int,
    temperature: float,
    lambda_: float,
) -> FloatArray:
    """Compute phase attention coupling through the original Julia runtime.

    Parameters
    ----------
    knm_flat : numpy.ndarray
        Row-major symmetric coupling graph, ``N*N`` finite real values.
    theta : numpy.ndarray
        ``N`` oscillator phases in radians.
    w_q, w_k, w_v, w_o : numpy.ndarray
        Row-major projection buffers containing ``D*D`` finite real values.
        ``D`` is even, at least two, and divisible by the head count.
    n : int
        Non-negative oscillator count matching the supplied graph and phases.
    n_heads : int
        Positive head count dividing the model width.
    block_size : int
        ``-1`` selects all neighbours; a positive value limits index distance.
    temperature : float
        Positive finite softmax temperature.
    lambda_ : float
        Non-negative finite multiplicative strength.

    Returns
    -------
    numpy.ndarray
        Flattened float64 coupling with validated symmetry, zero diagonal
        and preserved absent edges. Empty graphs need no optional runtime.

    Raises
    ------
    ValueError
        On invalid types, shapes, controls, topology, numerical intermediates
        or returned coupling values. Boolean, complex, text and temporal
        aliases are rejected before coercion.
    ImportError
        If the optional runtime or a compatible compiled artifact is missing.
    """
    (
        knm_flat,
        theta,
        w_q,
        w_k,
        w_v,
        w_o,
        n,
        n_heads,
        block_size,
        temperature,
        lambda_,
    ) = validate_attnres_backend_inputs(
        knm_flat,
        theta,
        w_q,
        w_k,
        w_v,
        w_o,
        n,
        n_heads,
        block_size,
        temperature,
        lambda_,
    )
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    jl_mod = _ensure_julia_loaded()
    try:
        result = jl_mod.attnres_modulate(
            knm_flat,
            theta,
            w_q,
            w_k,
            w_v,
            w_o,
            n,
            n_heads,
            block_size,
            temperature,
            lambda_,
        )
    except Exception as exc:
        error_class = getattr(importlib.import_module("juliacall"), "JuliaError", None)
        if isinstance(error_class, type) and isinstance(exc, error_class):
            raise ValueError(f"Julia attention calculation failed: {exc}") from exc
        raise
    return validate_attnres_backend_output(result, n=n, knm_flat=knm_flat)
