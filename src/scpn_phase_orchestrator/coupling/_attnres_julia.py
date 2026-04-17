# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for AttnRes coupling modulation

"""Julia backend for the AttnRes modulation fallback chain.

Loads ``juliacall`` lazily — a plain ``import`` here would crash every
Python process that does not have the Julia toolchain installed, which
defeats the point of the fallback chain. Instead, this module raises
``ImportError`` from its loader when either ``juliacall`` or the Julia
side-file is missing, and the dispatcher silently moves on to the next
backend (Go, or finally the NumPy fallback).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["attnres_modulate_julia"]

_JULIA_FILE = (
    Path(__file__).resolve().parents[3] / "julia" / "attnres.jl"
)

_JULIA_MODULE: Any | None = None


def _ensure_julia_loaded() -> Any:
    """Import juliacall + include ``julia/attnres.jl`` on first call."""
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE

    from juliacall import Main as JuliaMain  # type: ignore[import-not-found]

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.AttnRes
    return _JULIA_MODULE


def attnres_modulate_julia(
    knm_flat: NDArray,
    theta: NDArray,
    n: int,
    block_size: int,
    temperature: float,
    lambda_: float,
) -> NDArray:
    """Julia-backed AttnRes modulation. Signature matches the Rust FFI."""
    jl_mod = _ensure_julia_loaded()
    result = jl_mod.attnres_modulate(
        knm_flat, theta, n, block_size, temperature, lambda_
    )
    return np.asarray(result, dtype=np.float64)
