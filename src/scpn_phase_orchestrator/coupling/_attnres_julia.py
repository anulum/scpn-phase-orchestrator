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

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["attnres_modulate_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[3] / "julia" / "attnres.jl"
_JULIA_MODULE: Any | None = None


def _ensure_julia_loaded() -> Any:
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    from juliacall import Main as JuliaMain  # type: ignore[import-untyped]

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    JuliaMain.include(str(_JULIA_FILE))
    _JULIA_MODULE = JuliaMain.AttnRes
    return _JULIA_MODULE


def attnres_modulate_julia(
    knm_flat: NDArray,
    theta: NDArray,
    w_q: NDArray,
    w_k: NDArray,
    w_v: NDArray,
    w_o: NDArray,
    n: int,
    n_heads: int,
    block_size: int,
    temperature: float,
    lambda_: float,
) -> NDArray:
    """Julia-backed multi-head AttnRes modulation. Signature matches
    the Rust / Go / Mojo FFIs."""
    jl_mod = _ensure_julia_loaded()
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
    return np.asarray(result, dtype=np.float64)
