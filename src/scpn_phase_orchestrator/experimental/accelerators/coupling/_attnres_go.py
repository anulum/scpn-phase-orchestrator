# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for multi-head AttnRes

"""Go backend for the multi-head AttnRes dispatcher.

Calls ``libattnres.so`` (built from ``go/attnres.go``) via ctypes.
Raises ``ImportError`` when the compiled library is missing — the
dispatcher then falls through to the next backend.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling._attnres_validation import (
    validate_attnres_backend_inputs,
    validate_attnres_backend_output,
)

from .._go_runtime import load_go_library

__all__ = ["attnres_modulate_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libattnres.so"

_LIB: ctypes.CDLL | None = None
FloatArray: TypeAlias = NDArray[np.float64]


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libattnres.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libattnres.so attnres.go"
        )
    lib = load_go_library(_LIB_PATH)
    if not hasattr(lib, "AttnResModulateV2"):
        raise ImportError(
            "Go AttnRes library lacks the dimension-aware V2 ABI; rebuild it"
        )
    lib.AttnResModulateV2.restype = ctypes.c_int
    lib.AttnResModulateV2.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # knm
        ctypes.POINTER(ctypes.c_double),  # theta
        ctypes.POINTER(ctypes.c_double),  # w_q
        ctypes.POINTER(ctypes.c_double),  # w_k
        ctypes.POINTER(ctypes.c_double),  # w_v
        ctypes.POINTER(ctypes.c_double),  # w_o
        ctypes.c_int,  # n
        ctypes.c_int,  # n_heads
        ctypes.c_int,  # d_model
        ctypes.c_int,  # block_size
        ctypes.c_double,  # temperature
        ctypes.c_double,  # lambda
        ctypes.POINTER(ctypes.c_double),  # out
    ]
    _LIB = lib
    return lib


def attnres_modulate_go(
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
    """Compute phase attention coupling through the original Go runtime.

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
        knm64,
        theta64,
        wq64,
        wk64,
        wv64,
        wo64,
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
    lib = _load_lib()
    out = np.zeros(n * n, dtype=np.float64)
    rc = lib.AttnResModulateV2(
        knm64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        theta64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wq64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wk64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wv64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wo64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_int(n_heads),
        ctypes.c_int(math.isqrt(wo64.size)),
        ctypes.c_int(block_size),
        ctypes.c_double(temperature),
        ctypes.c_double(lambda_),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go AttnResModulateV2 returned error code {rc}")
    return validate_attnres_backend_output(out, n=n, knm_flat=knm64)
