# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for chimera local-R kernel

"""Go backend for ``monitor/chimera.py`` via ``libchimera.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor._chimera_validation import (
    validate_chimera_backend_inputs,
    validate_chimera_backend_output,
)

from .._go_runtime import load_go_library

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["local_order_parameter_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libchimera.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libchimera.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libchimera.so chimera.go"
        )
    lib = load_go_library(_LIB_PATH)
    try:
        native = lib.LocalOrderParameterV2
    except AttributeError as exc:
        raise ImportError(
            "libchimera.so lacks LocalOrderParameterV2; rebuild chimera.go"
        ) from exc
    native.restype = ctypes.c_int
    native.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    _LIB = lib
    return lib


def local_order_parameter_go(
    phases: FloatArray,
    knm_flat: FloatArray,
    n: int,
) -> FloatArray:
    """Measure positive non-self adjacency through the extent-aware Go ABI.

    Parameters
    ----------
    phases : FloatArray
        Finite real radian phases, N entries; original aliases are refused.
    knm_flat : FloatArray
        Finite row-major N*N coupling. Positive off-diagonal entries are
        equally weighted neighbours; the diagonal tolerance is 1e-15.
    n : int
        Plain nonnegative count at most 2**31-1, checked before buffers.

    Returns
    -------
    FloatArray
        N finite local magnitudes in [0,1]; an empty request returns the
        identity without loading a shared library.

    Raises
    ------
    ImportError
        If the library is missing or lacks LocalOrderParameterV2.
    ValueError
        If aliases, counts, cardinalities or numerical domains are invalid.
    """
    p, k, n = validate_chimera_backend_inputs(phases, knm_flat, n, maximum_n=2**31 - 1)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    lib = _load_lib()
    out = np.zeros(n, dtype=np.float64)
    rc = lib.LocalOrderParameterV2(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(p.size),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(k.size),
        ctypes.c_int(int(n)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(out.size),
    )
    if rc != 0:
        raise ValueError(f"Go LocalOrderParameterV2 rc={rc}")
    return validate_chimera_backend_output(out, n)
