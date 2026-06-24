# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for twin-confidence divergence

"""Go backend for ``monitor/twin_confidence.py`` via ``ctypes``.

Loads ``go/libtwin_confidence.so`` lazily and exposes ``twin_divergence_go``
with the same signature as the NumPy reference kernel.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._twin_confidence_validation import (
    validate_twin_divergence_backend_inputs,
    validate_twin_divergence_backend_output,
)

__all__ = ["twin_divergence_go"]
FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libtwin_confidence.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libtwin_confidence.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libtwin_confidence.so "
            f"twin_confidence.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.TwinDivergence.restype = ctypes.c_int
    lib.TwinDivergence.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # model_phases
        ctypes.POINTER(ctypes.c_double),  # observed_phases
        ctypes.POINTER(ctypes.c_double),  # model_order
        ctypes.POINTER(ctypes.c_double),  # observed_order
        ctypes.c_int,  # n
        ctypes.c_int,  # w
        ctypes.c_int,  # n_bins
        ctypes.POINTER(ctypes.c_double),  # out (2,)
    ]
    _LIB = lib
    return lib


def twin_divergence_go(
    model_phases: FloatArray,
    observed_phases: FloatArray,
    model_order: FloatArray,
    observed_order: FloatArray,
    n: int,
    w: int,
    n_bins: int,
) -> FloatArray:
    """Compute the twin divergence pair through the Go backend.

    Parameters
    ----------
    model_phases, observed_phases : FloatArray
        Model and observed phase vectors of length ``n``.
    model_order, observed_order : FloatArray
        Model and observed order-parameter windows of length ``w``.
    n, w, n_bins : int
        Phase count, order-window length, and histogram bin count.

    Returns
    -------
    FloatArray
        Two-element ``[phase_js_divergence, order_wasserstein]`` array.

    Raises
    ------
    ValueError
        If the inputs are invalid or the Go kernel reports a contract failure.
    """
    (
        model_phases64,
        observed_phases64,
        model_order64,
        observed_order64,
        n_int,
        w_int,
        n_bins_int,
    ) = validate_twin_divergence_backend_inputs(
        model_phases,
        observed_phases,
        model_order,
        observed_order,
        n,
        w,
        n_bins,
    )
    lib = _load_lib()
    mp = np.ascontiguousarray(model_phases64, dtype=np.float64)
    op = np.ascontiguousarray(observed_phases64, dtype=np.float64)
    mo = np.ascontiguousarray(model_order64, dtype=np.float64)
    oo = np.ascontiguousarray(observed_order64, dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    rc = lib.TwinDivergence(
        mp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        op.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mo.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        oo.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_int),
        ctypes.c_int(w_int),
        ctypes.c_int(n_bins_int),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go TwinDivergence rc={rc}")
    return validate_twin_divergence_backend_output(out)
