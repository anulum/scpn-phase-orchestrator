# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for dimension kernels

"""Go backend for ``monitor/dimension.py`` via ``libdimension.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._go_runtime import load_go_library
from ._dimension_validation import (
    expected_correlation_integral_backend_output,
    expected_kaplan_yorke_backend_output,
    validate_correlation_integral_backend_inputs,
    validate_correlation_integral_backend_output,
    validate_kaplan_yorke_backend_input,
    validate_kaplan_yorke_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = ["correlation_integral_go", "kaplan_yorke_dimension_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libdimension.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Admit a loadable Go library with both required numerical exports.

    Returns
    -------
    ctypes.CDLL
        Cached library with the dimension C ABI configured.

    Raises
    ------
    ImportError
        If the artifact is absent, unusable or lacks either dimension export.
    """
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libdimension.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libdimension.so "
            f"dimension.go"
        )
    lib = load_go_library(_LIB_PATH)
    if not all(
        hasattr(lib, name) for name in ("CorrelationIntegral", "KaplanYorkeDimension")
    ):
        raise ImportError("Go dimension library lacks required CI/KY exports")
    lib.CorrelationIntegral.restype = ctypes.c_int
    lib.CorrelationIntegral.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.KaplanYorkeDimension.restype = ctypes.c_int
    lib.KaplanYorkeDimension.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def correlation_integral_go(
    traj_flat: FloatArray,
    t: int,
    d: int,
    idx_i: IntArray,
    idx_j: IntArray,
    epsilons: FloatArray,
) -> FloatArray:
    """Count selected strict-radius pairs through the Go runtime.

    Parameters
    ----------
    traj_flat : ndarray of float64
        Flat finite real trajectory with exactly ``t * d`` entries.
    t, d : int
        Nonnegative sample count and positive coordinate count.
    idx_i, idx_j : ndarray of int64
        Equal-length, in-range, off-diagonal zero-based pair indices.
    epsilons : ndarray of float64
        Finite nonnegative radii in nondecreasing order.

    Returns
    -------
    ndarray of float64
        Normalised pair counts in radius order; zero when no pairs exist.

    Raises
    ------
    ValueError
        If inputs, runtime status or numerical outputs violate the contract.
    ImportError
        If the required Go artifact cannot be admitted.
    """
    traj, t_int, d_int, ii, jj, eps = validate_correlation_integral_backend_inputs(
        traj_flat,
        t,
        d,
        idx_i,
        idx_j,
        epsilons,
    )
    if any(
        count > np.iinfo(np.int32).max for count in (t_int, d_int, ii.size, eps.size)
    ):
        raise ValueError("Go dimension counts must fit signed C int")
    lib = _load_lib()
    out = np.zeros(eps.size, dtype=np.float64)
    rc = lib.CorrelationIntegral(
        traj.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t_int),
        ctypes.c_int(d_int),
        ii.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        jj.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_int(int(ii.size)),
        eps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(eps.size)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go CorrelationIntegral rc={rc}")
    expected = expected_correlation_integral_backend_output(
        traj,
        t_int,
        d_int,
        ii,
        jj,
        eps,
    )
    return validate_correlation_integral_backend_output(out, eps, expected=expected)


def kaplan_yorke_dimension_go(lyapunov_exponents: FloatArray) -> float:
    """Evaluate the Kaplan-Yorke spectrum formula through the Go runtime.

    Parameters
    ----------
    lyapunov_exponents : ndarray of float64
        One-dimensional finite real spectrum; sorted internally without mutation.

    Returns
    -------
    float
        Dimension between zero and spectrum length, including both endpoints.

    Raises
    ------
    ValueError
        If the input, native status or result violates the spectrum contract.
    ImportError
        If the required Go artifact cannot be admitted.
    """
    if np.asarray(lyapunov_exponents).size > np.iinfo(np.int32).max:
        raise ValueError("Go spectrum length must fit signed C int")
    le = validate_kaplan_yorke_backend_input(lyapunov_exponents)
    lib = _load_lib()
    out = ctypes.c_double(0.0)
    rc = lib.KaplanYorkeDimension(
        le.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(le.size)),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go KaplanYorkeDimension rc={rc}")
    expected = expected_kaplan_yorke_backend_output(le)
    return validate_kaplan_yorke_backend_output(out.value, le, expected=expected)
