# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for OPT-entropy

"""Go backend for ``monitor/opt_entropy.py``. Loads ``libopt_entropy.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._go_runtime import load_go_library
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

__all__ = ["ordinal_pattern_sequence_go", "transition_entropy_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libopt_entropy.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libopt_entropy.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libopt_entropy.so opt_entropy.go"
        )
    lib = load_go_library(_LIB_PATH)
    lib.OrdinalPatternSequence.restype = ctypes.c_int
    lib.OrdinalPatternSequence.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
    ]
    lib.TransitionEntropy.restype = ctypes.c_int
    lib.TransitionEntropy.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def ordinal_pattern_sequence_go(
    series: FloatArray,
    dimension: int,
    delay: int,
) -> IntArray:
    """Compute the ordinal-pattern code sequence through the Go backend.

    Parameters
    ----------
    series : FloatArray
        Finite real one-dimensional samples. Numeric object storage is allowed;
        Boolean, text, complex and temporal samples are refused before conversion.
    dimension : int
        Plain integer embedding dimension in [2, 7].
    delay : int
        Plain positive integer delay. An embedding exceeding the series length
        returns empty evidence before conversion to native parameter storage.

    Returns
    -------
    IntArray
        Stable Lehmer codes; empty when no embedding window fits.

    Raises
    ------
    ValueError
        If source types, dimensions, finite samples or backend outputs violate
        the ordinal embedding contract.
    """
    s, d, tau = validate_transition_entropy_backend_inputs(series, dimension, delay)
    count = ordinal_window_count(int(s.size), d, tau)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    lib = _load_lib()
    out = np.zeros(max(count, 1), dtype=np.int64)
    rc = lib.OrdinalPatternSequence(
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(s.size)),
        ctypes.c_int(d),
        ctypes.c_int(tau),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
    )
    if rc != 0:
        raise ValueError(f"Go OrdinalPatternSequence rc={rc}")
    return validate_ordinal_pattern_backend_output(
        out[:count],
        n_windows=count,
        dimension=d,
        expected=expected_ordinal_pattern_backend_output(s, d, tau),
    )


def transition_entropy_go(
    series: FloatArray,
    dimension: int,
    delay: int,
) -> float:
    """Compute the normalised transition entropy through the Go backend.

    Parameters
    ----------
    series : FloatArray
        Finite real one-dimensional samples. Numeric object storage is allowed;
        Boolean, text, complex and temporal samples are refused before conversion.
    dimension : int
        Plain integer embedding dimension in [2, 7].
    delay : int
        Plain positive integer delay. An embedding exceeding the series length
        returns empty evidence before conversion to native parameter storage.

    Returns
    -------
    float
        Normalised entropy; zero when fewer than two patterns fit.

    Raises
    ------
    ValueError
        If source types, dimensions, finite samples or backend outputs violate
        the ordinal embedding contract.
    """
    s, d, tau = validate_transition_entropy_backend_inputs(series, dimension, delay)
    if ordinal_window_count(int(s.size), d, tau) < 2:
        return 0.0
    lib = _load_lib()
    out = ctypes.c_double(0.0)
    rc = lib.TransitionEntropy(
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(s.size)),
        ctypes.c_int(d),
        ctypes.c_int(tau),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go TransitionEntropy rc={rc}")
    return validate_transition_entropy_backend_output(
        out.value,
        expected=expected_transition_entropy_backend_output(s, d, tau),
    )
