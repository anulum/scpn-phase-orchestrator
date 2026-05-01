# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for spectral eigendecomposition

"""Go backend for ``coupling/spectral.py`` via ``libspectral.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["spectral_eig_go"]
FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libspectral.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libspectral.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libspectral.so spectral.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.SpectralEig.restype = ctypes.c_int
    lib.SpectralEig.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def spectral_eig_go(
    knm_flat: FloatArray,
    n: int,
) -> tuple[FloatArray, FloatArray]:
    lib = _load_lib()
    k = np.ascontiguousarray(knm_flat, dtype=np.float64)
    eigvals = np.zeros(int(n), dtype=np.float64)
    fiedler = np.zeros(int(n), dtype=np.float64)
    rc = lib.SpectralEig(
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        eigvals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fiedler.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go SpectralEig rc={rc}")
    return eigvals, fiedler
