# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for Lyapunov spectrum

"""Go backend for ``monitor/lyapunov.py`` via ``ctypes``.

Loads ``go/liblyapunov.so`` lazily and exposes
``lyapunov_spectrum_go`` with the same signature as the Python
reference implementation.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["lyapunov_spectrum_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "liblyapunov.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"liblyapunov.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o liblyapunov.so "
            f"lyapunov.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.LyapunovSpectrum.restype = ctypes.c_int
    lib.LyapunovSpectrum.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # phases_init
        ctypes.POINTER(ctypes.c_double),  # omegas
        ctypes.POINTER(ctypes.c_double),  # knm_flat
        ctypes.POINTER(ctypes.c_double),  # alpha_flat
        ctypes.c_int,                     # n
        ctypes.c_double,                  # dt
        ctypes.c_int,                     # n_steps
        ctypes.c_int,                     # qr_interval
        ctypes.c_double,                  # zeta
        ctypes.c_double,                  # psi
        ctypes.POINTER(ctypes.c_double),  # out
    ]
    _LIB = lib
    return lib


def lyapunov_spectrum_go(
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    dt: float,
    n_steps: int,
    qr_interval: int,
    zeta: float,
    psi: float,
) -> NDArray:
    lib = _load_lib()
    n = int(phases_init.size)
    p = np.ascontiguousarray(phases_init.ravel(), dtype=np.float64)
    o = np.ascontiguousarray(omegas.ravel(), dtype=np.float64)
    k = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    rc = lib.LyapunovSpectrum(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_double(float(dt)),
        ctypes.c_int(int(n_steps)),
        ctypes.c_int(int(qr_interval)),
        ctypes.c_double(float(zeta)),
        ctypes.c_double(float(psi)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go LyapunovSpectrum rc={rc}")
    return out
