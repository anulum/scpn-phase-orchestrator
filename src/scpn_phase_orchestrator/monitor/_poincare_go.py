# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for Poincaré kernels

"""Go backend for ``monitor/poincare.py`` via ``libpoincare.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["phase_poincare_go", "poincare_section_go"]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libpoincare.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libpoincare.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libpoincare.so "
            f"poincare.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.PoincareSection.restype = ctypes.c_int
    lib.PoincareSection.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.PhasePoincare.restype = ctypes.c_int
    lib.PhasePoincare.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def poincare_section_go(
    traj_flat: NDArray,
    t: int,
    d: int,
    normal: NDArray,
    offset: float,
    direction_id: int,
) -> tuple[NDArray, NDArray, int]:
    lib = _load_lib()
    traj = np.ascontiguousarray(traj_flat.ravel(), dtype=np.float64)
    nrm = np.ascontiguousarray(normal.ravel(), dtype=np.float64)
    crossings = np.zeros(t * d, dtype=np.float64)
    times = np.zeros(t, dtype=np.float64)
    n_cr = lib.PoincareSection(
        traj.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)),
        ctypes.c_int(int(d)),
        nrm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(float(offset)),
        ctypes.c_int(int(direction_id)),
        crossings.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return crossings, times, int(n_cr)


def phase_poincare_go(
    phases_flat: NDArray,
    t: int,
    n: int,
    oscillator_idx: int,
    section_phase: float,
) -> tuple[NDArray, NDArray, int]:
    lib = _load_lib()
    phases = np.ascontiguousarray(phases_flat.ravel(), dtype=np.float64)
    crossings = np.zeros(t * n, dtype=np.float64)
    times = np.zeros(t, dtype=np.float64)
    n_cr = lib.PhasePoincare(
        phases.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)),
        ctypes.c_int(int(n)),
        ctypes.c_int(int(oscillator_idx)),
        ctypes.c_double(float(section_phase)),
        crossings.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return crossings, times, int(n_cr)
