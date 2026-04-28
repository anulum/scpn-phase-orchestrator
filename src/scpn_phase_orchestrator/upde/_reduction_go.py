# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for Ott-Antonsen reduction

"""Go backend for ``upde/reduction.py`` via ``libreduction.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

__all__ = ["oa_run_go"]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libreduction.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libreduction.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libreduction.so reduction.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.OARun.restype = ctypes.c_int
    lib.OARun.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def oa_run_go(
    z_re: float,
    z_im: float,
    omega_0: float,
    delta: float,
    k_coupling: float,
    dt: float,
    n_steps: int,
) -> tuple[float, float, float, float]:
    lib = _load_lib()
    re = ctypes.c_double(0.0)
    im = ctypes.c_double(0.0)
    r = ctypes.c_double(0.0)
    psi = ctypes.c_double(0.0)
    rc = lib.OARun(
        ctypes.c_double(float(z_re)),
        ctypes.c_double(float(z_im)),
        ctypes.c_double(float(omega_0)),
        ctypes.c_double(float(delta)),
        ctypes.c_double(float(k_coupling)),
        ctypes.c_double(float(dt)),
        ctypes.c_int(int(n_steps)),
        ctypes.byref(re),
        ctypes.byref(im),
        ctypes.byref(r),
        ctypes.byref(psi),
    )
    if rc != 0:
        raise ValueError(f"Go OARun rc={rc}")
    return (re.value, im.value, r.value, psi.value)
