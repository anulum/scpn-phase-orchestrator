# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for spatial coupling modulation

"""Go backend for ``coupling/spatial_modulator.py`` via ``libspatial_modulator.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._spatial_modulator_validation import (
    validate_spatial_modulator_inputs,
    validate_spatial_modulator_output,
)

__all__ = ["_load_lib", "spatial_modulate_go"]
FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libspatial_modulator.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libspatial_modulator.so not found at {_LIB_PATH}. Build with: "
            "cd go && go build -buildmode=c-shared -o libspatial_modulator.so "
            "spatial_modulator.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.SpatialModulate.restype = ctypes.c_int
    lib.SpatialModulate.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def spatial_modulate_go(
    k_nm_flat: FloatArray,
    positions_flat: FloatArray,
    n: int,
    dim: int,
    k_base: float,
    decay_form_code: int,
    decay_exponent: float,
    decay_length_scale: float,
    epsilon: float,
) -> FloatArray:
    """Compute a spatially modulated coupling matrix with the Go backend."""

    k, p, n, dim, k_base, form, exponent, length, eps = (
        validate_spatial_modulator_inputs(
            k_nm_flat,
            positions_flat,
            n,
            dim,
            k_base,
            decay_form_code,
            decay_exponent,
            decay_length_scale,
            epsilon,
        )
    )
    lib = _load_lib()
    out = np.zeros(n * n, dtype=np.float64)
    rc = lib.SpatialModulate(
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_int(dim),
        ctypes.c_double(k_base),
        ctypes.c_int(form),
        ctypes.c_double(exponent),
        ctypes.c_double(length),
        ctypes.c_double(eps),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go SpatialModulate rc={rc}")
    return validate_spatial_modulator_output(out, n=n)
