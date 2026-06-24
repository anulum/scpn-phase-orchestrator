# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for embedding primitives

"""Go backend for ``monitor/embedding.py`` via ``libembedding.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._embedding_validation import (
    validate_delay_embed_backend_inputs,
    validate_delay_embed_backend_output,
    validate_mutual_information_backend_inputs,
    validate_mutual_information_backend_output,
    validate_nearest_neighbor_backend_inputs,
    validate_nearest_neighbor_backend_outputs,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "delay_embed_go",
    "mutual_information_go",
    "nearest_neighbor_distances_go",
]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libembedding.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libembedding.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libembedding.so "
            f"embedding.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.DelayEmbed.restype = ctypes.c_int
    lib.DelayEmbed.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.MutualInformation.restype = ctypes.c_int
    lib.MutualInformation.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.NearestNeighborDistances.restype = ctypes.c_int
    lib.NearestNeighborDistances.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_longlong),
    ]
    _LIB = lib
    return lib


def delay_embed_go(
    signal: FloatArray,
    delay: int,
    dimension: int,
) -> FloatArray:
    """Build a delay-coordinate embedding through the Go backend."""
    s, delay_int, dimension_int, t_eff = validate_delay_embed_backend_inputs(
        signal,
        delay,
        dimension,
    )
    lib = _load_lib()
    out = np.zeros(t_eff * dimension_int, dtype=np.float64)
    rc = lib.DelayEmbed(
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(s.size)),
        ctypes.c_int(delay_int),
        ctypes.c_int(dimension_int),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go DelayEmbed rc={rc}")
    return validate_delay_embed_backend_output(
        out,
        signal=s,
        delay=delay_int,
        dimension=dimension_int,
        t_effective=t_eff,
    )


def mutual_information_go(
    signal: FloatArray,
    lag: int,
    n_bins: int,
) -> float:
    """Compute mutual information for embedded phase samples through the Go backend."""
    s, lag_int, bins_int = validate_mutual_information_backend_inputs(
        signal,
        lag,
        n_bins,
    )
    if s.size - lag_int <= 0:
        return 0.0
    lib = _load_lib()
    out = ctypes.c_double(0.0)
    rc = lib.MutualInformation(
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(s.size)),
        ctypes.c_int(lag_int),
        ctypes.c_int(bins_int),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go MutualInformation rc={rc}")
    return validate_mutual_information_backend_output(out.value)


def nearest_neighbor_distances_go(
    embedded: FloatArray,
    t: int,
    m: int,
) -> tuple[FloatArray, IntArray]:
    """Compute nearest-neighbour distances for embedded states.

    The calculation is delegated to the Go backend.
    """
    e, t_int, m_int = validate_nearest_neighbor_backend_inputs(embedded, t, m)
    if t_int == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    lib = _load_lib()
    dist = np.zeros(t_int, dtype=np.float64)
    idx = np.zeros(t_int, dtype=np.int64)
    rc = lib.NearestNeighborDistances(
        e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t_int),
        ctypes.c_int(m_int),
        dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
    )
    if rc != 0:
        raise ValueError(f"Go NearestNeighborDistances rc={rc}")
    return validate_nearest_neighbor_backend_outputs(dist, idx, t=t_int)
