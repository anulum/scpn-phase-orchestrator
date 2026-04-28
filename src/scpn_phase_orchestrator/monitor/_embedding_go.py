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

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "delay_embed_go",
    "mutual_information_go",
    "nearest_neighbor_distances_go",
]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libembedding.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
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
    signal: NDArray,
    delay: int,
    dimension: int,
) -> NDArray:
    lib = _load_lib()
    s = np.ascontiguousarray(signal.ravel(), dtype=np.float64)
    t_eff = int(s.size) - (int(dimension) - 1) * int(delay)
    out = np.zeros(t_eff * int(dimension), dtype=np.float64)
    rc = lib.DelayEmbed(
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(s.size)),
        ctypes.c_int(int(delay)),
        ctypes.c_int(int(dimension)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go DelayEmbed rc={rc}")
    return out


def mutual_information_go(
    signal: NDArray,
    lag: int,
    n_bins: int,
) -> float:
    lib = _load_lib()
    s = np.ascontiguousarray(signal.ravel(), dtype=np.float64)
    out = ctypes.c_double(0.0)
    rc = lib.MutualInformation(
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(s.size)),
        ctypes.c_int(int(lag)),
        ctypes.c_int(int(n_bins)),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go MutualInformation rc={rc}")
    return float(out.value)


def nearest_neighbor_distances_go(
    embedded: NDArray,
    t: int,
    m: int,
) -> tuple[NDArray, NDArray]:
    lib = _load_lib()
    e = np.ascontiguousarray(embedded.ravel(), dtype=np.float64)
    dist = np.zeros(t, dtype=np.float64)
    idx = np.zeros(t, dtype=np.int64)
    rc = lib.NearestNeighborDistances(
        e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)),
        ctypes.c_int(int(m)),
        dist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
    )
    if rc != 0:
        raise ValueError(f"Go NearestNeighborDistances rc={rc}")
    return dist, idx
