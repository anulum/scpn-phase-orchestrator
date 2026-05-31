# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for embedding primitives

"""Mojo backend for ``monitor/embedding.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._embedding_validation import (
    validate_delay_embed_backend_inputs,
    validate_mutual_information_backend_inputs,
    validate_nearest_neighbor_backend_inputs,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "_ensure_exe",
    "delay_embed_mojo",
    "mutual_information_mojo",
    "nearest_neighbor_distances_mojo",
]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "embedding_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/embedding.mojo "
            f"-o mojo/embedding_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> list[str]:
    exe = _ensure_exe()
    proc = subprocess.run(  # nosec B603
        [str(exe)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo embedding returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    return [line for line in proc.stdout.strip().splitlines() if line]


def delay_embed_mojo(
    signal: FloatArray,
    delay: int,
    dimension: int,
) -> FloatArray:
    """Build a delay-coordinate embedding through the Mojo backend."""

    s, delay_int, dimension_int, _ = validate_delay_embed_backend_inputs(
        signal,
        delay,
        dimension,
    )
    t = int(s.size)
    tokens: list[str] = [
        "DE",
        str(t),
        str(delay_int),
        str(dimension_int),
    ]
    tokens.extend(repr(float(x)) for x in s.tolist())
    result = _run(" ".join(tokens) + "\n")
    return np.array([float(line) for line in result], dtype=np.float64)


def mutual_information_mojo(
    signal: FloatArray,
    lag: int,
    n_bins: int,
) -> float:
    """Compute mutual information for embedded phase samples.

    The calculation is delegated to the Mojo backend.
    """

    s, lag_int, bins_int = validate_mutual_information_backend_inputs(
        signal,
        lag,
        n_bins,
    )
    if s.size - lag_int <= 0:
        return 0.0
    t = int(s.size)
    tokens: list[str] = [
        "MI",
        str(t),
        str(lag_int),
        str(bins_int),
    ]
    tokens.extend(repr(float(x)) for x in s.tolist())
    result = _run(" ".join(tokens) + "\n")
    return float(result[0])


def nearest_neighbor_distances_mojo(
    embedded: FloatArray,
    t: int,
    m: int,
) -> tuple[FloatArray, IntArray]:
    """Compute nearest-neighbour distances for embedded states.

    The calculation is delegated to the Mojo backend.
    """

    e, t_int, m_int = validate_nearest_neighbor_backend_inputs(embedded, t, m)
    if t_int == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    tokens: list[str] = ["NN", str(t_int), str(m_int)]
    tokens.extend(repr(float(x)) for x in e.tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != 2 * t_int:
        raise ValueError(f"Mojo NN returned {len(result)} lines, expected {2 * t_int}")
    dist = np.array([float(x) for x in result[:t_int]], dtype=np.float64)
    idx = np.array([int(x) for x in result[t_int:]], dtype=np.int64)
    return dist, idx
