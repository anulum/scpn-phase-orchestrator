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

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "_ensure_exe",
    "delay_embed_mojo",
    "mutual_information_mojo",
    "nearest_neighbor_distances_mojo",
]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "embedding_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/embedding.mojo "
            f"-o mojo/embedding_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> list[str]:
    exe = _ensure_exe()
    proc = subprocess.run(
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
    signal: NDArray,
    delay: int,
    dimension: int,
) -> NDArray:
    s = np.ascontiguousarray(signal.ravel(), dtype=np.float64)
    t = int(s.size)
    tokens: list[str] = [
        "DE",
        str(t),
        str(int(delay)),
        str(int(dimension)),
    ]
    tokens.extend(repr(float(x)) for x in s.tolist())
    result = _run(" ".join(tokens) + "\n")
    return np.array([float(line) for line in result], dtype=np.float64)


def mutual_information_mojo(
    signal: NDArray,
    lag: int,
    n_bins: int,
) -> float:
    s = np.ascontiguousarray(signal.ravel(), dtype=np.float64)
    t = int(s.size)
    tokens: list[str] = [
        "MI",
        str(t),
        str(int(lag)),
        str(int(n_bins)),
    ]
    tokens.extend(repr(float(x)) for x in s.tolist())
    result = _run(" ".join(tokens) + "\n")
    return float(result[0])


def nearest_neighbor_distances_mojo(
    embedded: NDArray,
    t: int,
    m: int,
) -> tuple[NDArray, NDArray]:
    e = np.ascontiguousarray(embedded.ravel(), dtype=np.float64)
    tokens: list[str] = ["NN", str(int(t)), str(int(m))]
    tokens.extend(repr(float(x)) for x in e.tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != 2 * t:
        raise ValueError(f"Mojo NN returned {len(result)} lines, expected {2 * t}")
    dist = np.array([float(x) for x in result[:t]], dtype=np.float64)
    idx = np.array([int(x) for x in result[t:]], dtype=np.int64)
    return dist, idx
