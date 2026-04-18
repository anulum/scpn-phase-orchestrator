# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for winding numbers

"""Mojo backend for ``monitor/winding.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "winding_numbers_mojo"]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "winding_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/winding.mojo "
            f"-o mojo/winding_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> list[int]:
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
            f"Mojo winding returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return [
        int(line)
        for line in proc.stdout.strip().splitlines()
        if line
    ]


def winding_numbers_mojo(
    phases_flat: NDArray, t: int, n: int,
) -> NDArray:
    if n == 0 or t < 2:
        return np.zeros(n, dtype=np.int64)
    tokens: list[str] = ["WIND", str(int(t)), str(int(n))]
    tokens.extend(repr(float(x)) for x in phases_flat.ravel().tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != n:
        raise ValueError(
            f"Mojo WIND returned {len(result)} values, expected {n}"
        )
    return np.array(result, dtype=np.int64)
