# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for chimera local-R kernel

"""Mojo backend for ``monitor/chimera.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["_ensure_exe", "local_order_parameter_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "chimera_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/chimera.mojo "
            f"-o mojo/chimera_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> list[float]:
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
            f"Mojo chimera returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    return [float(line) for line in proc.stdout.strip().splitlines() if line]


def local_order_parameter_mojo(
    phases: FloatArray,
    knm_flat: FloatArray,
    n: int,
) -> FloatArray:
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    tokens: list[str] = ["CHI", str(int(n))]
    tokens.extend(repr(float(x)) for x in phases.ravel().tolist())
    tokens.extend(repr(float(x)) for x in knm_flat.ravel().tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != n:
        raise ValueError(f"Mojo CHI returned {len(result)} values, expected {n}")
    return np.array(result, dtype=np.float64)
