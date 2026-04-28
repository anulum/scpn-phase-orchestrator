# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for swarmalator stepper

"""Mojo backend for ``upde/swarmalator.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "swarmalator_step_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "swarmalator_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/swarmalator.mojo "
            f"-o mojo/swarmalator_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def swarmalator_step_mojo(
    pos: NDArray,
    phases: NDArray,
    omegas: NDArray,
    n: int,
    dim: int,
    a: float,
    b: float,
    j: float,
    k: float,
    dt: float,
) -> tuple[NDArray, NDArray]:
    exe = _ensure_exe()
    tokens: list[str] = [
        "STEP",
        str(int(n)),
        str(int(dim)),
        repr(float(a)),
        repr(float(b)),
        repr(float(j)),
        repr(float(k)),
        repr(float(dt)),
    ]
    tokens.extend(repr(float(x)) for x in pos.ravel().tolist())
    tokens.extend(repr(float(x)) for x in phases.ravel().tolist())
    tokens.extend(repr(float(x)) for x in omegas.ravel().tolist())
    proc = subprocess.run(
        [str(exe)],
        input=" ".join(tokens) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo swarmalator exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.strip().splitlines()
    expected = n * dim + n
    if len(lines) != expected:
        raise ValueError(f"Mojo STEP returned {len(lines)} lines, expected {expected}")
    new_pos = np.array(
        [float(x) for x in lines[: n * dim]],
        dtype=np.float64,
    )
    new_phases = np.array(
        [float(x) for x in lines[n * dim :]],
        dtype=np.float64,
    )
    return new_pos.reshape(n, dim), new_phases
