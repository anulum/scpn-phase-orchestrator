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
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "swarmalator_step_mojo"]

FloatArray: TypeAlias = NDArray[np.float64]
TWO_PI = 2.0 * np.pi

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "swarmalator_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/swarmalator.mojo "
            f"-o mojo/swarmalator_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def swarmalator_step_mojo(
    pos: FloatArray,
    phases: FloatArray,
    omegas: FloatArray,
    n: int,
    dim: int,
    a: float,
    b: float,
    j: float,
    k: float,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Advance one swarmalator position-phase step.

    The calculation is delegated to the Mojo backend.
    """

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
    proc = subprocess.run(  # nosec B603
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
    lines = proc.stdout.splitlines()
    expected = n * dim + n
    if len(lines) != expected:
        raise ValueError(f"Mojo STEP returned {len(lines)} lines, expected {expected}")
    values: list[float] = []
    phase_offset = n * dim
    for index, line in enumerate(lines):
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                "Mojo STEP output must contain finite positions followed by "
                "finite phases in [0, 2*pi)"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                "Mojo STEP output must contain finite positions followed by "
                "finite phases in [0, 2*pi)"
            )
        if index >= phase_offset and (value < 0.0 or value >= TWO_PI):
            raise ValueError(
                "Mojo STEP output must contain finite positions followed by "
                "finite phases in [0, 2*pi)"
            )
        values.append(value)
    new_pos: FloatArray = np.array(values[:phase_offset], dtype=np.float64)
    new_phases: FloatArray = np.array(values[phase_offset:], dtype=np.float64)
    return new_pos.reshape(n, dim), new_phases
