# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for Strang splitting

"""Mojo backend for ``upde/splitting.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "splitting_run_mojo"]

FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "splitting_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/splitting.mojo -o mojo/splitting_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def splitting_run_mojo(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    exe = _ensure_exe()
    tokens: list[str] = [
        "SPLIT",
        str(int(n)),
        repr(float(zeta)),
        repr(float(psi)),
        repr(float(dt)),
        str(int(n_steps)),
    ]
    tokens.extend(repr(float(x)) for x in np.asarray(phases).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(omegas).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(knm_flat).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(alpha_flat).ravel().tolist())
    proc = subprocess.run(
        [str(exe)],
        input=" ".join(tokens) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo splitting exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.strip().splitlines()
    if len(lines) != n:
        raise ValueError(f"Mojo SPLIT returned {len(lines)} lines, expected {n}")
    result: FloatArray = np.array([float(x) for x in lines], dtype=np.float64)
    return result
