# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for UPDE engine

"""Mojo backend for ``upde/engine.py`` via ``mojo/upde_engine_mojo``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "upde_run_mojo"]

_METHOD_IDS = {"euler": 0, "rk4": 1, "rk45": 2}

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "upde_engine_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/upde_engine.mojo "
            f"-o mojo/upde_engine_mojo -Xlinker -lm"
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
            f"Mojo upde_engine returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    return [float(line) for line in proc.stdout.strip().splitlines() if line]


def upde_run_mojo(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> NDArray:
    if method not in _METHOD_IDS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {list(_METHOD_IDS)}"
        )
    n = int(phases.size)
    tokens: list[str] = [
        "RUN",
        str(n),
        repr(float(zeta)),
        repr(float(psi)),
        repr(float(dt)),
        str(int(n_steps)),
        str(int(_METHOD_IDS[method])),
        str(int(n_substeps)),
        repr(float(atol)),
        repr(float(rtol)),
    ]
    tokens.extend(repr(float(x)) for x in phases.ravel().tolist())
    tokens.extend(repr(float(x)) for x in omegas.ravel().tolist())
    tokens.extend(repr(float(x)) for x in knm.ravel().tolist())
    tokens.extend(repr(float(x)) for x in alpha.ravel().tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != n:
        raise ValueError(f"Mojo RUN returned {len(result)} values, expected {n}")
    return np.array(result, dtype=np.float64)
