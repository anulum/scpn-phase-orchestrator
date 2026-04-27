# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for entropy production rate

"""Mojo backend for ``monitor/entropy_prod.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

from numpy.typing import NDArray

__all__ = ["_ensure_exe", "entropy_production_rate_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "entropy_prod_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/entropy_prod.mojo "
            f"-o mojo/entropy_prod_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> float:
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
            f"Mojo entropy_prod returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    line = proc.stdout.strip().splitlines()[0]
    return float(line)


def entropy_production_rate_mojo(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: float,
    dt: float,
) -> float:
    n = int(phases.size)
    if n == 0 or dt <= 0.0:
        return 0.0
    tokens: list[str] = [
        "EP",
        str(n),
        repr(float(alpha)),
        repr(float(dt)),
    ]
    tokens.extend(repr(float(x)) for x in phases.ravel().tolist())
    tokens.extend(repr(float(x)) for x in omegas.ravel().tolist())
    tokens.extend(repr(float(x)) for x in knm.ravel().tolist())
    return _run(" ".join(tokens) + "\n")
