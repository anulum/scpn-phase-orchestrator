# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for envelope kernels

"""Mojo backend for ``upde/envelope.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "_ensure_exe",
    "envelope_modulation_depth_mojo",
    "extract_envelope_mojo",
]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "envelope_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/envelope.mojo "
            f"-o mojo/envelope_mojo -Xlinker -lm"
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
        raise ValueError(f"Mojo envelope exit {proc.returncode}: {proc.stderr.strip()}")
    return [line for line in proc.stdout.strip().splitlines() if line]


def extract_envelope_mojo(amps: NDArray, window: int) -> NDArray:
    a = np.ascontiguousarray(amps.ravel(), dtype=np.float64)
    if a.size == 0:
        return np.zeros(0, dtype=np.float64)
    tokens: list[str] = [
        "RMS",
        str(int(a.size)),
        str(int(window)),
    ]
    tokens.extend(repr(float(x)) for x in a.tolist())
    lines = _run(" ".join(tokens) + "\n")
    return np.array([float(x) for x in lines], dtype=np.float64)


def envelope_modulation_depth_mojo(env: NDArray) -> float:
    e = np.ascontiguousarray(env.ravel(), dtype=np.float64)
    if e.size == 0:
        return 0.0
    tokens: list[str] = ["MOD", str(int(e.size))]
    tokens.extend(repr(float(x)) for x in e.tolist())
    lines = _run(" ".join(tokens) + "\n")
    return float(lines[0])
