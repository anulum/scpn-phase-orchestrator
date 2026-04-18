# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for Ott-Antonsen reduction

"""Mojo backend for ``upde/reduction.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["_ensure_exe", "oa_run_mojo"]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "reduction_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/reduction.mojo -o mojo/reduction_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def oa_run_mojo(
    z_re: float,
    z_im: float,
    omega_0: float,
    delta: float,
    k_coupling: float,
    dt: float,
    n_steps: int,
) -> tuple[float, float, float, float]:
    exe = _ensure_exe()
    tokens = [
        "OARUN",
        repr(float(z_re)), repr(float(z_im)),
        repr(float(omega_0)), repr(float(delta)),
        repr(float(k_coupling)), repr(float(dt)),
        str(int(n_steps)),
    ]
    proc = subprocess.run(
        [str(exe)], input=" ".join(tokens) + "\n",
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo OARUN exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    lines = proc.stdout.strip().splitlines()
    if len(lines) != 4:
        raise ValueError(
            f"Mojo OARUN returned {len(lines)} lines, expected 4"
        )
    return (
        float(lines[0]), float(lines[1]),
        float(lines[2]), float(lines[3]),
    )
