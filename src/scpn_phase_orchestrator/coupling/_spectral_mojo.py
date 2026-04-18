# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for spectral eigendecomposition

"""Mojo backend for ``coupling/spectral.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "spectral_eig_mojo"]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "spectral_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/spectral.mojo -o mojo/spectral_mojo"
        )
    return _EXE_PATH


def spectral_eig_mojo(
    knm_flat: NDArray, n: int,
) -> tuple[NDArray, NDArray]:
    exe = _ensure_exe()
    tokens: list[str] = ["EIG", str(int(n))]
    tokens.extend(
        repr(float(x)) for x in np.asarray(knm_flat).ravel().tolist()
    )
    proc = subprocess.run(
        [str(exe)], input=" ".join(tokens) + "\n",
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo spectral exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    lines = proc.stdout.strip().splitlines()
    if lines and lines[0].startswith("ERR:"):
        raise ValueError(f"Mojo spectral LAPACK error: {lines[0]}")
    if len(lines) != 2 * n:
        raise ValueError(
            f"Mojo EIG returned {len(lines)} lines, expected {2 * n}"
        )
    eigvals = np.array([float(x) for x in lines[:n]], dtype=np.float64)
    fiedler = np.array([float(x) for x in lines[n:]], dtype=np.float64)
    return eigvals, fiedler
