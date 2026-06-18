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
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._spectral_validation import validate_spectral_backend_inputs

__all__ = ["_ensure_exe", "spectral_eig_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "spectral_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/spectral.mojo -o mojo/spectral_mojo"
        )
    return _EXE_PATH


def spectral_eig_mojo(
    knm_flat: FloatArray,
    n: int,
) -> tuple[FloatArray, FloatArray]:
    """Compute coupling-spectrum eigenvalues and Fiedler vector via Mojo."""
    k, n = validate_spectral_backend_inputs(knm_flat, n)
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy()
    exe = _ensure_exe()
    tokens: list[str] = ["EIG", str(int(n))]
    tokens.extend(repr(float(x)) for x in k.tolist())
    proc = subprocess.run(  # nosec B603
        [str(exe)],
        input=" ".join(tokens) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(f"Mojo spectral exit {proc.returncode}: {proc.stderr.strip()}")
    lines = proc.stdout.splitlines()
    if lines and lines[0].startswith("ERR:"):
        raise ValueError(f"Mojo spectral LAPACK error: {lines[0]}")
    if len(lines) != 2 * n:
        raise ValueError(f"Mojo EIG returned {len(lines)} lines, expected {2 * n}")
    try:
        values = [float(line) for line in lines]
    except ValueError as exc:
        raise ValueError(
            "Mojo EIG output must contain finite eigenvalues followed by "
            "a finite Fiedler vector"
        ) from exc
    parsed = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(parsed)):
        raise ValueError(
            "Mojo EIG output must contain finite eigenvalues followed by "
            "a finite Fiedler vector"
        )
    eigvals = parsed[:n].copy()
    fiedler = parsed[n:].copy()
    return eigvals, fiedler
