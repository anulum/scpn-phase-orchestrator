# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for steady-state R

"""Mojo backend for ``upde/basin_stability.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "steady_state_r_mojo"]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "basin_stability_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/basin_stability.mojo -o mojo/basin_stability_mojo "
            f"-Xlinker -lm"
        )
    return _EXE_PATH


def steady_state_r_mojo(
    phases_init: NDArray,
    omegas: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    n: int,
    k_scale: float,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> float:
    exe = _ensure_exe()
    tokens: list[str] = [
        "STEADY", str(int(n)),
        repr(float(k_scale)), repr(float(dt)),
        str(int(n_transient)), str(int(n_measure)),
    ]
    tokens.extend(
        repr(float(x)) for x in np.asarray(phases_init).ravel().tolist()
    )
    tokens.extend(
        repr(float(x)) for x in np.asarray(omegas).ravel().tolist()
    )
    tokens.extend(
        repr(float(x)) for x in np.asarray(knm_flat).ravel().tolist()
    )
    tokens.extend(
        repr(float(x)) for x in np.asarray(alpha_flat).ravel().tolist()
    )
    proc = subprocess.run(
        [str(exe)], input=" ".join(tokens) + "\n",
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo basin_stability exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    line = proc.stdout.strip()
    if not line:
        raise ValueError("Mojo STEADY returned empty output")
    return float(line)
