# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for inter-trial phase coherence

"""Mojo backend for ``monitor/itpc.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._itpc_validation import (
    validate_compute_itpc_backend_inputs,
    validate_compute_itpc_backend_output,
    validate_itpc_persistence_backend_inputs,
    validate_itpc_persistence_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = ["_ensure_exe", "compute_itpc_mojo", "itpc_persistence_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "itpc_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/itpc.mojo "
            f"-o mojo/itpc_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> list[float]:
    exe = _ensure_exe()
    proc = subprocess.run(  # nosec B603
        [str(exe)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo itpc returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    return [float(line) for line in proc.stdout.strip().splitlines() if line]


def compute_itpc_mojo(phases_flat: FloatArray, n_trials: int, n_tp: int) -> FloatArray:
    """Compute inter-trial phase coherence through the Mojo backend."""

    phases, n_trials, n_tp = validate_compute_itpc_backend_inputs(
        phases_flat,
        n_trials,
        n_tp,
    )
    if n_trials == 0 or n_tp == 0:
        return np.zeros(n_tp, dtype=np.float64)
    tokens: list[str] = ["ITPC", str(n_trials), str(n_tp)]
    tokens.extend(repr(float(x)) for x in phases.tolist())
    result = _run(" ".join(tokens) + "\n")
    return validate_compute_itpc_backend_output(result, n_tp)


def itpc_persistence_mojo(
    phases_flat: FloatArray,
    n_trials: int,
    n_tp: int,
    pause_indices: IntArray,
) -> float:
    """Compute inter-trial phase-coherence persistence through the Mojo backend."""

    phases, n_trials, n_tp, idx = validate_itpc_persistence_backend_inputs(
        phases_flat,
        n_trials,
        n_tp,
        pause_indices,
    )
    if idx.size == 0:
        return 0.0
    tokens: list[str] = [
        "PERS",
        str(n_trials),
        str(n_tp),
        str(int(idx.size)),
    ]
    tokens.extend(str(int(x)) for x in idx.tolist())
    tokens.extend(repr(float(x)) for x in phases.tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != 1:
        raise ValueError(f"Mojo ITPC persistence returned {len(result)} values")
    return validate_itpc_persistence_backend_output(result[0])
