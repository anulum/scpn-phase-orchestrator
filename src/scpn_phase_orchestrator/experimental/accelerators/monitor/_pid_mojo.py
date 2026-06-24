# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for partial information decomposition

"""Mojo backend for ``monitor/pid.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._pid_validation import (
    validate_pid_backend_inputs,
    validate_pid_scalar_output,
)

__all__ = ["_ensure_exe", "pid_decomposition_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "pid_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/pid.mojo "
            f"-o mojo/pid_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def pid_decomposition_mojo(
    phase_history_flat: FloatArray,
    t: int,
    n: int,
    group_a: IntArray,
    group_b: IntArray,
    n_bins: int,
) -> tuple[float, float]:
    """Compute (redundancy, synergy) via the Mojo executable."""
    history, t, n, group_a_idx, group_b_idx, bins = validate_pid_backend_inputs(
        phase_history_flat, t, n, group_a, group_b, n_bins
    )
    exe = _ensure_exe()
    tokens: list[str] = [
        "PID",
        str(int(t)),
        str(int(n)),
        str(int(bins)),
        str(int(group_a_idx.size)),
        str(int(group_b_idx.size)),
    ]
    tokens.extend(repr(float(x)) for x in history.tolist())
    tokens.extend(str(int(x)) for x in group_a_idx.tolist())
    tokens.extend(str(int(x)) for x in group_b_idx.tolist())
    proc = subprocess.run(  # nosec B603
        [str(exe)],
        input=" ".join(tokens) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(f"Mojo pid exit {proc.returncode}: {proc.stderr.strip()}")
    lines = proc.stdout.splitlines()
    if len(lines) != 2:
        raise ValueError(f"Mojo PID returned {len(lines)} lines, expected 2")
    try:
        red = float(lines[0])
        syn = float(lines[1])
    except ValueError as exc:
        raise ValueError(
            "Mojo PID output must contain finite redundancy/synergy"
        ) from exc
    return (
        validate_pid_scalar_output(red, name="redundancy"),
        validate_pid_scalar_output(syn, name="synergy"),
    )
