# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for recurrence kernels

"""Mojo backend for ``monitor/recurrence.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._mojo_runtime import require_mojo_executable, run_mojo_executable
from ._recurrence_validation import (
    expected_recurrence_backend_output,
    validate_cross_recurrence_backend_inputs,
    validate_recurrence_backend_inputs,
    validate_recurrence_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
ByteArray: TypeAlias = NDArray[np.uint8]

__all__ = [
    "_ensure_exe",
    "cross_recurrence_matrix_mojo",
    "recurrence_matrix_mojo",
]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "recurrence_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/recurrence.mojo "
            f"-o mojo/recurrence_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def _run(payload: str, *, expected_count: int, label: str) -> list[int]:
    """Call the backend kernel with the prepared inputs and return its result."""
    exe = _ensure_exe()
    proc = run_mojo_executable(exe, payload, runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo recurrence returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"Mojo {label} must emit exactly {expected_count} integer line(s), "
            f"got {len(lines)}"
        )
    values: list[int] = []
    for line in lines:
        try:
            values.append(int(line))
        except ValueError as exc:
            raise ValueError(
                f"Mojo {label} emitted a non-integer recurrence value: {line!r}"
            ) from exc
    return values


def recurrence_matrix_mojo(
    traj_flat: FloatArray,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> ByteArray:
    """Compute the recurrence matrix through the Mojo backend."""
    p, t_int, d_int, radius, angular_flag = validate_recurrence_backend_inputs(
        traj_flat,
        t,
        d,
        epsilon,
        angular,
    )
    tokens: list[str] = [
        "REC",
        str(t_int),
        str(d_int),
        str(int(angular_flag)),
        repr(radius),
    ]
    tokens.extend(repr(float(x)) for x in p.tolist())
    result = _run(
        " ".join(tokens) + "\n",
        expected_count=t_int * t_int,
        label="REC",
    )
    return validate_recurrence_backend_output(
        result,
        t=t_int,
        name="recurrence_matrix",
        expected=expected_recurrence_backend_output(
            p,
            p,
            t=t_int,
            d=d_int,
            epsilon=radius,
            angular=angular_flag,
        ),
    )


def cross_recurrence_matrix_mojo(
    traj_a_flat: FloatArray,
    traj_b_flat: FloatArray,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> ByteArray:
    """Compute the cross-recurrence matrix through the Mojo backend."""
    (
        a,
        b,
        t_int,
        d_int,
        radius,
        angular_flag,
    ) = validate_cross_recurrence_backend_inputs(
        traj_a_flat,
        traj_b_flat,
        t,
        d,
        epsilon,
        angular,
    )
    tokens: list[str] = [
        "CROSS",
        str(t_int),
        str(d_int),
        str(int(angular_flag)),
        repr(radius),
    ]
    tokens.extend(repr(float(x)) for x in a.tolist())
    tokens.extend(repr(float(x)) for x in b.tolist())
    result = _run(
        " ".join(tokens) + "\n",
        expected_count=t_int * t_int,
        label="CROSS",
    )
    return validate_recurrence_backend_output(
        result,
        t=t_int,
        name="cross_recurrence_matrix",
        expected=expected_recurrence_backend_output(
            a,
            b,
            t=t_int,
            d=d_int,
            epsilon=radius,
            angular=angular_flag,
        ),
    )
