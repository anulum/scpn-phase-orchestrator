# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for Koopman EDMD

"""Mojo backend for ``monitor/koopman_edmd.py``. Text-stdin subprocess bridge."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._koopman_edmd_validation import (
    validate_edmd_backend_inputs,
    validate_edmd_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["_ensure_exe", "koopman_edmd_solve_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "koopman_edmd_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/koopman_edmd.mojo "
            f"-o mojo/koopman_edmd_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _serialise(matrix: FloatArray) -> str:
    return " ".join(repr(float(value)) for value in matrix.ravel().tolist())


def _run(payload: str, *, expected_count: int) -> list[float]:
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
            f"Mojo koopman_edmd returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"Mojo koopman_edmd must emit {expected_count} scalar line(s), "
            f"got {len(lines)}"
        )
    values: list[float] = []
    for line in lines:
        try:
            values.append(float(line))
        except ValueError as exc:
            raise ValueError(
                f"Mojo koopman_edmd emitted a non-scalar value: {line!r}"
            ) from exc
    return values


def koopman_edmd_solve_mojo(
    x_lift: FloatArray,
    inputs: FloatArray,
    y_lift: FloatArray,
    states: FloatArray,
    regularisation: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Solve the EDMD-with-control least squares through the Mojo backend."""
    dims = validate_edmd_backend_inputs(x_lift, inputs, y_lift, states)
    n_lift, m, n_state = dims.lift_dim, dims.input_dim, dims.state_dim
    header = f"EDMD {dims.samples} {n_lift} {m} {n_state} {regularisation!r}"
    payload = (
        f"{header} {_serialise(x_lift)} {_serialise(inputs)} "
        f"{_serialise(y_lift)} {_serialise(states)}\n"
    )
    a_size = n_lift * n_lift
    b_size = n_lift * m
    c_size = n_state * n_lift
    flat = np.asarray(
        _run(payload, expected_count=a_size + b_size + c_size), dtype=np.float64
    )
    return validate_edmd_backend_output(
        flat[:a_size].reshape(n_lift, n_lift),
        flat[a_size : a_size + b_size].reshape(n_lift, m),
        flat[a_size + b_size :].reshape(n_state, n_lift),
        dims,
    )
