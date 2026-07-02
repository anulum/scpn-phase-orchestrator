# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for OPT-entropy

"""Mojo backend for ``monitor/opt_entropy.py``. Text-stdin subprocess bridge."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._mojo_runtime import require_mojo_executable, run_mojo_executable
from ._opt_entropy_validation import (
    expected_ordinal_pattern_backend_output,
    expected_transition_entropy_backend_output,
    ordinal_window_count,
    validate_ordinal_pattern_backend_output,
    validate_transition_entropy_backend_inputs,
    validate_transition_entropy_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = ["ordinal_pattern_sequence_mojo", "transition_entropy_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "opt_entropy_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/opt_entropy.mojo "
            f"-o mojo/opt_entropy_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def _run(payload: str, *, expected_count: int, label: str) -> list[float]:
    """Call the backend kernel with the prepared inputs and return its result."""
    exe = _ensure_exe()
    proc = run_mojo_executable(exe, payload, runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo opt_entropy returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"Mojo {label} must emit exactly {expected_count} scalar line(s), "
            f"got {len(lines)}"
        )
    values: list[float] = []
    for line in lines:
        try:
            values.append(float(line))
        except ValueError as exc:
            raise ValueError(
                f"Mojo {label} emitted a non-scalar value: {line!r}"
            ) from exc
    return values


def _serialise(series: FloatArray) -> str:
    """Serialise the inputs into the backend wire format."""
    return " ".join(repr(float(x)) for x in series.tolist())


def ordinal_pattern_sequence_mojo(
    series: FloatArray,
    dimension: int,
    delay: int,
) -> IntArray:
    """Compute the ordinal-pattern code sequence through the Mojo backend."""
    s, d, tau = validate_transition_entropy_backend_inputs(series, dimension, delay)
    n = int(s.size)
    count = ordinal_window_count(n, d, tau)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    payload = f"OPS {n} {d} {tau} {_serialise(s)}\n"
    result = _run(payload, expected_count=count, label="OPS")
    codes = np.rint(np.asarray(result, dtype=np.float64)).astype(np.int64)
    return validate_ordinal_pattern_backend_output(
        codes,
        n_windows=count,
        dimension=d,
        expected=expected_ordinal_pattern_backend_output(s, d, tau),
    )


def transition_entropy_mojo(
    series: FloatArray,
    dimension: int,
    delay: int,
) -> float:
    """Compute the normalised transition entropy through the Mojo backend."""
    s, d, tau = validate_transition_entropy_backend_inputs(series, dimension, delay)
    n = int(s.size)
    payload = f"OTE {n} {d} {tau} {_serialise(s)}\n"
    result = _run(payload, expected_count=1, label="OTE")
    return validate_transition_entropy_backend_output(
        result[0],
        expected=expected_transition_entropy_backend_output(s, d, tau),
        atol=1.0e-9,
    )
