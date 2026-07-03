# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for order parameters

"""Mojo backend for ``upde/order_params.py``.

Subprocess bridge — same rationale as the AttnRes Mojo path
(Mojo 0.26 ``UnsafePointer`` C-ABI in transition).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._order_params_validation import (
    validate_layer_coherence_inputs,
    validate_order_parameter_inputs,
    validate_order_parameter_output,
    validate_plv_inputs,
    validate_unit_interval_output,
)

from .._mojo_runtime import require_mojo_executable, run_mojo_executable

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "order_parameter_mojo",
    "plv_mojo",
    "layer_coherence_mojo",
]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "order_params_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/order_params.mojo "
            f"-o mojo/order_params_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def _run(payload: str, *, expected_count: int, label: str) -> list[float]:
    """Call the backend kernel with the prepared inputs and return its result."""
    exe = _ensure_exe()
    proc = run_mojo_executable(exe, payload, runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo order_params returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"Mojo order_params {label} returned {len(lines)} lines, "
            f"expected {expected_count}"
        )
    values: list[float] = []
    for line in lines:
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                f"Mojo order_params {label} output must be finite real values"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"Mojo order_params {label} output must be finite real values"
            )
        values.append(value)
    return values


def order_parameter_mojo(phases: FloatArray) -> tuple[float, float]:
    """Compute the Kuramoto order parameter.

    The calculation is delegated to the Mojo backend.
    """
    p = validate_order_parameter_inputs(phases)
    if p.size == 0:
        return (0.0, 0.0)
    tokens = ["R", str(p.size)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=2, label="R")
    return validate_order_parameter_output(result[0], result[1])


def plv_mojo(phases_a: FloatArray, phases_b: FloatArray) -> float:
    """Compute phase-locking value.

    The calculation is delegated to the Mojo backend.
    """
    a, b = validate_plv_inputs(phases_a, phases_b)
    if a.size == 0:
        return 0.0
    tokens = ["PLV", str(a.size)]
    tokens.extend(repr(float(x)) for x in a.tolist())
    tokens.extend(repr(float(x)) for x in b.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=1, label="PLV")
    return validate_unit_interval_output(result[0], name="PLV")


def layer_coherence_mojo(phases: FloatArray, indices: IntArray) -> float:
    """Compute layer-wise phase coherence.

    The calculation is delegated to the Mojo backend.
    """
    p, idx = validate_layer_coherence_inputs(phases, indices)
    if idx.size == 0:
        return 0.0
    tokens = ["LC", str(p.size)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    tokens.append(str(idx.size))
    tokens.extend(str(int(i)) for i in idx.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=1, label="LC")
    return validate_unit_interval_output(result[0], name="layer coherence")
