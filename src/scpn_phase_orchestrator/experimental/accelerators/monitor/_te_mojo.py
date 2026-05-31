# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for transfer entropy

"""Mojo backend for ``monitor/transfer_entropy.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._te_validation import (
    expected_phase_te_backend_output,
    expected_te_matrix_backend_output,
    validate_phase_te_backend_inputs,
    validate_te_backend_output,
    validate_te_matrix_backend_inputs,
    validate_te_matrix_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["phase_te_mojo", "te_matrix_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "transfer_entropy_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/transfer_entropy.mojo "
            f"-o mojo/transfer_entropy_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str, *, expected_count: int, label: str) -> list[float]:
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
            f"Mojo transfer_entropy returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
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
                f"Mojo {label} emitted a non-scalar transfer-entropy value: {line!r}"
            ) from exc
    return values


def phase_te_mojo(source: FloatArray, target: FloatArray, n_bins: int) -> float:
    """Compute pairwise phase transfer entropy through the Mojo backend."""

    source, target, n_bins = validate_phase_te_backend_inputs(
        source,
        target,
        n_bins,
    )
    s = np.ascontiguousarray(source.ravel(), dtype=np.float64)
    t = np.ascontiguousarray(target.ravel(), dtype=np.float64)
    n = int(min(s.size, t.size))
    if n < 3:
        return 0.0
    tokens = ["PTE", str(n), str(n_bins)]
    tokens.extend(repr(float(x)) for x in s[:n].tolist())
    tokens.extend(repr(float(x)) for x in t[:n].tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=1, label="PTE")
    expected = expected_phase_te_backend_output(s[:n], t[:n], n_bins)
    return validate_te_backend_output(
        result[0],
        n_bins=n_bins,
        expected=expected,
        atol=1e-9,
    )


def te_matrix_mojo(
    phase_series: FloatArray,
    n_osc: int,
    n_time: int,
    n_bins: int,
) -> FloatArray:
    """Compute the phase transfer-entropy matrix through the Mojo backend."""

    phase_series, n_osc, n_time, n_bins = validate_te_matrix_backend_inputs(
        phase_series,
        n_osc,
        n_time,
        n_bins,
    )
    s = np.ascontiguousarray(phase_series, dtype=np.float64)
    tokens = ["MAT", str(n_osc), str(n_time), str(n_bins)]
    tokens.extend(repr(float(x)) for x in s.tolist())
    result = _run(
        " ".join(tokens) + "\n",
        expected_count=n_osc * n_osc,
        label="MAT",
    )
    expected = expected_te_matrix_backend_output(s, n_osc, n_time, n_bins)
    return validate_te_matrix_backend_output(
        result,
        n_osc=n_osc,
        n_bins=n_bins,
        expected=expected,
        atol=1e-9,
    )
