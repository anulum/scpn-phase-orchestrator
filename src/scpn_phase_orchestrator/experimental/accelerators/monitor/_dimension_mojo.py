# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for dimension kernels

"""Mojo backend for ``monitor/dimension.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._dimension_validation import (
    expected_correlation_integral_backend_output,
    expected_kaplan_yorke_backend_output,
    validate_correlation_integral_backend_inputs,
    validate_correlation_integral_backend_output,
    validate_kaplan_yorke_backend_input,
    validate_kaplan_yorke_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "_ensure_exe",
    "correlation_integral_mojo",
    "kaplan_yorke_dimension_mojo",
]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "dimension_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/dimension.mojo "
            f"-o mojo/dimension_mojo -Xlinker -lm"
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
            f"Mojo dimension returned exit {proc.returncode}: {proc.stderr.strip()}"
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
                f"Mojo {label} emitted a non-scalar dimension value: {line!r}"
            ) from exc
    return values


def correlation_integral_mojo(
    traj_flat: FloatArray,
    t: int,
    d: int,
    idx_i: IntArray,
    idx_j: IntArray,
    epsilons: FloatArray,
) -> FloatArray:
    """Compute the phase-space correlation integral through the Mojo backend."""
    traj, t_int, d_int, ii, jj, eps = validate_correlation_integral_backend_inputs(
        traj_flat,
        t,
        d,
        idx_i,
        idx_j,
        epsilons,
    )
    n_p = int(idx_i.size)
    n_k = int(eps.size)
    tokens: list[str] = [
        "CI",
        str(t_int),
        str(d_int),
        str(n_p),
        str(n_k),
    ]
    tokens.extend(str(int(x)) for x in ii.tolist())
    tokens.extend(str(int(x)) for x in jj.tolist())
    tokens.extend(repr(float(x)) for x in eps.tolist())
    tokens.extend(repr(float(x)) for x in traj.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=n_k, label="CI")
    expected = expected_correlation_integral_backend_output(
        traj,
        t_int,
        d_int,
        ii,
        jj,
        eps,
    )
    return validate_correlation_integral_backend_output(
        result,
        eps,
        expected=expected,
        atol=1e-9,
    )


def kaplan_yorke_dimension_mojo(lyapunov_exponents: FloatArray) -> float:
    """Estimate the Kaplan-Yorke dimension through the Mojo backend."""
    le = validate_kaplan_yorke_backend_input(lyapunov_exponents)
    n = int(le.size)
    tokens: list[str] = ["KY", str(n)]
    tokens.extend(repr(float(x)) for x in le.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=1, label="KY")
    expected = expected_kaplan_yorke_backend_output(le)
    return validate_kaplan_yorke_backend_output(
        result[0],
        le,
        expected=expected,
        atol=1e-9,
    )
