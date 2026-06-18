# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for entropy production rate

"""Mojo backend for ``monitor/entropy_prod.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._entropy_prod_validation import (
    validate_entropy_prod_backend_inputs,
    validate_entropy_prod_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["_ensure_exe", "entropy_production_rate_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "entropy_prod_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/entropy_prod.mojo "
            f"-o mojo/entropy_prod_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> float:
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
            f"Mojo entropy_prod returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != 1:
        raise ValueError(
            f"Mojo entropy_prod must emit exactly one scalar line, got {len(lines)}"
        )
    try:
        value = float(lines[0])
    except ValueError as exc:
        raise ValueError(
            f"Mojo entropy_prod emitted a non-scalar entropy-production value: "
            f"{lines[0]!r}"
        ) from exc
    return validate_entropy_prod_backend_output(value)


def entropy_production_rate_mojo(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: float,
    dt: float,
) -> float:
    """Compute the entropy-production-rate monitor through the Mojo backend."""
    phases, omegas, knm, alpha, dt = validate_entropy_prod_backend_inputs(
        phases,
        omegas,
        knm,
        alpha,
        dt,
    )
    n = int(phases.size)
    if n == 0 or dt <= 0.0:
        return 0.0
    tokens: list[str] = [
        "EP",
        str(n),
        repr(float(alpha)),
        repr(float(dt)),
    ]
    tokens.extend(repr(float(x)) for x in phases.ravel().tolist())
    tokens.extend(repr(float(x)) for x in omegas.ravel().tolist())
    tokens.extend(repr(float(x)) for x in knm.ravel().tolist())
    return validate_entropy_prod_backend_output(_run(" ".join(tokens) + "\n"))
