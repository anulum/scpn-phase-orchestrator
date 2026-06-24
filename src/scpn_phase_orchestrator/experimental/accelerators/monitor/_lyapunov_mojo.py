# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for Lyapunov spectrum

"""Mojo backend for ``monitor/lyapunov.py`` via a subprocess executable.

Loads ``mojo/lyapunov_mojo``; feeds the arguments as a single
whitespace-separated stdin payload and parses the ``n``-line f64 output.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._lyapunov_validation import (
    validate_lyapunov_backend_inputs,
    validate_lyapunov_backend_output,
)

__all__ = ["_ensure_exe", "lyapunov_spectrum_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "lyapunov_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/lyapunov.mojo "
            f"-o mojo/lyapunov_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str, *, expected_count: int, label: str) -> list[float]:
    """Call the backend kernel with the prepared inputs and return its result."""
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
            f"Mojo lyapunov returned exit {proc.returncode}: {proc.stderr.strip()}"
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
                f"Mojo {label} emitted a non-scalar Lyapunov exponent: {line!r}"
            ) from exc
    return values


def lyapunov_spectrum_mojo(
    phases_init: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    dt: float,
    n_steps: int,
    qr_interval: int,
    zeta: float,
    psi: float,
) -> FloatArray:
    """Estimate the Lyapunov spectrum through the Mojo backend."""
    (
        phases_init,
        omegas,
        knm,
        alpha,
        dt,
        n_steps,
        qr_interval,
        zeta,
        psi,
    ) = validate_lyapunov_backend_inputs(
        phases_init,
        omegas,
        knm,
        alpha,
        dt,
        n_steps,
        qr_interval,
        zeta,
        psi,
    )
    n = int(phases_init.size)
    tokens: list[str] = [
        "SPEC",
        str(n),
        repr(float(dt)),
        str(int(n_steps)),
        str(int(qr_interval)),
        repr(float(zeta)),
        repr(float(psi)),
    ]
    tokens.extend(repr(float(x)) for x in phases_init.ravel().tolist())
    tokens.extend(repr(float(x)) for x in omegas.ravel().tolist())
    tokens.extend(repr(float(x)) for x in knm.ravel().tolist())
    tokens.extend(repr(float(x)) for x in alpha.ravel().tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=n, label="SPEC")
    return validate_lyapunov_backend_output(result, n)
