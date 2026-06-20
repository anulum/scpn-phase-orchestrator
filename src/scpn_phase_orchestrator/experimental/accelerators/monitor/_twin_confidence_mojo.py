# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for twin-confidence divergence

"""Mojo backend for ``monitor/twin_confidence.py`` via a subprocess executable.

Loads ``mojo/twin_confidence_mojo``; feeds the arguments as a single
whitespace-separated stdin payload and parses the two-line f64 output.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._twin_confidence_validation import (
    validate_twin_divergence_backend_inputs,
    validate_twin_divergence_backend_output,
)

__all__ = ["_ensure_exe", "twin_divergence_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "twin_confidence_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/twin_confidence.mojo "
            f"-o mojo/twin_confidence_mojo -Xlinker -lm"
        )
    return _EXE_PATH


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
            f"Mojo twin_confidence returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"Mojo twin_confidence must emit exactly {expected_count} scalar "
            f"line(s), got {len(lines)}"
        )
    values: list[float] = []
    for line in lines:
        try:
            values.append(float(line))
        except ValueError as exc:
            raise ValueError(
                f"Mojo twin_confidence emitted a non-scalar value: {line!r}"
            ) from exc
    return values


def twin_divergence_mojo(
    model_phases: FloatArray,
    observed_phases: FloatArray,
    model_order: FloatArray,
    observed_order: FloatArray,
    n: int,
    w: int,
    n_bins: int,
) -> FloatArray:
    """Compute the twin divergence pair through the Mojo backend.

    Parameters
    ----------
    model_phases, observed_phases : FloatArray
        Model and observed phase vectors of length ``n``.
    model_order, observed_order : FloatArray
        Model and observed order-parameter windows of length ``w``.
    n, w, n_bins : int
        Phase count, order-window length, and histogram bin count.

    Returns
    -------
    FloatArray
        Two-element ``[phase_js_divergence, order_wasserstein]`` array.

    Raises
    ------
    ValueError
        If the inputs are invalid or the Mojo kernel reports a contract failure.
    """
    (
        model_phases64,
        observed_phases64,
        model_order64,
        observed_order64,
        n_int,
        w_int,
        n_bins_int,
    ) = validate_twin_divergence_backend_inputs(
        model_phases,
        observed_phases,
        model_order,
        observed_order,
        n,
        w,
        n_bins,
    )
    tokens: list[str] = [str(n_int), str(w_int), str(n_bins_int)]
    tokens.extend(repr(float(x)) for x in model_phases64.tolist())
    tokens.extend(repr(float(x)) for x in observed_phases64.tolist())
    tokens.extend(repr(float(x)) for x in model_order64.tolist())
    tokens.extend(repr(float(x)) for x in observed_order64.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=2)
    return validate_twin_divergence_backend_output(result)
