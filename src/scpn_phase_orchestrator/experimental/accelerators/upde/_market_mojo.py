# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for market PLV / R(t)

"""Mojo backend for ``upde/market.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._market_validation import (
    validate_market_order_inputs,
    validate_market_order_output,
    validate_market_plv_inputs,
    validate_market_plv_output,
)

__all__ = [
    "_ensure_exe",
    "market_order_parameter_mojo",
    "market_plv_mojo",
]

FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "market_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/market.mojo -o mojo/market_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run_mojo(
    tokens: list[str],
    *,
    expected_lines: int,
    label: str,
) -> list[float]:
    exe = _ensure_exe()
    proc = subprocess.run(  # nosec B603
        [str(exe)],
        input=" ".join(tokens) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(f"Mojo market exit {proc.returncode}: {proc.stderr.strip()}")
    lines = proc.stdout.splitlines()
    if len(lines) != expected_lines:
        message = (
            f"Mojo market {label} returned {len(lines)} lines, "
            f"expected {expected_lines}"
        )
        raise ValueError(message)
    values: list[float] = []
    for line in lines:
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                f"Mojo market {label} output must be finite real values"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Mojo market {label} output must be finite real values")
        values.append(value)
    return values


def market_order_parameter_mojo(
    phases_flat: FloatArray,
    t: int,
    n: int,
) -> FloatArray:
    """Compute market phase order parameter.

    The calculation is delegated to the Mojo backend.
    """
    p, t_i, n_i = validate_market_order_inputs(phases_flat, t, n)
    tokens = ["ORDER", str(t_i), str(n_i)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    values = _run_mojo(tokens, expected_lines=t_i, label="ORDER")
    return validate_market_order_output(values, t=t_i)


def market_plv_mojo(
    phases_flat: FloatArray,
    t: int,
    n: int,
    window: int,
) -> FloatArray:
    """Compute market phase-locking value.

    The calculation is delegated to the Mojo backend.
    """
    p, t_i, n_i, window_i = validate_market_plv_inputs(
        phases_flat,
        t,
        n,
        window,
    )
    n_windows = t_i - window_i + 1
    tokens = ["PLV", str(t_i), str(n_i), str(window_i)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    values = _run_mojo(
        tokens,
        expected_lines=n_windows * n_i * n_i,
        label="PLV",
    )
    return validate_market_plv_output(values, t=t_i, n=n_i, window=window_i)
