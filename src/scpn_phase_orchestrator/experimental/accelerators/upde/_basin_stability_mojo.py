# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for steady-state R

"""Mojo backend for ``upde/basin_stability.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._basin_stability_validation import (
    validate_basin_stability_inputs,
    validate_basin_stability_output,
)

__all__ = ["_ensure_exe", "steady_state_r_mojo"]

FloatArray = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "basin_stability_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/basin_stability.mojo -o mojo/basin_stability_mojo "
            f"-Xlinker -lm"
        )
    return _EXE_PATH


def steady_state_r_mojo(
    phases_init: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    k_scale: float,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> float:
    """Compute steady-state order parameter for basin-stability trials.

    The calculation is delegated to the Mojo backend.
    """
    (
        p,
        o,
        k,
        a,
        n_i,
        k_scale_f,
        dt_f,
        n_transient_i,
        n_measure_i,
    ) = validate_basin_stability_inputs(
        phases_init,
        omegas,
        knm_flat,
        alpha_flat,
        n,
        k_scale,
        dt,
        n_transient,
        n_measure,
    )
    if n_measure_i == 0:
        return 0.0
    exe = _ensure_exe()
    tokens: list[str] = [
        "STEADY",
        str(n_i),
        repr(k_scale_f),
        repr(dt_f),
        str(n_transient_i),
        str(n_measure_i),
    ]
    tokens.extend(repr(float(x)) for x in p.tolist())
    tokens.extend(repr(float(x)) for x in o.tolist())
    tokens.extend(repr(float(x)) for x in k.tolist())
    tokens.extend(repr(float(x)) for x in a.tolist())
    proc = subprocess.run(  # nosec B603
        [str(exe)],
        input=" ".join(tokens) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo basin_stability exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != 1:
        raise ValueError(f"Mojo STEADY returned {len(lines)} lines, expected 1")
    try:
        steady_state_r = float(lines[0])
    except ValueError as exc:
        raise ValueError(
            "Mojo STEADY output must contain one finite steady-state R scalar"
        ) from exc
    return validate_basin_stability_output(steady_state_r)
