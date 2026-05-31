# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for UPDE engine

"""Mojo backend for ``upde/engine.py`` via ``mojo/upde_engine_mojo``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_validation import (
    validate_upde_backend_inputs,
    validate_upde_backend_output,
)

__all__ = ["_ensure_exe", "upde_run_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]

_METHOD_IDS = {"euler": 0, "rk4": 1, "rk45": 2}

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "upde_engine_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/upde_engine.mojo "
            f"-o mojo/upde_engine_mojo -Xlinker -lm"
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
            f"Mojo upde_engine returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"Mojo upde_engine returned {len(lines)} lines, expected {expected_count}"
        )
    values: list[float] = []
    for line in lines:
        try:
            values.append(float(line))
        except ValueError as exc:
            raise ValueError(
                "Mojo upde_engine output must be finite real values"
            ) from exc
    return values


def upde_run_mojo(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> FloatArray:
    """Run the core UPDE phase integrator.

    The calculation is delegated to the Mojo backend.
    """

    (
        p,
        o,
        k,
        a,
        zeta_f,
        psi_f,
        dt_f,
        n_steps_i,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validate_upde_backend_inputs(
        phases,
        omegas,
        knm,
        alpha,
        zeta,
        psi,
        dt,
        n_steps,
        method,
        n_substeps,
        atol,
        rtol,
    )
    n = int(p.size)
    if n_steps_i == 0:
        return p.copy()
    tokens: list[str] = [
        "RUN",
        str(n),
        repr(zeta_f),
        repr(psi_f),
        repr(dt_f),
        str(n_steps_i),
        str(_METHOD_IDS[method_s]),
        str(n_substeps_i),
        repr(atol_f),
        repr(rtol_f),
    ]
    tokens.extend(repr(float(x)) for x in p.tolist())
    tokens.extend(repr(float(x)) for x in o.tolist())
    tokens.extend(repr(float(x)) for x in k.tolist())
    tokens.extend(repr(float(x)) for x in a.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=n)
    return validate_upde_backend_output(np.array(result, dtype=np.float64), n=n)
