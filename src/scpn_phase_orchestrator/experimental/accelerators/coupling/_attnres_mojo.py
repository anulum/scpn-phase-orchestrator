# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for multi-head AttnRes

"""Execute the compiled Mojo phase attention kernel through versioned text.

The bridge negotiates protocol V2, carries the projection width explicitly,
and validates finite coupling output. Subprocess and serialization costs are
part of its measured runtime; no future pointer ABI or speed is promised.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling._attnres_validation import (
    validate_attnres_backend_inputs,
    validate_attnres_backend_output,
)

from .._mojo_runtime import require_mojo_executable, run_mojo_executable

__all__ = ["attnres_modulate_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "attnres_mojo"
FloatArray: TypeAlias = NDArray[np.float64]


def _ensure_exe() -> Path:
    """Require an existing executable implementing the dimension-aware protocol."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/attnres.mojo "
            f"-o mojo/attnres_mojo -Xlinker -lm"
        )
    exe = require_mojo_executable(_EXE_PATH)
    protocol = run_mojo_executable(exe, "PROTOCOL\n", runner=subprocess.run)
    if protocol.returncode != 0 or protocol.stdout.strip() != "2":
        raise ImportError(
            "Mojo AttnRes executable lacks the V2 dimension protocol; rebuild it"
        )
    return exe


def attnres_modulate_mojo(
    knm_flat: FloatArray,
    theta: FloatArray,
    w_q: FloatArray,
    w_k: FloatArray,
    w_v: FloatArray,
    w_o: FloatArray,
    n: int,
    n_heads: int,
    block_size: int,
    temperature: float,
    lambda_: float,
) -> FloatArray:
    """Compute phase attention coupling through the original Mojo runtime.

    Parameters
    ----------
    knm_flat : numpy.ndarray
        Row-major symmetric coupling graph, ``N*N`` finite real values.
    theta : numpy.ndarray
        ``N`` oscillator phases in radians.
    w_q, w_k, w_v, w_o : numpy.ndarray
        Row-major projection buffers containing ``D*D`` finite real values.
        ``D`` is even, at least two, and divisible by the head count.
    n : int
        Non-negative oscillator count matching the supplied graph and phases.
    n_heads : int
        Positive head count dividing the model width.
    block_size : int
        ``-1`` selects all neighbours; a positive value limits index distance.
    temperature : float
        Positive finite softmax temperature.
    lambda_ : float
        Non-negative finite multiplicative strength.

    Returns
    -------
    numpy.ndarray
        Flattened float64 coupling with validated symmetry, zero diagonal
        and preserved absent edges. Empty graphs need no optional runtime.

    Raises
    ------
    ValueError
        On invalid types, shapes, controls, topology, numerical intermediates
        or returned coupling values. Boolean, complex, text and temporal
        aliases are rejected before coercion.
    ImportError
        If the optional runtime or a compatible compiled artifact is missing.
    """
    (
        knm_flat,
        theta,
        w_q,
        w_k,
        w_v,
        w_o,
        n,
        n_heads,
        block_size,
        temperature,
        lambda_,
    ) = validate_attnres_backend_inputs(
        knm_flat,
        theta,
        w_q,
        w_k,
        w_v,
        w_o,
        n,
        n_heads,
        block_size,
        temperature,
        lambda_,
    )
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    exe = _ensure_exe()

    tokens: list[str] = [
        "V2",
        str(n),
        str(n_heads),
        str(block_size),
        repr(float(temperature)),
        repr(float(lambda_)),
        str(math.isqrt(w_o.size)),
    ]
    tokens.extend(repr(float(x)) for x in knm_flat.tolist())
    tokens.extend(repr(float(x)) for x in theta.tolist())
    tokens.extend(repr(float(x)) for x in w_q.tolist())
    tokens.extend(repr(float(x)) for x in w_k.tolist())
    tokens.extend(repr(float(x)) for x in w_v.tolist())
    tokens.extend(repr(float(x)) for x in w_o.tolist())
    payload = " ".join(tokens) + "\n"

    proc = run_mojo_executable(exe, payload, runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo attnres returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != n * n:
        raise ValueError(f"Mojo returned {len(lines)} values, expected {n * n}")
    try:
        values = [float(line) for line in lines]
    except ValueError as exc:
        raise ValueError(
            "Mojo AttnRes output must contain finite modulated coupling values"
        ) from exc
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "Mojo AttnRes output must contain finite modulated coupling values"
        )
    return validate_attnres_backend_output(result, n=n, knm_flat=knm_flat)
