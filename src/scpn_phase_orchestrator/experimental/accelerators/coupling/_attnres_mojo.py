# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for multi-head AttnRes

"""Mojo backend for the multi-head AttnRes dispatcher.

Mojo 0.26 has not yet stabilised the ``UnsafePointer`` C-ABI surface,
so this bridge shells out to the compiled ``mojo/attnres_mojo``
executable with a one-line whitespace-separated text protocol. Swap
to ctypes + a ``mojo build --emit shared-lib`` artefact once the
pointer ABI ships (Mojo 0.27+); the algorithm in ``mojo/attnres.mojo``
stays the same.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._mojo_runtime import require_mojo_executable, run_mojo_executable
from ._attnres_validation import (
    validate_attnres_backend_inputs,
    validate_attnres_backend_output,
)

__all__ = ["attnres_modulate_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "attnres_mojo"
FloatArray: TypeAlias = NDArray[np.float64]


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/attnres.mojo "
            f"-o mojo/attnres_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


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
    """Mojo-backed multi-head AttnRes. Signature matches the Rust FFI.

    Pays a subprocess spawn + text-serialisation cost; used as the
    fallback between Julia and Go rather than the fast path.
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
        str(n),
        str(n_heads),
        str(block_size),
        repr(float(temperature)),
        repr(float(lambda_)),
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
    if result.size != n * n:
        raise ValueError(f"Mojo returned {result.size} values, expected {n * n}")
    return validate_attnres_backend_output(result, n=n, knm_flat=knm_flat)
