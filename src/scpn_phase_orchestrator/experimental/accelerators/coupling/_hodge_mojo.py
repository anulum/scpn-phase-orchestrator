# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for Hodge decomposition

"""Mojo backend for ``coupling/hodge.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._mojo_runtime import require_mojo_executable, run_mojo_executable
from ._hodge_validation import (
    validate_hodge_backend_inputs,
    validate_hodge_backend_output,
)

__all__ = ["_ensure_exe", "hodge_decomposition_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "hodge_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/hodge.mojo "
            f"-o mojo/hodge_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def hodge_decomposition_mojo(
    knm_flat: FloatArray,
    phases: FloatArray,
    n: int,
    edges_flat: NDArray[np.int64],
    n_edges: int,
    tris_flat: NDArray[np.int64],
    n_tris: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute the Hodge gradient, curl, and harmonic flow matrices via Mojo."""
    k, p, n, edges, n_edges, tris, n_tris = validate_hodge_backend_inputs(
        knm_flat,
        phases,
        n,
        edges_flat,
        n_edges,
        tris_flat,
        n_tris,
    )
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    exe = _ensure_exe()
    tokens: list[str] = ["HODGE", str(int(n)), str(int(n_edges)), str(int(n_tris))]
    tokens.extend(repr(float(x)) for x in k.tolist())
    tokens.extend(repr(float(x)) for x in p.tolist())
    tokens.extend(str(int(x)) for x in edges.tolist())
    tokens.extend(str(int(x)) for x in tris.tolist())
    proc = run_mojo_executable(exe, " ".join(tokens) + "\n", runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(f"Mojo hodge exit {proc.returncode}: {proc.stderr.strip()}")
    lines = proc.stdout.splitlines()
    expected = 3 * n * n
    if len(lines) != expected:
        raise ValueError(f"Mojo HODGE returned {len(lines)} lines, expected {expected}")
    try:
        values = [float(line) for line in lines]
    except ValueError as exc:
        raise ValueError(
            "Mojo HODGE output must contain finite gradient, curl, and "
            "harmonic components"
        ) from exc
    parsed = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(parsed)):
        raise ValueError(
            "Mojo HODGE output must contain finite gradient, curl, and "
            "harmonic components"
        )
    block = n * n
    gradient = parsed[:block].reshape(n, n).copy()
    curl = parsed[block : 2 * block].reshape(n, n).copy()
    harmonic = parsed[2 * block : 3 * block].reshape(n, n).copy()
    return validate_hodge_backend_output((gradient, curl, harmonic), n=n)
