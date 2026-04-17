# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for order parameters

"""Mojo backend for ``upde/order_params.py``.

Subprocess bridge — same rationale as the AttnRes Mojo path
(Mojo 0.26 ``UnsafePointer`` C-ABI in transition).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "order_parameter_mojo",
    "plv_mojo",
    "layer_coherence_mojo",
]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "order_params_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/order_params.mojo "
            f"-o mojo/order_params_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> list[float]:
    exe = _ensure_exe()
    proc = subprocess.run(
        [str(exe)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo order_params returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return [
        float(line)
        for line in proc.stdout.strip().splitlines()
        if line
    ]


def order_parameter_mojo(phases: NDArray) -> tuple[float, float]:
    if phases.size == 0:
        return (0.0, 0.0)
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    tokens = ["R", str(p.size)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != 2:
        raise ValueError(f"Mojo returned {len(result)} values, expected 2")
    return float(result[0]), float(result[1])


def plv_mojo(phases_a: NDArray, phases_b: NDArray) -> float:
    if phases_a.size == 0 or phases_b.size == 0:
        return 0.0
    if phases_a.size != phases_b.size:
        raise ValueError(
            f"PLV requires equal-length arrays, got "
            f"{phases_a.size} vs {phases_b.size}"
        )
    a = np.ascontiguousarray(phases_a.ravel(), dtype=np.float64)
    b = np.ascontiguousarray(phases_b.ravel(), dtype=np.float64)
    tokens = ["PLV", str(a.size)]
    tokens.extend(repr(float(x)) for x in a.tolist())
    tokens.extend(repr(float(x)) for x in b.tolist())
    result = _run(" ".join(tokens) + "\n")
    return float(result[0])


def layer_coherence_mojo(phases: NDArray, indices: NDArray) -> float:
    if indices.size == 0:
        return 0.0
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    idx = np.ascontiguousarray(indices.ravel(), dtype=np.int64)
    tokens = ["LC", str(p.size)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    tokens.append(str(idx.size))
    tokens.extend(str(int(i)) for i in idx.tolist())
    result = _run(" ".join(tokens) + "\n")
    return float(result[0])
