# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for AttnRes coupling modulation

"""Mojo backend for the AttnRes fallback chain.

Mojo 0.26 has not yet finalised the ``UnsafePointer`` C-ABI story:
``MutableAnyOrigin`` is not exported in the public stdlib and the
origin-inference rules for pointer-typed ``@export(ABI="C")`` args
still reject straightforward declarations. Rather than pin the whole
chain to a single experimental API revision, the bridge shells out
to the compiled ``mojo/attnres_mojo`` executable with a text stdin
protocol. The subprocess cost is real — Rust and Go stay preferred
for perf — but the Mojo path works today and serves the dispatcher
as a functional alternative.

Upgrade path: when Mojo stabilises `UnsafePointer`'s C-ABI surface
(likely 0.27+), swap this bridge for a ctypes wrapper around a
shared library built with ``mojo build --emit shared-lib``. The
algorithm in ``mojo/attnres.mojo`` stays unchanged; only the I/O
plumbing moves.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["attnres_modulate_mojo"]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "attnres_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: "
            f"mojo build mojo/attnres.mojo -o mojo/attnres_mojo "
            f"-Xlinker -lm"
        )
    return _EXE_PATH


def attnres_modulate_mojo(
    knm_flat: NDArray,
    theta: NDArray,
    n: int,
    block_size: int,
    temperature: float,
    lambda_: float,
) -> NDArray:
    """Mojo-backed AttnRes modulation. Signature matches the Rust FFI."""
    exe = _ensure_exe()

    # Flatten all data onto a single whitespace-separated line (Mojo
    # 0.26 ``input()`` reads a single line only). Precision is 17 digits
    # which is enough to round-trip float64 without loss.
    knm_arr = np.ascontiguousarray(knm_flat, dtype=np.float64)
    theta_arr = np.ascontiguousarray(theta, dtype=np.float64)

    tokens: list[str] = [
        str(n),
        str(block_size),
        repr(float(temperature)),
        repr(float(lambda_)),
    ]
    tokens.extend(repr(float(x)) for x in knm_arr.tolist())
    tokens.extend(repr(float(x)) for x in theta_arr.tolist())
    payload = " ".join(tokens) + "\n"

    proc = subprocess.run(
        [str(exe)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo attnres returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    result = np.array(
        [float(line) for line in proc.stdout.strip().splitlines() if line],
        dtype=np.float64,
    )
    if result.size != n * n:
        raise ValueError(
            f"Mojo returned {result.size} values, expected {n * n}"
        )
    return result
