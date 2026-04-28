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

import numpy as np
from numpy.typing import NDArray

__all__ = ["attnres_modulate_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "attnres_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/attnres.mojo "
            f"-o mojo/attnres_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def attnres_modulate_mojo(
    knm_flat: NDArray,
    theta: NDArray,
    w_q: NDArray,
    w_k: NDArray,
    w_v: NDArray,
    w_o: NDArray,
    n: int,
    n_heads: int,
    block_size: int,
    temperature: float,
    lambda_: float,
) -> NDArray:
    """Mojo-backed multi-head AttnRes. Signature matches the Rust FFI.

    Pays a subprocess spawn + text-serialisation cost; used as the
    fallback between Julia and Go rather than the fast path.
    """
    exe = _ensure_exe()

    tokens: list[str] = [
        str(n),
        str(n_heads),
        str(block_size),
        repr(float(temperature)),
        repr(float(lambda_)),
    ]
    tokens.extend(repr(float(x)) for x in np.asarray(knm_flat).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(theta).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(w_q).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(w_k).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(w_v).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(w_o).ravel().tolist())
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
            f"Mojo attnres returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    result = np.array(
        [float(line) for line in proc.stdout.strip().splitlines() if line],
        dtype=np.float64,
    )
    if result.size != n * n:
        raise ValueError(f"Mojo returned {result.size} values, expected {n * n}")
    return result
