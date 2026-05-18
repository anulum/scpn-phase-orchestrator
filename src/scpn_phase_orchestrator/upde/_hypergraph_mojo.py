# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for hypergraph Kuramoto

"""Mojo backend for ``upde/hypergraph.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "hypergraph_run_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "hypergraph_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/hypergraph.mojo -o mojo/hypergraph_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def hypergraph_run_mojo(
    phases: NDArray[np.float64],
    omegas: NDArray[np.float64],
    n: int,
    edge_nodes: NDArray[np.int64],
    edge_offsets: NDArray[np.int64],
    edge_strengths: NDArray[np.float64],
    knm_flat: NDArray[np.float64],
    alpha_flat: NDArray[np.float64],
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    """Integrate hypergraph Kuramoto dynamics.

    The calculation is delegated to the Mojo backend.
    """

    exe = _ensure_exe()
    en = np.asarray(edge_nodes).ravel()
    eo = np.asarray(edge_offsets).ravel()
    es = np.asarray(edge_strengths).ravel()
    kn = np.asarray(knm_flat).ravel()
    al = np.asarray(alpha_flat).ravel()
    tokens: list[str] = [
        "HGRUN",
        str(int(n)),
        str(int(en.size)),
        str(int(eo.size)),
        str(int(kn.size)),
        str(int(al.size)),
        repr(float(zeta)),
        repr(float(psi)),
        repr(float(dt)),
        str(int(n_steps)),
    ]
    tokens.extend(repr(float(x)) for x in np.asarray(phases).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(omegas).ravel().tolist())
    tokens.extend(str(int(x)) for x in en.tolist())
    tokens.extend(str(int(x)) for x in eo.tolist())
    tokens.extend(repr(float(x)) for x in es.tolist())
    tokens.extend(repr(float(x)) for x in kn.tolist())
    tokens.extend(repr(float(x)) for x in al.tolist())
    proc = subprocess.run(  # nosec B603
        [str(exe)],
        input=" ".join(tokens) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo hypergraph exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.strip().splitlines()
    if len(lines) != n:
        raise ValueError(f"Mojo HGRUN returned {len(lines)} lines, expected {n}")
    return np.array([float(x) for x in lines], dtype=np.float64)
