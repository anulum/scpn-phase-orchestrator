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
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._hypergraph_validation import (
    TWO_PI,
    validate_hypergraph_inputs,
    validate_hypergraph_output,
)

__all__ = ["_ensure_exe", "hypergraph_run_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "hypergraph_mojo"
Float64Array: TypeAlias = NDArray[np.float64]
Int64Array: TypeAlias = NDArray[np.int64]


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/hypergraph.mojo -o mojo/hypergraph_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def hypergraph_run_mojo(
    phases: Float64Array,
    omegas: Float64Array,
    n: int,
    edge_nodes: Int64Array,
    edge_offsets: Int64Array,
    edge_strengths: Float64Array,
    knm_flat: Float64Array,
    alpha_flat: Float64Array,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> Float64Array:
    """Integrate hypergraph Kuramoto dynamics.

    The calculation is delegated to the Mojo backend.
    """
    (
        p,
        o,
        n_i,
        en,
        eo,
        es,
        kn,
        al,
        zeta_f,
        psi_f,
        dt_f,
        steps_i,
    ) = validate_hypergraph_inputs(
        phases,
        omegas,
        n,
        edge_nodes,
        edge_offsets,
        edge_strengths,
        knm_flat,
        alpha_flat,
        zeta,
        psi,
        dt,
        n_steps,
    )
    if steps_i == 0:
        return np.mod(p, TWO_PI)
    exe = _ensure_exe()
    tokens: list[str] = [
        "HGRUN",
        str(n_i),
        str(int(en.size)),
        str(int(eo.size)),
        str(int(kn.size)),
        str(int(al.size)),
        repr(zeta_f),
        repr(psi_f),
        repr(dt_f),
        str(steps_i),
    ]
    tokens.extend(repr(float(x)) for x in p.tolist())
    tokens.extend(repr(float(x)) for x in o.tolist())
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
    lines = proc.stdout.splitlines()
    if len(lines) != n_i:
        raise ValueError(f"Mojo HGRUN returned {len(lines)} lines, expected {n_i}")
    values: list[float] = []
    for line in lines:
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                "Mojo HGRUN output must be finite phases in [0, 2*pi)"
            ) from exc
        if not np.isfinite(value) or value < 0.0 or value >= TWO_PI:
            raise ValueError("Mojo HGRUN output must be finite phases in [0, 2*pi)")
        values.append(value)
    return validate_hypergraph_output(np.array(values, dtype=np.float64), n=n_i)
