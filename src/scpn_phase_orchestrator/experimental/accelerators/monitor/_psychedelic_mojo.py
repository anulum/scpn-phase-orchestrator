# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for psychedelic observables

"""Mojo backend for ``monitor/psychedelic.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._psychedelic_validation import (
    validate_psychedelic_backend_inputs,
    validate_psychedelic_entropy_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["_ensure_exe", "entropy_from_phases_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "psychedelic_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/psychedelic.mojo "
            f"-o mojo/psychedelic_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def entropy_from_phases_mojo(phases: FloatArray, n_bins: int) -> float:
    """Compute phase-distribution entropy through the Mojo backend."""

    p, bin_count = validate_psychedelic_backend_inputs(phases, n_bins)
    if p.size == 0:
        return 0.0
    exe = _ensure_exe()
    tokens: list[str] = ["ENT", str(int(p.size)), str(bin_count)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    proc = subprocess.run(  # nosec B603
        [str(exe)],
        input=" ".join(tokens) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo psychedelic exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = [line for line in proc.stdout.strip().splitlines() if line]
    if len(lines) != 1:
        raise ValueError(f"Mojo entropy returned {len(lines)} values")
    return validate_psychedelic_entropy_backend_output(float(lines[0]), bin_count)
