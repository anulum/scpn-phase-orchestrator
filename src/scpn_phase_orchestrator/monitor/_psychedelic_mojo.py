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

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "entropy_from_phases_mojo"]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "psychedelic_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/psychedelic.mojo "
            f"-o mojo/psychedelic_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def entropy_from_phases_mojo(phases: NDArray, n_bins: int) -> float:
    exe = _ensure_exe()
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    tokens: list[str] = ["ENT", str(int(p.size)), str(int(n_bins))]
    tokens.extend(repr(float(x)) for x in p.tolist())
    proc = subprocess.run(
        [str(exe)], input=" ".join(tokens) + "\n",
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo psychedelic exit {proc.returncode}: {proc.stderr.strip()}"
        )
    return float(proc.stdout.strip().splitlines()[0])
