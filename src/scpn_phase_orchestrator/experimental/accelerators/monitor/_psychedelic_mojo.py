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

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["_ensure_exe", "entropy_from_phases_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "psychedelic_mojo"


def _contains_boolean_alias(value: object) -> bool:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _validated_backend_inputs(phases: object, n_bins: object) -> tuple[FloatArray, int]:
    if _contains_boolean_alias(phases):
        raise ValueError("phases must not contain boolean values")
    raw = np.asarray(phases)
    if np.iscomplexobj(raw):
        raise ValueError("phases must contain real-valued samples")
    try:
        phase_values = raw.astype(np.float64, copy=True).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a finite real-valued vector") from exc
    if not np.all(np.isfinite(phase_values)):
        raise ValueError("phases must contain only finite values")
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, int):
        raise TypeError("n_bins must be an integer greater than or equal to 2")
    bin_count = int(n_bins)
    if bin_count < 2:
        raise ValueError("n_bins must be greater than or equal to 2")
    return np.ascontiguousarray(phase_values, dtype=np.float64), bin_count


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/psychedelic.mojo "
            f"-o mojo/psychedelic_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def entropy_from_phases_mojo(phases: FloatArray, n_bins: int) -> float:
    """Compute phase-distribution entropy through the Mojo backend."""

    p, bin_count = _validated_backend_inputs(phases, n_bins)
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
    return float(proc.stdout.strip().splitlines()[0])
