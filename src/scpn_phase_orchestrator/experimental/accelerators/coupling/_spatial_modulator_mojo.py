# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for spatial coupling modulation

"""Mojo backend for ``coupling/spatial_modulator.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._spatial_modulator_validation import (
    validate_spatial_modulator_inputs,
    validate_spatial_modulator_output,
)

__all__ = ["_ensure_exe", "spatial_modulate_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "spatial_modulator_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/spatial_modulator.mojo "
            "-o mojo/spatial_modulator_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def spatial_modulate_mojo(
    k_nm_flat: FloatArray,
    positions_flat: FloatArray,
    n: int,
    dim: int,
    k_base: float,
    decay_form_code: int,
    decay_exponent: float,
    decay_length_scale: float,
    epsilon: float,
) -> FloatArray:
    """Compute a spatially modulated coupling matrix through the Mojo executable."""

    k, p, n, dim, k_base, form, exponent, length, eps = (
        validate_spatial_modulator_inputs(
            k_nm_flat,
            positions_flat,
            n,
            dim,
            k_base,
            decay_form_code,
            decay_exponent,
            decay_length_scale,
            epsilon,
        )
    )
    exe = _ensure_exe()
    tokens = [
        "SPATIAL_MODULATE",
        str(n),
        str(dim),
        str(form),
        repr(float(k_base)),
        repr(float(exponent)),
        repr(float(length)),
        repr(float(eps)),
    ]
    tokens.extend(repr(float(x)) for x in k.tolist())
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
            f"Mojo spatial modulator exit {proc.returncode}: {proc.stderr.strip()}"
        )
    try:
        values = [float(line) for line in proc.stdout.splitlines()]
    except ValueError as exc:
        raise ValueError(
            "Mojo spatial modulator output must be finite numeric values"
        ) from exc
    return validate_spatial_modulator_output(np.asarray(values, dtype=np.float64), n=n)
