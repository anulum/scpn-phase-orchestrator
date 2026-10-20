# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for delayed Kuramoto

"""Go backend for ``upde/delay.py`` via ``libdelay.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._go_runtime import load_go_library
from ._delay_validation import (
    validate_delay_backend_inputs,
    validate_delay_backend_output,
)

__all__ = ["delayed_kuramoto_run_go"]
FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libdelay.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libdelay.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libdelay.so delay.go"
        )
    lib = load_go_library(_LIB_PATH)
    lib.DelayedKuramotoRun.restype = ctypes.c_int
    lib.DelayedKuramotoRun.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def delayed_kuramoto_run_go(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    delay_steps: int,
    n_steps: int,
) -> FloatArray:
    """Integrate through Go, returning an independent input copy for zero steps.

    Positive batches must fit the C ABI's signed count range. Delays longer
    than the batch are reduced to its length, preserving current-phase warmup.
    Invalid controls, native refusals and invalid phase results raise ValueError.

    Parameters
    ----------
    phases, omegas : FloatArray
        Finite source phases in radians and frequencies in rad/s, shape ``(n,)``.
    knm_flat, alpha_flat : FloatArray
        Row-major coupling and phase-lag matrices, each with ``n*n`` entries.
    n : int
        Positive oscillator count matching the supplied buffers.
    zeta, psi : float
        Finite external drive strength and reference phase in radians.
    dt : float
        Positive finite Euler timestep in seconds.
    delay_steps : int
        Positive discrete delay; steps before this delay use current phases.
    n_steps : int
        Nonnegative number of integration steps.

    Returns
    -------
    FloatArray
        Independent float64 final phases. Positive batches project into
        ``[0, 2*pi)`` with positive zero; zero steps preserve the validated input.

    Raises
    ------
    ValueError
        If inputs, native arithmetic or the returned phase vector are invalid.
    ImportError
        If a positive batch cannot load its actual native artifact.
    """
    ph, om, knm, alpha, n, zeta, psi, dt, delay_steps, n_steps = (
        validate_delay_backend_inputs(
            phases, omegas, knm_flat, alpha_flat, n, zeta, psi, dt, delay_steps, n_steps
        )
    )
    if n_steps == 0:
        return ph
    if n > np.iinfo(np.int32).max or n_steps > np.iinfo(np.int32).max:
        raise ValueError("Go delay n and n_steps must fit signed 32-bit counts")
    lib = _load_lib()
    out: FloatArray = np.zeros(n, dtype=np.float64)
    rc = lib.DelayedKuramotoRun(
        ph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        om.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        knm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        ctypes.c_double(float(zeta)),
        ctypes.c_double(float(psi)),
        ctypes.c_double(float(dt)),
        ctypes.c_int(min(delay_steps, n_steps)),
        ctypes.c_int(int(n_steps)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go DelayedKuramotoRun rc={rc}")
    return validate_delay_backend_output(out, n=n)
