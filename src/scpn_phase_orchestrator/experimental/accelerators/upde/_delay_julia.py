# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for delayed Kuramoto

"""Julia backend for ``upde/delay.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

from ._delay_validation import (
    validate_delay_backend_inputs,
    validate_delay_backend_output,
)

__all__ = ["delayed_kuramoto_run_julia"]
FloatArray: TypeAlias = NDArray[np.float64]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "delay.jl"


class _DelayJulia(Protocol):
    """Numerical entry point exported by the loaded DelayJL Julia module."""

    def delayed_kuramoto_run(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        n: int,
        zeta: float,
        psi: float,
        dt: float,
        delay: int,
        steps: int,
    ) -> object:
        """Return the Julia phase vector or propagate the native refusal."""
        ...


_JULIA_MODULE: _DelayJulia | None = None


def _ensure() -> _DelayJulia:
    """Load a real Julia side-file, refusing absent or unusable artifacts."""
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    JuliaMain = require_julia_main()

    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    try:
        module = JuliaMain.include(str(_JULIA_FILE))
    except Exception as exc:
        raise ImportError(
            f"Julia delay side-file could not load: {_JULIA_FILE}"
        ) from exc
    if not callable(getattr(module, "delayed_kuramoto_run", None)):
        raise ImportError(
            "Julia delay side-file has no delayed_kuramoto_run entry point"
        )
    _JULIA_MODULE = cast("_DelayJulia", module)
    return _JULIA_MODULE


def delayed_kuramoto_run_julia(
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
    """Integrate through Julia, returning an independent input copy for zero steps.

    A positive batch requires its actual side-file. Julia numerical refusals
    and invalid output become ValueError with the native cause retained.

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
    jl = _ensure()
    try:
        result = jl.delayed_kuramoto_run(
            ph, om, knm, alpha, n, zeta, psi, dt, min(delay_steps, n_steps), n_steps
        )
    except Exception as exc:
        raise ValueError(
            "Julia delayed integration refused the numerical step"
        ) from exc
    return validate_delay_backend_output(result, n=n)
