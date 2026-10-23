# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for dimension kernels

"""Julia backend for ``monitor/dimension.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators._julia_runtime import (
    require_julia_main,
)

from ._dimension_validation import (
    expected_correlation_integral_backend_output,
    expected_kaplan_yorke_backend_output,
    validate_correlation_integral_backend_inputs,
    validate_correlation_integral_backend_output,
    validate_kaplan_yorke_backend_input,
    validate_kaplan_yorke_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = ["correlation_integral_julia", "kaplan_yorke_dimension_julia"]

_JULIA_FILE = Path(__file__).resolve().parents[5] / "julia" / "dimension.jl"


class _DimensionJulia(Protocol):
    """Actual numerical exports required from the loaded DimensionJL module."""

    def correlation_integral(
        self,
        trajectory: FloatArray,
        t: int,
        d: int,
        idx_i: IntArray,
        idx_j: IntArray,
        epsilons: FloatArray,
    ) -> object:
        """Return native correlation fractions or propagate the native refusal."""
        ...

    def kaplan_yorke_dimension(self, exponents: FloatArray) -> object:
        """Return the native scalar or propagate a numerical refusal."""
        ...


_JULIA_MODULE: _DimensionJulia | None = None


def _ensure() -> _DimensionJulia:
    """Load the real side-file and admit only its callable numerical exports."""
    global _JULIA_MODULE
    if _JULIA_MODULE is not None:
        return _JULIA_MODULE
    main = require_julia_main()
    if not _JULIA_FILE.exists():
        raise ImportError(f"julia side-file not found: {_JULIA_FILE}")
    try:
        module = main.include(str(_JULIA_FILE))
    except Exception as exc:
        raise ImportError(
            f"Julia dimension side-file could not load: {_JULIA_FILE}"
        ) from exc
    if not all(
        callable(getattr(module, name, None))
        for name in (
            "correlation_integral",
            "kaplan_yorke_dimension",
        )
    ):
        raise ImportError("Julia dimension side-file lacks callable CI/KY exports")
    _JULIA_MODULE = cast("_DimensionJulia", module)
    return _JULIA_MODULE


def correlation_integral_julia(
    traj_flat: FloatArray,
    t: int,
    d: int,
    idx_i: IntArray,
    idx_j: IntArray,
    epsilons: FloatArray,
) -> FloatArray:
    """Count selected strict-radius pairs through the Julia runtime.

    Parameters
    ----------
    traj_flat : ndarray of float64
        Flat finite real trajectory with exactly ``t * d`` entries.
    t, d : int
        Nonnegative sample count and positive coordinate count.
    idx_i, idx_j : ndarray of int64
        Equal-length, in-range, off-diagonal zero-based pair indices.
    epsilons : ndarray of float64
        Finite nonnegative radii in nondecreasing order.

    Returns
    -------
    ndarray of float64
        Normalised pair counts in radius order; zero when no pairs exist.

    Raises
    ------
    ValueError
        If inputs, runtime status or numerical outputs violate the contract.
    ImportError
        If the required Julia artifact cannot be admitted.
    """
    traj, t_int, d_int, ii, jj, eps = validate_correlation_integral_backend_inputs(
        traj_flat,
        t,
        d,
        idx_i,
        idx_j,
        epsilons,
    )
    jl = _ensure()
    expected = expected_correlation_integral_backend_output(
        traj,
        t_int,
        d_int,
        ii,
        jj,
        eps,
    )
    return validate_correlation_integral_backend_output(
        jl.correlation_integral(
            traj,
            t_int,
            d_int,
            ii,
            jj,
            eps,
        ),
        eps,
        expected=expected,
    )


def kaplan_yorke_dimension_julia(lyapunov_exponents: FloatArray) -> float:
    """Evaluate the Kaplan-Yorke spectrum formula through the Julia runtime.

    Parameters
    ----------
    lyapunov_exponents : ndarray of float64
        One-dimensional finite real spectrum; sorted internally without mutation.

    Returns
    -------
    float
        Dimension between zero and spectrum length, including both endpoints.

    Raises
    ------
    ValueError
        If the input, native status or result violates the spectrum contract.
    ImportError
        If the required Julia artifact cannot be admitted.
    """
    le = validate_kaplan_yorke_backend_input(lyapunov_exponents)
    jl = _ensure()
    expected = expected_kaplan_yorke_backend_output(le)
    return validate_kaplan_yorke_backend_output(
        jl.kaplan_yorke_dimension(
            le,
        ),
        le,
        expected=expected,
    )
