# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cybernetic closure public contracts

"""Focused public contract tests for the SSGF cybernetic closure loop."""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier
from scpn_phase_orchestrator.ssgf.closure import CyberneticClosure

FloatArray: TypeAlias = NDArray[np.float64]


def _carrier() -> GeometryCarrier:
    """Build a deterministic carrier for cybernetic closure contract tests."""
    return GeometryCarrier(n_oscillators=4, z_dim=3, lr=0.05, seed=17)


@pytest.mark.parametrize(
    ("carrier", "match"),
    (
        (object(), "carrier must be GeometryCarrier"),
        (None, "carrier must be GeometryCarrier"),
    ),
)
def test_constructor_rejects_non_geometry_carrier(
    carrier: object,
    match: str,
) -> None:
    """The closure accepts only a real GeometryCarrier instance."""
    with pytest.raises(TypeError, match=match):
        CyberneticClosure(cast(GeometryCarrier, carrier))


@pytest.mark.parametrize(
    "cost_weights",
    ([], (), [1.0, 0.5]),
)
def test_constructor_rejects_non_tuple_or_empty_cost_weights(
    cost_weights: object,
) -> None:
    """The closure requires a non-empty tuple of cost weights."""
    with pytest.raises(TypeError, match="non-empty tuple"):
        CyberneticClosure(
            _carrier(),
            cost_weights=cast(tuple[float, ...], cost_weights),
        )


@pytest.mark.parametrize(
    "cost_weights",
    (
        (True,),
        (1.0, "0.5"),
    ),
)
def test_constructor_rejects_non_real_cost_weights(
    cost_weights: tuple[object, ...],
) -> None:
    """The closure rejects boolean and non-real cost weights."""
    with pytest.raises(TypeError, match="tuple of finite reals"):
        CyberneticClosure(
            _carrier(),
            cost_weights=cast(tuple[float, ...], cost_weights),
        )


@pytest.mark.parametrize(
    "cost_weights",
    (
        (float("nan"),),
        (float("inf"),),
    ),
)
def test_constructor_rejects_non_finite_cost_weights(
    cost_weights: tuple[float, ...],
) -> None:
    """The closure rejects non-finite cost weights."""
    with pytest.raises(ValueError, match="finite reals"):
        CyberneticClosure(_carrier(), cost_weights=cost_weights)


@pytest.mark.parametrize("max_steps", (True, 1.25, "3"))
def test_constructor_rejects_non_integer_max_steps(max_steps: object) -> None:
    """The closure rejects boolean and non-integer maximum-step limits."""
    with pytest.raises(TypeError, match="non-negative integer"):
        CyberneticClosure(_carrier(), max_steps=cast(int, max_steps))


@pytest.mark.parametrize(
    "phases",
    (
        [0.0, 0.1, 0.2, 0.3],
        (0.0, 0.1, 0.2, 0.3),
    ),
)
def test_step_rejects_non_ndarray_phase_vectors(phases: object) -> None:
    """The step boundary requires a NumPy phase vector."""
    with pytest.raises(TypeError, match="numpy.ndarray"):
        CyberneticClosure(_carrier()).step(cast(FloatArray, phases))


def test_step_rejects_non_vector_phase_arrays() -> None:
    """The step boundary rejects matrix-shaped phase arrays."""
    phases = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="1D vector"):
        CyberneticClosure(_carrier()).step(phases)


def test_step_rejects_non_finite_phase_values() -> None:
    """The step boundary rejects NaN or infinite phase values before updates."""
    phases = np.array([0.0, 0.5, np.nan, 1.5], dtype=np.float64)

    with pytest.raises(ValueError, match="finite values"):
        CyberneticClosure(_carrier()).step(phases)


def test_run_rejects_non_ndarray_phase_vectors() -> None:
    """The run boundary requires a NumPy phase vector."""
    phases = [0.0, 0.1, 0.2, 0.3]

    with pytest.raises(TypeError, match="numpy.ndarray"):
        CyberneticClosure(_carrier()).run(cast(FloatArray, phases), 1)
