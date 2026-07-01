# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — autotune coupling-estimator tests

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_phase_orchestrator.autotune.coupling_est as coupling_mod
from scpn_phase_orchestrator.autotune.coupling_est import (
    estimate_coupling,
    estimate_coupling_harmonics,
)

FloatArray = NDArray[np.float64]


def _phase_history() -> tuple[FloatArray, FloatArray, float]:
    dt = 0.05
    times = np.arange(0.0, 2.0, dt, dtype=np.float64)
    phases = np.vstack(
        [
            0.7 * times,
            1.1 * times + 0.2 * np.sin(times),
            0.9 * times + 0.15 * np.cos(2.0 * times),
        ]
    ).astype(np.float64)
    return phases, np.asarray([0.7, 1.1, 0.9], dtype=np.float64), dt


def test_estimate_coupling_and_harmonics_return_finite_zero_diagonal_matrices() -> None:
    phases, omegas, dt = _phase_history()

    first_order = estimate_coupling(phases, omegas, dt)
    harmonics = estimate_coupling_harmonics(phases, omegas, dt, n_harmonics=3)

    assert first_order.shape == (3, 3)
    assert np.all(np.isfinite(first_order))
    np.testing.assert_array_equal(np.diag(first_order), np.zeros(3))
    assert tuple(harmonics) == ("sin_1", "cos_1", "sin_2", "cos_2", "sin_3", "cos_3")
    for matrix in harmonics.values():
        assert matrix.shape == (3, 3)
        assert np.all(np.isfinite(matrix))
        np.testing.assert_array_equal(np.diag(matrix), np.zeros(3))


@pytest.mark.parametrize(
    ("phases", "omegas", "dt", "match"),
    [
        ([[False, True, False]], [1.0], 0.1, "boolean"),
        ([[0.0, 1.0 + 0.0j, 2.0]], [1.0], 0.1, "finite 2-D"),
        ([0.0, 1.0, 2.0], [1.0], 0.1, "finite 2-D"),
        (np.empty((0, 3), dtype=np.float64), [], 0.1, "at least one oscillator"),
        ([[0.0, np.inf, 2.0]], [1.0], 0.1, "finite values"),
        ([[0.0, 1.0, 2.0]], [True], 0.1, "boolean"),
        ([[0.0, 1.0, 2.0]], [1.0 + 0.0j], 0.1, "finite 1-D"),
        ([[0.0, 1.0, 2.0]], [[1.0]], 0.1, "finite 1-D"),
        ([[0.0, 1.0, 2.0]], [np.inf], 0.1, "finite values"),
        ([[0.0, 1.0, 2.0]], [1.0, 2.0], 0.1, "length"),
        ([[0.0, 1.0]], [1.0], 0.1, "Need >= 3 timesteps"),
        ([[0.0, 1.0, 2.0]], [1.0], True, "dt"),
        ([[0.0, 1.0, 2.0]], [1.0], 0.0, "dt"),
        ([[0.0, 1.0, 2.0]], [1.0], np.inf, "dt"),
    ],
)
def test_estimate_coupling_rejects_invalid_inputs(
    phases: object,
    omegas: object,
    dt: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        estimate_coupling(
            cast(FloatArray, phases),
            cast(FloatArray, omegas),
            cast(float, dt),
        )


@pytest.mark.parametrize("n_harmonics", [True, 0, -1, 1.5])
def test_estimate_coupling_harmonics_rejects_invalid_harmonic_counts(
    n_harmonics: object,
) -> None:
    phases, omegas, dt = _phase_history()

    with pytest.raises(ValueError, match="n_harmonics"):
        estimate_coupling_harmonics(phases, omegas, dt, cast(int, n_harmonics))


def test_estimate_coupling_rejects_raw_boolean_arrays_after_alias_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coupling_mod, "_contains_boolean_alias", lambda _value: False)

    with pytest.raises(ValueError, match="phases"):
        estimate_coupling(
            np.asarray([[False, True, False]], dtype=np.bool_),
            np.asarray([1.0], dtype=np.float64),
            0.1,
        )

    with pytest.raises(ValueError, match="omegas"):
        estimate_coupling(
            np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
            np.asarray([True], dtype=np.bool_),
            0.1,
        )


def test_estimate_coupling_rejects_non_numeric_conversion_payloads() -> None:
    with pytest.raises(ValueError, match="finite 2-D"):
        estimate_coupling(
            cast(FloatArray, np.asarray([["x", "y", "z"]], dtype=object)),
            np.asarray([1.0], dtype=np.float64),
            0.1,
        )

    with pytest.raises(ValueError, match="finite 1-D"):
        estimate_coupling(
            np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
            cast(FloatArray, np.asarray(["x"], dtype=object)),
            0.1,
        )


def test_estimate_coupling_harmonics_rejects_short_trajectories() -> None:
    with pytest.raises(ValueError, match="Need >= 3 timesteps"):
        estimate_coupling_harmonics(
            np.asarray([[0.0, 1.0]], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            0.1,
        )


def test_estimate_coupling_suppresses_singular_solver_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases, omegas, dt = _phase_history()

    def fail_lstsq(*_args: object, **_kwargs: object) -> tuple[FloatArray]:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(coupling_mod.np.linalg, "lstsq", fail_lstsq)

    np.testing.assert_array_equal(
        estimate_coupling(phases, omegas, dt), np.zeros((3, 3))
    )
    for matrix in estimate_coupling_harmonics(phases, omegas, dt).values():
        np.testing.assert_array_equal(matrix, np.zeros((3, 3)))


def test_estimate_coupling_skips_non_finite_solver_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases, omegas, dt = _phase_history()

    def nan_lstsq(*_args: object, **_kwargs: object) -> tuple[FloatArray]:
        return (np.full(18, np.nan, dtype=np.float64),)

    monkeypatch.setattr(coupling_mod.np.linalg, "lstsq", nan_lstsq)

    np.testing.assert_array_equal(
        estimate_coupling(phases, omegas, dt), np.zeros((3, 3))
    )
    for matrix in estimate_coupling_harmonics(phases, omegas, dt).values():
        np.testing.assert_array_equal(matrix, np.zeros((3, 3)))
