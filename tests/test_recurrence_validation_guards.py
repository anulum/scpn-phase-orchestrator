# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend recurrence validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _recurrence_validation as recurrence_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._recurrence_validation import (  # noqa: E501
    expected_recurrence_backend_output,
    validate_cross_recurrence_backend_inputs,
    validate_recurrence_backend_inputs,
    validate_recurrence_backend_output,
)


class _ArrayConversionFailure:
    def __array__(self, dtype: object | None = None) -> np.ndarray[Any, Any]:
        raise ValueError("synthetic conversion failure")


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "traj_flat": np.array([0.0, 1.0], dtype=np.float64),
        "t": 2,
        "d": 1,
        "epsilon": 0.5,
        "angular": False,
    }
    payload.update(overrides)
    return payload


class TestRecurrenceInputs:
    def test_valid_round_trips(self) -> None:
        traj, t, d, eps, ang = validate_recurrence_backend_inputs(**_inputs())
        assert (t, d) == (2, 1)

    def test_numeric_string_helper_ignores_non_string_scalars(self) -> None:
        assert not recurrence_validation._is_numeric_string_alias(1.0)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"t": True}, "t must"),
            ({"t": -1}, "t must"),
            ({"d": 0}, "d must"),
            ({"traj_flat": np.array([True, False])}, "must not contain boolean"),
            (
                {"traj_flat": np.array([True, 1.0], dtype=object)},
                "must not contain boolean",
            ),
            ({"traj_flat": np.array(["0.0", "1.0"])}, "numeric-string"),
            ({"traj_flat": np.array([0.0, "1.0"], dtype=object)}, "numeric-string"),
            ({"traj_flat": np.array([0.0 + 1j, 1.0])}, "must contain real-valued"),
            ({"traj_flat": np.array(["not-float", "1.0"])}, "must be a finite"),
            ({"traj_flat": np.array([[0.0], [1.0]])}, "must be one-dimensional"),
            ({"traj_flat": np.array([0.0, 1.0, 2.0])}, "does not match"),
            ({"traj_flat": np.array([0.0, np.inf])}, "only finite"),
            ({"epsilon": "wide"}, "epsilon must"),
            ({"epsilon": -0.5}, "epsilon must"),
            ({"angular": "yes"}, "angular must"),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_recurrence_backend_inputs(**_inputs(**overrides))


class TestCrossRecurrenceInputs:
    def test_valid_round_trips(self) -> None:
        result = validate_cross_recurrence_backend_inputs(
            traj_a_flat=np.array([0.0, 1.0]),
            traj_b_flat=np.array([0.5, 1.5]),
            t=2,
            d=1,
            epsilon=0.5,
            angular=False,
        )
        assert result[2] == 2  # t

    def test_rejects_corrupt_second_trajectory(self) -> None:
        with pytest.raises(ValueError, match="traj_b_flat .* does not match"):
            validate_cross_recurrence_backend_inputs(
                traj_a_flat=np.array([0.0, 1.0]),
                traj_b_flat=np.array([0.0, 1.0, 2.0]),
                t=2,
                d=1,
                epsilon=0.5,
                angular=False,
            )

    def test_rejects_numeric_string_second_trajectory(self) -> None:
        with pytest.raises(ValueError, match="traj_b_flat .*numeric-string"):
            validate_cross_recurrence_backend_inputs(
                traj_a_flat=np.array([0.0, 1.0]),
                traj_b_flat=np.array(["0.0", "1.0"]),
                t=2,
                d=1,
                epsilon=0.5,
                angular=False,
            )


class TestRecurrenceOutput:
    def test_valid_round_trips(self) -> None:
        out = validate_recurrence_backend_output(
            np.array([1, 0, 0, 1], dtype=np.uint8), t=2, name="recurrence"
        )
        assert out.size == 4

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (np.zeros(3), "output size must be 4"),
            (np.array([np.inf, 0, 0, 1]), "only finite values"),
            (np.array([0.5, 0, 0, 1]), "only 0/1 values"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_recurrence_backend_output(value, t=2, name="recurrence")

    def test_rejects_array_conversion_failure(self) -> None:
        with pytest.raises(ValueError, match="output must be array-like"):
            validate_recurrence_backend_output(
                _ArrayConversionFailure(), t=2, name="recurrence"
            )

    def test_rejects_nonnumeric_output(self) -> None:
        with pytest.raises(ValueError, match="output must be numeric"):
            validate_recurrence_backend_output(
                np.array(["one", "zero", "zero", "one"], dtype=object),
                t=2,
                name="recurrence",
            )

    def test_rejects_numeric_string_output_values(self) -> None:
        with pytest.raises(ValueError, match="output .*numeric-string"):
            validate_recurrence_backend_output(
                np.array(["1", "0", "0", "1"]),
                t=2,
                name="recurrence",
            )

    def test_recurrence_matrix_requires_true_diagonal(self) -> None:
        with pytest.raises(ValueError, match="must have true diagonal"):
            validate_recurrence_backend_output(
                np.array([0, 1, 1, 0], dtype=np.uint8), t=2, name="recurrence_matrix"
            )

    def test_recurrence_matrix_requires_symmetry(self) -> None:
        with pytest.raises(ValueError, match="must be symmetric"):
            validate_recurrence_backend_output(
                np.array([1, 1, 0, 1], dtype=np.uint8), t=2, name="recurrence_matrix"
            )

    def test_rejects_divergence_from_expected(self) -> None:
        with pytest.raises(ValueError, match="must match exact recurrence threshold"):
            validate_recurrence_backend_output(
                np.array([1, 0, 0, 1], dtype=np.uint8),
                t=2,
                name="recurrence",
                expected=np.array([1, 1, 1, 1], dtype=np.uint8),
            )

    def test_rejects_expected_output_size_mismatch(self) -> None:
        with pytest.raises(ValueError, match="expected output must have size 4"):
            validate_recurrence_backend_output(
                np.array([1, 0, 0, 1], dtype=np.uint8),
                t=2,
                name="recurrence",
                expected=np.array([1, 1, 1], dtype=np.uint8),
            )

    def test_rejects_nonnumeric_expected_output(self) -> None:
        with pytest.raises(ValueError, match="expected output must be numeric"):
            validate_recurrence_backend_output(
                np.array([1, 0, 0, 1], dtype=np.uint8),
                t=2,
                name="recurrence",
                expected=np.array(["one", "zero", "zero", "one"], dtype=object),
            )

    def test_rejects_numeric_string_expected_output(self) -> None:
        with pytest.raises(ValueError, match="expected output .*numeric-string"):
            validate_recurrence_backend_output(
                np.array([1, 0, 0, 1], dtype=np.uint8),
                t=2,
                name="recurrence",
                expected=np.array(["1", "0", "0", "1"]),
            )


def test_expected_recurrence_backend_output_reference() -> None:
    a = np.array([0.0, 1.0], dtype=np.float64)
    out = expected_recurrence_backend_output(a, a, t=2, d=1, epsilon=0.5, angular=False)
    # identical trajectories: diagonal recurs, off-diagonal exceeds epsilon
    assert out.reshape(2, 2).tolist() == [[1, 0], [0, 1]]


def test_expected_recurrence_backend_output_angular_reference() -> None:
    a = np.array([0.0, np.pi], dtype=np.float64)
    out = expected_recurrence_backend_output(a, a, t=2, d=1, epsilon=1.0, angular=True)
    assert out.reshape(2, 2).tolist() == [[1, 0], [0, 1]]
