# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Koopman EDMD backend-contract validation tests

"""Tests for the shared Koopman EDMD backend input/output contract."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor._koopman_edmd_validation import (  # noqa: E501
    validate_edmd_backend_inputs,
    validate_edmd_backend_output,
)


class _ArrayProtocolFailure:
    """Fixture object whose NumPy array protocol fails deterministically."""

    def __array__(
        self, dtype: object | None = None, copy: object | None = None
    ) -> np.ndarray:
        raise TypeError("array protocol failed")


def _valid_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((5, 3)),
        np.zeros((5, 2)),
        np.zeros((5, 3)),
        np.zeros((5, 2)),
    )


def test_valid_inputs_return_their_dimensions() -> None:
    dims = validate_edmd_backend_inputs(*_valid_inputs())
    assert dims.samples == 5
    assert dims.lift_dim == 3
    assert dims.input_dim == 2
    assert dims.state_dim == 2


def test_inputs_reject_a_non_two_dimensional_matrix() -> None:
    _, inputs, y_lift, states = _valid_inputs()
    with pytest.raises(ValueError, match="must be a 2-D array"):
        validate_edmd_backend_inputs(np.zeros(5), inputs, y_lift, states)


def test_inputs_reject_an_empty_matrix() -> None:
    _, inputs, y_lift, states = _valid_inputs()
    with pytest.raises(ValueError, match="must be non-empty"):
        validate_edmd_backend_inputs(np.zeros((0, 3)), inputs, y_lift, states)


def test_inputs_reject_non_finite_values() -> None:
    x_lift, inputs, y_lift, states = _valid_inputs()
    x_lift = x_lift.copy()
    x_lift[0, 0] = np.inf
    with pytest.raises(ValueError, match="only finite values"):
        validate_edmd_backend_inputs(x_lift, inputs, y_lift, states)


@pytest.mark.parametrize(
    ("index", "name"),
    [
        (0, "x_lift"),
        (1, "inputs"),
        (2, "y_lift"),
        (3, "states"),
    ],
)
def test_inputs_reject_boolean_alias_matrices(index: int, name: str) -> None:
    matrices = list(_valid_inputs())
    matrices[index] = np.zeros(matrices[index].shape, dtype=np.bool_)
    with pytest.raises(ValueError, match=f"{name} must not contain boolean values"):
        validate_edmd_backend_inputs(*matrices)


def test_inputs_reject_complex_alias_matrices() -> None:
    x_lift, inputs, y_lift, states = _valid_inputs()
    complex_lift = x_lift.astype(np.complex128)
    complex_lift[0, 0] = 1.0 + 1.0j
    with pytest.raises(ValueError, match="x_lift must be real-valued"):
        validate_edmd_backend_inputs(complex_lift, inputs, y_lift, states)


def test_inputs_reject_numeric_string_matrices() -> None:
    x_lift, inputs, y_lift, states = _valid_inputs()
    string_inputs = np.full(inputs.shape, "0.0", dtype=object)
    with pytest.raises(ValueError, match="inputs must contain only real numbers"):
        validate_edmd_backend_inputs(x_lift, string_inputs, y_lift, states)


def test_inputs_reject_unicode_string_matrices() -> None:
    x_lift, inputs, y_lift, states = _valid_inputs()
    string_inputs = np.full(inputs.shape, "0.0")
    with pytest.raises(ValueError, match="inputs must contain only real numbers"):
        validate_edmd_backend_inputs(x_lift, string_inputs, y_lift, states)


def test_inputs_wrap_array_protocol_failures() -> None:
    _, inputs, y_lift, states = _valid_inputs()
    with pytest.raises(ValueError, match="x_lift must contain only real numbers"):
        validate_edmd_backend_inputs(_ArrayProtocolFailure(), inputs, y_lift, states)


def test_inputs_reject_a_sample_count_mismatch() -> None:
    x_lift, _, y_lift, states = _valid_inputs()
    with pytest.raises(ValueError, match="share the sample count"):
        validate_edmd_backend_inputs(x_lift, np.zeros((4, 2)), y_lift, states)


def test_inputs_reject_a_lift_dimension_mismatch() -> None:
    x_lift, inputs, _, states = _valid_inputs()
    with pytest.raises(ValueError, match="share the lift dimension"):
        validate_edmd_backend_inputs(x_lift, inputs, np.zeros((5, 4)), states)


def test_output_rejects_a_wrong_shape() -> None:
    dims = validate_edmd_backend_inputs(*_valid_inputs())
    with pytest.raises(ValueError, match="must have shape"):
        validate_edmd_backend_output(
            np.zeros((2, 2)), np.zeros((3, 2)), np.zeros((2, 3)), dims
        )


def test_output_rejects_non_finite_values() -> None:
    dims = validate_edmd_backend_inputs(*_valid_inputs())
    bad = np.full((3, 3), np.inf)
    with pytest.raises(ValueError, match="only finite values"):
        validate_edmd_backend_output(bad, np.zeros((3, 2)), np.zeros((2, 3)), dims)


def test_output_rejects_boolean_aliases_before_float_coercion() -> None:
    dims = validate_edmd_backend_inputs(*_valid_inputs())
    bad = np.ones((3, 3), dtype=np.bool_)
    with pytest.raises(ValueError, match="A must not contain boolean values"):
        validate_edmd_backend_output(bad, np.zeros((3, 2)), np.zeros((2, 3)), dims)


def test_output_rejects_complex_aliases_before_float_coercion() -> None:
    dims = validate_edmd_backend_inputs(*_valid_inputs())
    bad = np.ones((3, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="B must be real-valued"):
        validate_edmd_backend_output(np.zeros((3, 3)), bad, np.zeros((2, 3)), dims)


def test_output_rejects_numeric_strings_before_float_coercion() -> None:
    dims = validate_edmd_backend_inputs(*_valid_inputs())
    bad = np.full((2, 3), "0.0", dtype=object)
    with pytest.raises(ValueError, match="C must contain only real numbers"):
        validate_edmd_backend_output(np.zeros((3, 3)), np.zeros((3, 2)), bad, dims)


def test_output_accepts_correctly_shaped_finite_matrices() -> None:
    dims = validate_edmd_backend_inputs(*_valid_inputs())
    state_matrix, input_matrix, output_matrix = validate_edmd_backend_output(
        np.ones((3, 3)), np.ones((3, 2)), np.ones((2, 3)), dims
    )
    assert state_matrix.shape == (3, 3)
    assert input_matrix.shape == (3, 2)
    assert output_matrix.shape == (2, 3)
