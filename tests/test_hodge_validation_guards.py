# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend Hodge input validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling import (
    _hodge_validation as hodge_validation,
)
from scpn_phase_orchestrator.coupling._hodge_validation import (  # noqa: E501
    validate_hodge_backend_inputs,
    validate_hodge_backend_output,
)


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "knm_flat": np.zeros(9, dtype=np.float64),
        "phases": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "n": 3,
        "edges_flat": np.array([0, 1, 1, 2, 0, 2], dtype=np.int64),
        "n_edges": 3,
        "tris_flat": np.array([0, 1, 2], dtype=np.int64),
        "n_tris": 1,
    }
    payload.update(overrides)
    return payload


class _ArrayFailure:
    """Object whose NumPy array protocol fails for defensive helper tests."""

    def __array__(self, dtype: object = None) -> np.ndarray:
        """Raise during array materialisation."""
        raise TypeError("array protocol failed")


class _ObjectArrayFailure:
    """Object that fails only when NumPy requests an object array."""

    def __array__(self, dtype: object = None) -> np.ndarray:
        """Return a numeric array unless object dtype is requested."""
        if dtype is not None and np.dtype(dtype) == np.dtype(object):
            raise TypeError("object array protocol failed")
        return np.array([1.0], dtype=np.float64)


class TestHodgeInputs:
    def test_alias_helpers_tolerate_failed_array_protocols(self) -> None:
        assert hodge_validation._contains_boolean_alias(_ArrayFailure()) is False
        assert hodge_validation._contains_complex_alias(_ArrayFailure()) is False
        assert hodge_validation._contains_complex_alias(_ObjectArrayFailure()) is False

    def test_numeric_string_helpers_classify_only_numeric_strings(self) -> None:
        assert hodge_validation.contains_numeric_string_alias("1.0")
        assert hodge_validation.is_numeric_string_alias(np.bytes_(b"1.0"))
        assert not hodge_validation.is_numeric_string_alias(b"\xff")
        assert not hodge_validation.is_numeric_string_alias("")
        assert not hodge_validation.is_numeric_string_alias("not-a-number")
        assert not hodge_validation.contains_numeric_string_alias(_ArrayFailure())
        assert not hodge_validation.contains_numeric_string_alias(
            np.array([1.0, 2.0], dtype=np.float64)
        )
        assert not hodge_validation.contains_numeric_string_alias(
            np.array(["not-a-number", object()], dtype=object)
        )
        assert hodge_validation.contains_numeric_string_alias(
            np.array([object(), "1.0"], dtype=object)
        )

    def test_valid_round_trips(self) -> None:
        k, p, n, edges, n_edges, tris, n_tris = validate_hodge_backend_inputs(
            **_inputs()
        )
        assert (n, n_edges, n_tris) == (3, 3, 1)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n": True}, "n must be a non-negative integer"),
            ({"n": "3"}, "n.*numeric-string"),
            ({"n": -1}, "n must be non-negative"),
            ({"knm_flat": np.zeros(4)}, "knm_flat length .* does not match"),
            (
                {"knm_flat": np.array(["0.0"] * 9, dtype=object)},
                "knm_flat.*numeric-string",
            ),
            ({"phases": np.array([0.1, 0.2])}, "phases length .* does not match"),
            (
                {"phases": np.array(["0.1", "0.2", "0.3"], dtype=object)},
                "phases.*numeric-string",
            ),
            (
                {"phases": np.array([True, False, False])},
                "phases must not contain boolean",
            ),
            ({"phases": np.array([0.1 + 1j, 0.2, 0.3])}, "phases must be real-valued"),
            (
                {"phases": np.array([0.1, np.inf, 0.3])},
                "phases must contain only finite",
            ),
            (
                {"phases": np.array(["bad", "data", "x"], dtype=object)},
                "phases must be a finite one-dimensional",
            ),
            ({"phases": np.zeros((1, 3), dtype=np.float64)}, "one-dimensional"),
            ({"n_edges": True}, "n_edges must be a non-negative integer"),
            ({"n_edges": "3"}, "n_edges.*numeric-string"),
            ({"n_edges": -1}, "n_edges must be non-negative"),
            (
                {"edges_flat": np.array(["0", "1", "1", "2", "0", "2"])},
                "edges_flat.*numeric-string",
            ),
            (
                {"edges_flat": np.array([True, False])},
                "edges_flat must not contain boolean",
            ),
            (
                {"edges_flat": np.array([0.0 + 0.0j, 1.0 + 0.0j])},
                "edges_flat must be integer-valued",
            ),
            (
                {"edges_flat": np.array(["left", "right"], dtype=object)},
                "edges_flat must be an integer index array",
            ),
            (
                {
                    "edges_flat": np.array(
                        [[0, 1], [1, 2], [0, 2]],
                        dtype=np.int64,
                    )
                },
                "edges_flat must be one-dimensional",
            ),
            ({"edges_flat": np.array([0, 1])}, "edges_flat length .* does not match"),
            (
                {"edges_flat": np.array([0, 9, 1, 2, 0, 2])},
                r"edges_flat indices must lie in \[0, 3\)",
            ),
            ({"n_tris": "1"}, "n_tris.*numeric-string"),
            (
                {"tris_flat": np.array(["0", "1", "2"])},
                "tris_flat.*numeric-string",
            ),
            ({"tris_flat": np.array([0, 1])}, "tris_flat length .* does not match"),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_hodge_backend_inputs(**_inputs(**overrides))


def _valid_output_matrix() -> np.ndarray:
    return np.array(
        [[0.0, 1.0, -0.5], [-1.0, 0.0, 0.25], [0.5, -0.25, 0.0]],
        dtype=np.float64,
    )


class TestHodgeOutputs:
    def test_valid_output_round_trips_as_contiguous_float64(self) -> None:
        matrix = _valid_output_matrix()
        gradient, curl, harmonic = validate_hodge_backend_output(
            (matrix.ravel(), matrix * 2.0, matrix * 3.0),
            n=3,
        )
        assert gradient.dtype == curl.dtype == harmonic.dtype == np.float64
        assert gradient.shape == curl.shape == harmonic.shape == (3, 3)
        assert gradient.flags.c_contiguous
        np.testing.assert_allclose(gradient, matrix)

    @pytest.mark.parametrize(
        ("output", "match"),
        [
            ("not-a-hodge-tuple", "must contain"),
            (
                (
                    np.array([[0.0, np.bool_(True)], [-1.0, 0.0]], dtype=object),
                    np.zeros((2, 2), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                ),
                "finite real-valued",
            ),
            (
                (
                    np.array([[0.0, 1.0 + 0.0j], [-1.0, 0.0]], dtype=object),
                    np.zeros((2, 2), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                ),
                "finite real-valued",
            ),
            (
                (
                    np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=np.complex128),
                    np.zeros((2, 2), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                ),
                "finite real-valued",
            ),
            (
                (
                    np.array([["bad", "value"], ["x", "y"]], dtype=object),
                    np.zeros((2, 2), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                ),
                "finite real-valued",
            ),
            (
                (
                    np.array([["0.0", "1.0"], ["-1.0", "0.0"]], dtype=object),
                    np.zeros((2, 2), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                ),
                "numeric-string",
            ),
            (
                (
                    np.zeros((1, 1), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                ),
                "invalid shape",
            ),
            (
                (
                    np.full((2, 2), np.nan, dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                ),
                "non-finite",
            ),
            (
                (
                    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                    np.zeros((2, 2), dtype=np.float64),
                ),
                "antisymmetric",
            ),
        ],
    )
    def test_rejects_corrupt_output(
        self,
        output: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_hodge_backend_output(output, n=2)
