# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend attention-residual validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_validation import (  # noqa: E501
    validate_attnres_backend_inputs,
    validate_attnres_backend_output,
)


class _RejectingArray:
    def __array__(self, dtype: Any = None) -> np.ndarray:
        raise TypeError("array protocol rejected")


class _ComplexAfterObjectArray:
    def __array__(self, dtype: Any = None) -> np.ndarray:
        if dtype is not None and np.dtype(dtype) == np.dtype(object):
            return np.array([1.0], dtype=object)
        return np.array([1.0 + 0.0j], dtype=np.complex128)


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "knm_flat": np.zeros(4, dtype=np.float64),
        "theta": np.array([0.1, 0.2], dtype=np.float64),
        "w_q": np.zeros(4, dtype=np.float64),
        "w_k": np.zeros(4, dtype=np.float64),
        "w_v": np.zeros(4, dtype=np.float64),
        "w_o": np.zeros(4, dtype=np.float64),
        "n": 2,
        "n_heads": 1,
        "block_size": 2,
        "temperature": 1.0,
        "lambda_": 0.5,
    }
    payload.update(overrides)
    return payload


class TestAttnresInputs:
    def test_valid_round_trips(self) -> None:
        result = validate_attnres_backend_inputs(**_inputs())
        assert result[6] == 2  # n

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n": True}, "n must"),
            ({"n_heads": 0}, "n_heads must"),
            ({"knm_flat": np.zeros(9)}, "knm_flat length .* does not match"),
            ({"theta": np.array([0.1])}, "theta length .* does not match"),
            ({"w_k": np.zeros(2)}, "w_q, w_k, and w_v flattened lengths must match"),
            ({"w_o": np.zeros(0)}, "w_o must contain at least one value"),
            ({"w_o": np.zeros(9)}, "even square d_model"),
            ({"n_heads": 3}, "d_model must be divisible by n_heads"),
            ({"temperature": True}, "temperature"),
            ({"lambda_": True}, "lambda_"),
            (
                {
                    "w_q": np.zeros(8),
                    "w_k": np.zeros(8),
                    "w_v": np.zeros(8),
                    "n_heads": 2,
                },
                "does not match H.d_model.d_head",
            ),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_attnres_backend_inputs(**_inputs(**overrides))


class TestAttnresOutput:
    def test_valid_round_trips(self) -> None:
        out = validate_attnres_backend_output(np.zeros(4), n=2)
        assert out.size == 4

    def test_accepts_matrix_output_and_returns_flat_vector(self) -> None:
        matrix = np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64)
        out = validate_attnres_backend_output(
            matrix,
            n=2,
            knm_flat=matrix.ravel(),
        )

        assert out.shape == (4,)
        assert out.flags.c_contiguous
        np.testing.assert_array_equal(out, matrix.ravel())

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            validate_attnres_backend_output(np.zeros(3), n=2)

    @pytest.mark.parametrize(
        ("output", "match"),
        [
            (_RejectingArray(), "finite real float array"),
            (_ComplexAfterObjectArray(), "real-valued"),
            (np.array([object()], dtype=object), "finite real float array"),
            (np.zeros((2, 3), dtype=np.float64), "shape"),
        ],
    )
    def test_rejects_uncoercible_or_wrong_shaped_output(
        self,
        output: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_attnres_backend_output(output, n=2)

    def test_rejects_reference_length_mismatch_for_zero_edge_check(self) -> None:
        output = np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64)

        with pytest.raises(ValueError, match="knm_flat length"):
            validate_attnres_backend_output(
                output,
                n=2,
                knm_flat=np.zeros(3, dtype=np.float64),
            )

    @pytest.mark.parametrize(
        ("output", "match"),
        [
            (np.array([1.0, 0.0, 0.0, 0.0]), "diagonal"),
            (np.array([0.0, 1.0, 0.0, 0.0]), "symmetric"),
        ],
    )
    def test_rejects_non_physical_coupling_output(
        self,
        output: np.ndarray,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_attnres_backend_output(output, n=2)

    def test_rejects_backend_output_that_creates_zero_edges(self) -> None:
        knm_flat = np.array(
            [
                0.0,
                0.25,
                0.0,
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        output = np.array(
            [
                0.0,
                0.25,
                0.75,
                0.25,
                0.0,
                0.0,
                0.75,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        with pytest.raises(ValueError, match="preserve zero"):
            validate_attnres_backend_output(output, n=3, knm_flat=knm_flat)
