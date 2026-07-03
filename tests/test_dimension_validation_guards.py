# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend dimension input validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor._dimension_validation import (  # noqa: E501
    expected_correlation_integral_backend_output,
    validate_correlation_integral_backend_inputs,
    validate_correlation_integral_backend_output,
    validate_kaplan_yorke_backend_input,
    validate_kaplan_yorke_backend_output,
)


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "traj_flat": np.array([0.0, 1.0], dtype=np.float64),
        "t": 2,
        "d": 1,
        "idx_i": np.array([0], dtype=np.int64),
        "idx_j": np.array([1], dtype=np.int64),
        "epsilons": np.array([0.5, 1.0], dtype=np.float64),
    }
    payload.update(overrides)
    return payload


def _call(**overrides: Any) -> Any:
    return validate_correlation_integral_backend_inputs(**_inputs(**overrides))


class TestCorrelationIntegralInputs:
    def test_valid_round_trips(self) -> None:
        traj, t, d, ii, jj, eps = _call()
        assert (t, d) == (2, 1)
        assert ii.tolist() == [0]

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"t": True}, "t must be an integer"),
            ({"t": -1}, "t must be >= 0"),
            ({"d": 0}, "d must be >= 1"),
            ({"traj_flat": np.array([True, False])}, "must not contain boolean"),
            ({"traj_flat": np.array([0.0 + 1j, 1.0])}, "must contain real values"),
            (
                {"traj_flat": np.array(["0.0", "1.0"])},
                "numeric-string",
            ),
            (
                {"traj_flat": np.array([0.0, "1.0"], dtype=object)},
                "numeric-string",
            ),
            (
                {"traj_flat": np.array([["a"], ["b"]], dtype=object)},
                "finite one-dimensional float array",
            ),
            ({"traj_flat": np.zeros((2, 1))}, "traj_flat must be one-dimensional"),
            ({"traj_flat": np.array([0.0, np.inf])}, "only finite values"),
            ({"traj_flat": np.array([0.0, 1.0, 2.0])}, "does not match t.d"),
            ({"idx_i": np.array([True])}, "must not contain boolean"),
            ({"idx_i": np.array(["0"])}, "numeric-string"),
            ({"idx_i": np.array([0.5])}, "must contain integer indices"),
            ({"idx_i": np.zeros((1, 1))}, "idx_i must be one-dimensional"),
            ({"idx_i": np.array([5])}, r"indices must lie in \[0, 2\)"),
            ({"idx_j": np.array([0, 1])}, "must have the same length"),
            ({"idx_j": np.array([0])}, "must not describe self-pairs"),
            ({"epsilons": np.array(["0.5", "1.0"])}, "numeric-string"),
            ({"epsilons": np.array([-0.5, 1.0])}, "non-negative"),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _call(**overrides)

    def test_rejects_non_empty_index_when_t_is_zero(self) -> None:
        with pytest.raises(ValueError, match="must be empty when t is zero"):
            validate_correlation_integral_backend_inputs(
                traj_flat=np.array([], dtype=np.float64),
                t=0,
                d=1,
                idx_i=np.array([0], dtype=np.int64),
                idx_j=np.array([0], dtype=np.int64),
                epsilons=np.array([0.5], dtype=np.float64),
            )


class TestKaplanYorkeInput:
    def test_valid_round_trips(self) -> None:
        out = validate_kaplan_yorke_backend_input(np.array([1.0, -2.0]))
        assert out.tolist() == [1.0, -2.0]

    def test_rejects_two_dimensional(self) -> None:
        with pytest.raises(ValueError, match="must be one-dimensional"):
            validate_kaplan_yorke_backend_input(np.zeros((2, 2)))

    @pytest.mark.parametrize(
        "spectrum",
        [
            np.array(["1.0", "-2.0"]),
            np.array([1.0, "-2.0"], dtype=object),
        ],
    )
    def test_rejects_numeric_string_spectrum(self, spectrum: np.ndarray) -> None:
        with pytest.raises(ValueError, match="numeric-string"):
            validate_kaplan_yorke_backend_input(spectrum)


class TestCorrelationIntegralOutput:
    def test_rejects_numeric_string_values(self) -> None:
        with pytest.raises(ValueError, match="numeric-string"):
            validate_correlation_integral_backend_output(
                np.array(["0.0", "0.5"]),
                np.array([0.1, 0.2]),
            )

    def test_rejects_numeric_string_expected_values(self) -> None:
        with pytest.raises(ValueError, match="expected.*numeric-string"):
            validate_correlation_integral_backend_output(
                np.array([0.0, 0.5]),
                np.array([0.1, 0.2]),
                expected=np.array(["0.0", "0.5"]),
            )


class TestKaplanYorkeOutput:
    def test_rejects_numeric_string_value(self) -> None:
        with pytest.raises(ValueError, match="numeric-string"):
            validate_kaplan_yorke_backend_output(
                "1.0",
                np.array([0.2, 0.0, -0.5]),
            )

    def test_rejects_numeric_string_expected_value(self) -> None:
        with pytest.raises(ValueError, match="expected.*numeric-string"):
            validate_kaplan_yorke_backend_output(
                1.0,
                np.array([0.2, 0.0, -0.5]),
                expected="1.0",
            )


class TestExpectedCorrelationIntegralOutput:
    def test_empty_index_yields_zero_vector(self) -> None:
        out = expected_correlation_integral_backend_output(
            np.array([0.0, 1.0], dtype=np.float64),
            2,
            1,
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([0.5, 1.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(out, np.zeros(2))
