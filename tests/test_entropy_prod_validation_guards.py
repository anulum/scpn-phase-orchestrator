# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct-backend entropy-production validation guards

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor._entropy_prod_validation import (  # noqa: E501
    _contains_numeric_string_alias,
    _is_numeric_string_alias,
    validate_entropy_prod_backend_inputs,
    validate_entropy_prod_backend_output,
)

_KNM = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


class _ArrayProtocolFailure:
    """Array-like payload that rejects every NumPy coercion attempt."""

    def __array__(self, dtype: object | None = None) -> np.ndarray:
        """Raise as a hostile array-protocol implementation would."""
        raise TypeError("array protocol unavailable")


def _inputs(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phases": np.array([0.1, 0.2], dtype=np.float64),
        "omegas": np.array([0.0, 0.0], dtype=np.float64),
        "knm": _KNM,
        "alpha": 0.0,
        "dt": 0.01,
    }
    payload.update(overrides)
    return payload


def _call(**overrides: Any) -> Any:
    return validate_entropy_prod_backend_inputs(**_inputs(**overrides))


class TestEntropyProdNumericStringHelpers:
    def test_numeric_string_helpers_reject_only_numeric_string_aliases(self) -> None:
        assert _is_numeric_string_alias(1.0) is False
        assert _is_numeric_string_alias("not-a-number") is False
        assert _contains_numeric_string_alias(_ArrayProtocolFailure()) is False
        assert (
            _contains_numeric_string_alias(
                np.array([1.0, "not-a-number"], dtype=object)
            )
            is False
        )
        assert (
            _contains_numeric_string_alias(np.array([1.0, "2.0"], dtype=object)) is True
        )


class TestEntropyProdInputs:
    def test_valid_round_trips(self) -> None:
        phases, omegas, knm, alpha, dt = _call()
        assert phases.shape == (2,)

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"phases": np.array([True, False])}, "phases must not contain boolean"),
            (
                {"phases": np.array(["0.1", "0.2"], dtype=object)},
                "numeric-string",
            ),
            ({"phases": np.array([0.1 + 1j, 0.2])}, "phases must contain real-valued"),
            ({"phases": np.zeros((2, 2))}, "phases shape .* must be one-dimensional"),
            ({"phases": np.array([0.1, np.inf])}, "phases must contain only finite"),
            ({"omegas": np.array([0.0, 0.0, 0.0])}, "omegas shape .* does not match"),
            (
                {"omegas": np.array(["0.0", "0.0"], dtype=object)},
                "numeric-string",
            ),
            (
                {"knm": np.array([[True, False], [False, True]])},
                "knm must not contain boolean",
            ),
            (
                {"knm": np.array([["0.0", "1.0"], ["1.0", "0.0"]], dtype=object)},
                "numeric-string",
            ),
            ({"knm": np.array([[0.0, 1j], [1j, 0.0]])}, "knm must contain real-valued"),
            ({"knm": np.zeros((2, 3))}, "knm shape .* does not match"),
            (
                {"knm": np.array([[0.0, np.inf], [np.inf, 0.0]])},
                "knm must contain only finite",
            ),
            ({"alpha": 0.5 + 1j}, "alpha must be a finite real-valued scalar"),
            ({"alpha": "0.5"}, "numeric-string"),
            ({"alpha": "x"}, "alpha must be a finite real"),
            ({"alpha": float("inf")}, "alpha must be finite"),
            ({"dt": "0.01"}, "numeric-string"),
            ({"dt": -0.1}, "dt must be non-negative"),
        ],
    )
    def test_rejects_corrupt_input(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _call(**overrides)


class TestEntropyProdOutput:
    def test_valid_round_trips(self) -> None:
        assert validate_entropy_prod_backend_output(0.5) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (True, "must not be a boolean"),
            ("0.5", "numeric-string"),
            (np.array(0.0 + 1j), "must be real-valued"),
            (np.inf, "must be finite"),
            (-0.5, "must be non-negative"),
        ],
    )
    def test_rejects_corrupt_output(self, value: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_entropy_prod_backend_output(value)
