# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — shared non-negative validator tests

"""Contracts for the canonical shared non-negative scalar validators."""

from __future__ import annotations

import math

import numpy as np
import pytest

import scpn_phase_orchestrator._validation as validation_module
from scpn_phase_orchestrator._validation import non_negative_int, non_negative_real


class TestNonNegativeReal:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0.0),
            (0.0, 0.0),
            (2.5, 2.5),
            (np.float64(1.25), 1.25),
            (np.int32(3), 3.0),
        ],
    )
    def test_accepts_non_negative_reals(self, value: object, expected: float) -> None:
        result = non_negative_real(value, name="knob")
        assert isinstance(result, float)
        assert result == expected

    @pytest.mark.parametrize(
        "value",
        [True, False, np.bool_(True), "1.0", None, [1.0], 1 + 2j],
    )
    def test_rejects_non_real_and_boolean_aliases(self, value: object) -> None:
        with pytest.raises(ValueError, match="knob must be finite and non-negative"):
            non_negative_real(value, name="knob")

    @pytest.mark.parametrize(
        "value",
        [-1.0, -1e-12, math.inf, -math.inf, math.nan, np.float64("nan")],
    )
    def test_rejects_negative_and_non_finite(self, value: object) -> None:
        with pytest.raises(ValueError, match="knob must be finite and non-negative"):
            non_negative_real(value, name="knob")

    def test_error_context_carries_the_parameter_name(self) -> None:
        with pytest.raises(ValueError, match="drive_gain"):
            non_negative_real(-0.5, name="drive_gain")


class TestNonNegativeInt:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0),
            (7, 7),
            (np.int64(11), 11),
        ],
    )
    def test_accepts_non_negative_integers(self, value: object, expected: int) -> None:
        result = non_negative_int(value, name="steps")
        assert isinstance(result, int)
        assert result == expected

    @pytest.mark.parametrize(
        "value",
        [True, False, np.bool_(False), 1.0, "3", None, -1, np.int64(-4)],
    )
    def test_rejects_booleans_non_integers_and_negatives(self, value: object) -> None:
        with pytest.raises(ValueError, match="steps must be a non-negative integer"):
            non_negative_int(value, name="steps")

    def test_error_context_carries_the_parameter_name(self) -> None:
        with pytest.raises(ValueError, match="horizon_steps"):
            non_negative_int(-2, name="horizon_steps")


def test_module_exports_exactly_the_two_validators() -> None:
    assert set(validation_module.__all__) == {
        "non_negative_int",
        "non_negative_real",
    }
