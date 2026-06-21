# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — OPT-entropy backend validation guards

"""Direct guards for the shared OPT-entropy backend validation module."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _opt_entropy_validation as guard,
)


class TestFactorialAndWindowCount:
    @pytest.mark.parametrize(
        ("value", "expected"), [(0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (7, 5040)]
    )
    def test_factorial(self, value: int, expected: int) -> None:
        assert guard.factorial(value) == expected
        assert guard.factorial(value) == math.factorial(value)

    @pytest.mark.parametrize(
        ("length", "dimension", "delay", "expected"),
        [
            (10, 3, 1, 8),
            (10, 3, 2, 6),
            (10, 5, 2, 2),
            (4, 3, 2, 0),
            (3, 3, 1, 1),
            (2, 3, 1, 0),
        ],
    )
    def test_window_count(
        self, length: int, dimension: int, delay: int, expected: int
    ) -> None:
        assert guard.ordinal_window_count(length, dimension, delay) == expected


class TestSeriesValidation:
    def test_accepts_finite_one_dimensional(self) -> None:
        out = guard.validate_series_backend_input([1.0, 2.0, 3.0])
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize(
        ("series", "match"),
        [
            (np.array([True, False]), "boolean"),
            (np.array([0.0, np.bool_(True)], dtype=object), "boolean"),
            (np.array([1.0 + 1.0j]), "real-valued"),
            (np.array([0.0, 1.0j], dtype=object), "real-valued"),
            (np.array([0.0, np.nan]), "finite"),
            (np.array([0.0, np.inf]), "finite"),
            (np.zeros((2, 2)), "one-dimensional"),
            (np.array(["a", "b", "c"], dtype=object), "one-dimensional float array"),
        ],
    )
    def test_rejects_invalid(self, series: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            guard.validate_series_backend_input(series)


class TestParamValidation:
    @pytest.mark.parametrize("dimension", [2, 3, 7])
    def test_accepts_valid_dimension(self, dimension: int) -> None:
        d, tau = guard.validate_ordinal_params(dimension, 1)
        assert d == dimension and tau == 1

    @pytest.mark.parametrize("dimension", [1, 8, 0, -1, True, 3.5, "3", None])
    def test_rejects_invalid_dimension(self, dimension: Any) -> None:
        with pytest.raises(ValueError, match="dimension"):
            guard.validate_ordinal_params(dimension, 1)

    @pytest.mark.parametrize("delay", [0, -3, True, 1.5, "1", None])
    def test_rejects_invalid_delay(self, delay: Any) -> None:
        with pytest.raises(ValueError, match="delay"):
            guard.validate_ordinal_params(3, delay)


class TestExpectedReferences:
    def test_expected_codes_match_dispatcher_reference(self) -> None:
        from scpn_phase_orchestrator.monitor import opt_entropy as oe

        rng = np.random.default_rng(0)
        for _ in range(5):
            series = rng.standard_normal(200)
            d = int(rng.integers(2, 6))
            expected = guard.expected_ordinal_pattern_backend_output(series, d, 1)
            reference = oe._ordinal_codes_reference(series, d, 1)
            np.testing.assert_array_equal(expected, reference)

    def test_expected_entropy_matches_dispatcher_reference(self) -> None:
        from scpn_phase_orchestrator.monitor import opt_entropy as oe

        rng = np.random.default_rng(1)
        for _ in range(5):
            series = rng.standard_normal(300)
            d = int(rng.integers(2, 6))
            expected = guard.expected_transition_entropy_backend_output(series, d, 1)
            reference = oe._transition_entropy_reference(series, d, 1)
            assert expected == pytest.approx(reference, abs=0.0)

    def test_expected_entropy_too_few_transitions_is_zero(self) -> None:
        # One window → fewer than two patterns → no transition.
        assert (
            guard.expected_transition_entropy_backend_output(
                np.array([1.0, 2.0, 3.0]), 3, 1
            )
            == 0.0
        )

    def test_expected_entropy_single_transition_type_is_zero(self) -> None:
        # A constant series visits one ordinal pattern → one self-transition.
        assert (
            guard.expected_transition_entropy_backend_output(np.ones(50), 3, 1) == 0.0
        )


class TestOrdinalOutputGuard:
    def test_accepts_float_encoded_integers(self) -> None:
        codes = guard.validate_ordinal_pattern_backend_output(
            np.array([0.0, 5.0, 3.0]), n_windows=3, dimension=3
        )
        assert codes.tolist() == [0, 5, 3]

    @pytest.mark.parametrize(
        ("codes", "match"),
        [
            (np.array([0.0, 1.7]), "integer-valued"),
            (np.array([np.inf, 0.0]), "finite"),
            (np.array([True, False]), "booleans"),
            (np.array([0.0 + 1.0j, 1.0]), "real values"),
            (np.array([0, 1, 2], dtype=np.int64), "does not match"),
            (np.array([-1, 0], dtype=np.int64), r"\[0, "),
        ],
    )
    def test_rejects_invalid(self, codes: np.ndarray, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            guard.validate_ordinal_pattern_backend_output(
                codes, n_windows=2, dimension=3
            )


class TestEntropyOutputGuard:
    def test_clamps_within_tolerance(self) -> None:
        assert guard.validate_transition_entropy_backend_output(1.0 + 5e-13) == 1.0
        assert guard.validate_transition_entropy_backend_output(-5e-13) == 0.0

    @pytest.mark.parametrize("score", [np.nan, np.inf, -0.01, 1.01, True, 0.5 + 0.0j])
    def test_rejects_invalid(self, score: Any) -> None:
        with pytest.raises(ValueError, match="transition entropy backend output"):
            guard.validate_transition_entropy_backend_output(score)

    def test_tolerance_band_is_enforced(self) -> None:
        with pytest.raises(ValueError, match="exact reference"):
            guard.validate_transition_entropy_backend_output(
                0.5, expected=0.5 + 1e-6, atol=1e-9
            )
        assert (
            guard.validate_transition_entropy_backend_output(
                0.5, expected=0.5 + 1e-11, atol=1e-9
            )
            == 0.5
        )
