# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Dimension reference arithmetic against exact oracles

"""Check the shared dimension arithmetic against independent exact oracles."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor._dimension_numerics import (
    kaplan_yorke_reference,
    row_distances,
)


def _exact_kaplan_yorke(exponents: list[float]) -> float:
    """Return D_KY from exact rational prefix sums of the sorted spectrum."""
    if not exponents:
        return 0.0
    ordered = sorted((Fraction(value) for value in exponents), reverse=True)
    total = Fraction(0)
    accepted = Fraction(0)
    for index, value in enumerate(ordered):
        total += value
        if total < 0:
            return float(index + accepted / abs(value))
        accepted = total
    return float(len(ordered))


def _hypot_rows(left: np.ndarray, right: np.ndarray) -> list[float]:
    """Return per-row distances from the standard library's scaled hypot."""
    rows = np.broadcast_arrays(left, right)
    return [
        math.hypot(*(float(a) - float(b) for a, b in zip(lrow, rrow, strict=True)))
        for lrow, rrow in zip(rows[0], rows[1], strict=True)
    ]


class TestRowDistances:
    def test_matches_scaled_hypot_on_random_rows(self) -> None:
        rng = np.random.default_rng(20261029)
        left = rng.normal(size=(64, 5)) * 10.0 ** rng.integers(-3, 4, size=(64, 1))
        right = rng.normal(size=(64, 5))
        got = row_distances(left, right)
        assert got.dtype == np.float64
        np.testing.assert_allclose(got, _hypot_rows(left, right), rtol=4e-16, atol=0.0)

    def test_extreme_coordinates_do_not_overflow_through_squares(self) -> None:
        left = np.array([[1e300, 1e300], [3e-300, 4e-300]])
        right = np.zeros_like(left)
        got = row_distances(left, right)
        assert got[0] == pytest.approx(math.sqrt(2.0) * 1e300, rel=1e-15)
        assert got[1] == pytest.approx(5e-300, rel=1e-15)
        assert np.all(np.isfinite(got))

    def test_distance_beyond_float64_is_infinite(self) -> None:
        got = row_distances(np.array([[1.5e308]]), np.array([[-1.5e308]]))
        assert got.tolist() == [math.inf]

    def test_broadcasts_one_row_against_many(self) -> None:
        left = np.array([0.0, 0.0])
        right = np.array([[3.0, 4.0], [0.0, 0.0], [-6.0, 8.0]])
        assert row_distances(left, right).tolist() == [5.0, 0.0, 10.0]

    def test_zero_width_rows_have_zero_distance(self) -> None:
        got = row_distances(np.empty((3, 0)), np.empty((3, 0)))
        assert got.tolist() == [0.0, 0.0, 0.0]


class TestKaplanYorkeReference:
    @pytest.mark.parametrize(
        ("spectrum", "expected"),
        [
            ([], 0.0),
            ([0.0, 0.0, 0.0], 3.0),
            ([0.5, 0.25], 2.0),
            ([-1.0, -2.0], 0.0),
            ([0.906, 0.0, -14.572], 2.0 + 0.906 / 14.572),
        ],
    )
    def test_known_spectra(self, spectrum: list[float], expected: float) -> None:
        got = kaplan_yorke_reference(np.array(spectrum, dtype=np.float64))
        assert got == pytest.approx(expected, rel=1e-15, abs=0.0)
        assert got == pytest.approx(_exact_kaplan_yorke(spectrum), rel=1e-15, abs=0.0)

    def test_order_of_the_input_does_not_matter(self) -> None:
        spectrum = np.array([-14.572, 0.906, 0.0])
        assert kaplan_yorke_reference(spectrum) == kaplan_yorke_reference(
            spectrum[::-1].copy()
        )

    def test_matches_exact_rational_prefixes_on_random_spectra(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(200):
            size = int(rng.integers(1, 9))
            spectrum = rng.normal(size=size) * 10.0 ** float(rng.integers(-6, 7))
            got = kaplan_yorke_reference(spectrum)
            exact = _exact_kaplan_yorke(spectrum.tolist())
            assert 0.0 <= got <= size
            assert got == pytest.approx(exact, rel=2e-15, abs=1e-15)

    def test_large_cancelling_terms_match_the_exact_oracle(self) -> None:
        # D_KY is continuous in the prefix sums, so this case checks agreement
        # with exact arithmetic; it does not single out compensated summation.
        spectrum = [1e16, 1.0, 1.0, -1e16, -3.0]
        got = kaplan_yorke_reference(np.array(spectrum))
        assert got == pytest.approx(_exact_kaplan_yorke(spectrum), rel=1e-15)
