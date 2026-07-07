# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — early-warning skill primitive tests

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.evaluation.skill import (
    PermutationSignificance,
    calibrate_score_threshold,
    matched_false_alarm_rate,
    permutation_significance_from_alarms,
    surrogate_rank_pvalue,
)


class TestCalibrateScoreThreshold:
    def test_holds_target_false_alarm_exactly(self):
        nulls = list(range(10))  # 0..9
        threshold = calibrate_score_threshold(nulls, target_fa=0.10)
        # Exactly one null (score 9) may exceed the threshold at fa = 0.1.
        assert matched_false_alarm_rate(nulls, threshold) == pytest.approx(0.10)
        assert threshold > 8.0

    def test_zero_target_admits_no_null(self):
        nulls = list(range(10))
        threshold = calibrate_score_threshold(nulls, target_fa=0.0)
        assert matched_false_alarm_rate(nulls, threshold) == 0.0
        assert threshold > 9.0

    def test_full_budget_opens_gate_to_minus_infinity(self):
        threshold = calibrate_score_threshold([0.0, 1.0, 2.0], target_fa=1.0)
        assert threshold == float("-inf")

    def test_accepts_numpy_array_and_negative_scores(self):
        nulls = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
        threshold = calibrate_score_threshold(nulls, target_fa=0.20)
        # A falling-trend statistic may legitimately be negative; no clamp to 0.
        assert matched_false_alarm_rate(nulls, threshold) == pytest.approx(0.20)

    def test_empty_null_scores_rejected(self):
        with pytest.raises(ValueError, match="null_scores must not be empty"):
            calibrate_score_threshold([])

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
    def test_target_fa_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError, match="target_fa must be in"):
            calibrate_score_threshold([0.0, 1.0], target_fa=bad)


class TestMatchedFalseAlarmRate:
    def test_fraction_at_or_above_threshold(self):
        assert matched_false_alarm_rate([0.0, 1.0, 2.0, 3.0], 2.0) == 0.5

    def test_boundary_is_inclusive(self):
        assert matched_false_alarm_rate([1.0, 1.0], 1.0) == 1.0

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="null_scores must not be empty"):
            matched_false_alarm_rate([], 0.0)


class TestPermutationSignificanceFromAlarms:
    def test_all_events_alarm_beats_silent_null(self):
        result = permutation_significance_from_alarms(
            [True, True, True, True],
            [False] * 40,
            n_permutations=2000,
            seed=0,
        )
        assert result.observed_alarms == 4
        assert result.p_value < 0.05
        assert result.p_value > 0.0  # add-one correction never zero

    def test_no_event_alarm_is_certain_under_null(self):
        # observed = 0, every relabelling reaches >= 0, so p = 1 exactly.
        result = permutation_significance_from_alarms(
            [False, False],
            [False, False, False],
            n_permutations=500,
            seed=3,
        )
        assert result.observed_alarms == 0
        assert result.expected_alarms == 0.0
        assert result.pooled_alarm_rate == 0.0
        assert result.p_value == 1.0

    def test_pooled_rate_and_expectation(self):
        result = permutation_significance_from_alarms(
            [True, True],
            [False, False, False],
            n_permutations=100,
            seed=1,
        )
        assert result.pooled_alarm_rate == pytest.approx(0.4)
        assert result.expected_alarms == pytest.approx(0.8)
        assert result.n_events == 2

    def test_reproducible_under_fixed_seed(self):
        args = ([True, False, True], [False, True, False, False])
        a = permutation_significance_from_alarms(*args, n_permutations=500, seed=7)
        b = permutation_significance_from_alarms(*args, n_permutations=500, seed=7)
        assert a.p_value == b.p_value

    def test_to_record_round_trips_fields(self):
        result = permutation_significance_from_alarms(
            [True], [False, False], n_permutations=10, seed=0
        )
        record = result.to_record()
        assert set(record) == {
            "observed_alarms",
            "n_events",
            "pooled_alarm_rate",
            "expected_alarms",
            "p_value",
            "n_permutations",
            "seed",
        }

    def test_empty_event_alarms_rejected(self):
        with pytest.raises(ValueError, match="event_alarms must not be empty"):
            permutation_significance_from_alarms([], [False])

    def test_empty_null_alarms_rejected(self):
        with pytest.raises(ValueError, match="null_alarms must not be empty"):
            permutation_significance_from_alarms([True], [])

    @pytest.mark.parametrize("bad", [0, -5, True, 1.5, "many"])
    def test_non_positive_permutations_rejected(self, bad):
        with pytest.raises(ValueError, match="n_permutations must be a positive"):
            permutation_significance_from_alarms([True], [False], n_permutations=bad)

    def test_is_frozen_dataclass(self):
        result = permutation_significance_from_alarms([True], [False])
        assert isinstance(result, PermutationSignificance)
        with pytest.raises(AttributeError):
            # Deliberate frozen-attribute write to assert immutability at runtime.
            result.p_value = 0.0  # type: ignore[misc]


class TestSurrogateRankPValue:
    def test_extreme_observation_is_significant(self):
        assert surrogate_rank_pvalue(10.0, [0.0, 1.0, 2.0]) == pytest.approx(0.25)

    def test_add_one_correction_never_zero(self):
        assert surrogate_rank_pvalue(100.0, [1.0, 2.0]) == pytest.approx(1 / 3)

    def test_ties_count_towards_reaching(self):
        assert surrogate_rank_pvalue(2.0, [2.0, 2.0, 1.0]) == pytest.approx(3 / 4)

    def test_accepts_numpy_surrogates(self):
        value = surrogate_rank_pvalue(0.5, np.array([0.0, 1.0]))
        assert value == pytest.approx(2 / 3)

    def test_empty_surrogates_rejected(self):
        with pytest.raises(ValueError, match="surrogates must not be empty"):
            surrogate_rank_pvalue(1.0, [])

    def test_never_exceeds_one(self):
        assert surrogate_rank_pvalue(-math.inf, [0.0]) == 1.0
