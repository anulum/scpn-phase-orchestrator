# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — early-warning lead-time benchmark logic tests

"""Tests for the load-bearing evaluation logic of the lead-time head-to-head.

The scientific verdict rides on four pieces of pure logic: the shared alarm rule,
the matched-false-alarm calibration, the lead-time evaluation, and the verdict
gate. A bug in any of them would fake the comparison, so they are pinned here on
deterministic inputs, independent of the slow Kuramoto integration. A single
small end-to-end run confirms the physics pipeline wires together.
"""

from __future__ import annotations

import numpy as np

from bench.early_warning_leadtime import (
    THRESHOLD_GRID,
    ConditionResult,
    DetectorScore,
    _alarm_sample,
    calibrate_threshold,
    coupling_schedule,
    evaluate_leads,
    false_alarm_rate,
    order_parameter_transition,
    run_condition,
    verdict,
)


def _score(
    values: list[float],
    *,
    relative: float = 1.0,
    n_baseline: int = 0,
) -> DetectorScore:
    """Return a detector score with a constant relative gate and 10-sample hops."""
    array = np.asarray(values, dtype=np.float64)
    return DetectorScore(
        score=array,
        relative=np.full(array.shape[0], relative, dtype=np.float64),
        window_starts=np.arange(array.shape[0], dtype=np.int64) * 10,
        n_baseline=n_baseline,
    )


# --------------------------------------------------------------------------- #
# _alarm_sample — the shared alarm rule                                        #
# --------------------------------------------------------------------------- #


def test_alarm_fires_on_a_sustained_breach() -> None:
    score = _score([0.0, 0.0, 5.0, 5.0, 0.0])
    sample = _alarm_sample(score, threshold=3.0, relative_gate=0.0, persistence=2)
    assert sample == 20  # window_starts[2]


def test_alarm_silent_without_a_breach() -> None:
    score = _score([0.0, 1.0, 2.0, 1.0])
    assert _alarm_sample(score, threshold=3.0, relative_gate=0.0, persistence=2) is None


def test_alarm_requires_the_full_persistence_run() -> None:
    score = _score([0.0, 5.0, 0.0, 5.0])
    assert _alarm_sample(score, threshold=3.0, relative_gate=0.0, persistence=2) is None


def test_alarm_relative_gate_blocks_a_high_score() -> None:
    score = _score([0.0, 5.0, 5.0], relative=0.01)
    assert _alarm_sample(score, threshold=3.0, relative_gate=0.5, persistence=2) is None


def test_alarm_ignores_breaches_inside_the_baseline() -> None:
    score = _score([5.0, 5.0, 5.0, 5.0], n_baseline=2)
    sample = _alarm_sample(score, threshold=3.0, relative_gate=0.0, persistence=2)
    assert sample == 20  # first breach at or past the baseline window


# --------------------------------------------------------------------------- #
# calibrate_threshold / false_alarm_rate                                       #
# --------------------------------------------------------------------------- #


def test_calibrate_picks_the_smallest_threshold_meeting_the_target() -> None:
    nulls = [_score([0.0, 5.0, 5.0]) for _ in range(3)]
    nulls += [_score([0.0, 0.0, 0.0]) for _ in range(7)]
    # The three alarming nulls fire at any threshold up to 5.0 (0.3 false alarm);
    # the smallest grid threshold above 5.0 first meets the 0.10 target.
    threshold = calibrate_threshold(nulls, 0.10)
    assert threshold == 5.25
    assert false_alarm_rate(nulls, threshold) == 0.0
    assert false_alarm_rate(nulls, 1.0) == 0.3


def test_calibrate_falls_back_to_the_largest_threshold() -> None:
    nulls = [_score([0.0, 100.0, 100.0]) for _ in range(4)]
    # Every null alarms even at the top of the grid, so no threshold meets a
    # zero-tolerance target and the largest grid threshold is returned.
    assert calibrate_threshold(nulls, 0.0) == THRESHOLD_GRID[-1]


# --------------------------------------------------------------------------- #
# evaluate_leads                                                               #
# --------------------------------------------------------------------------- #


def test_evaluate_leads_counts_only_pre_transition_alarms() -> None:
    early = _score([0.0, 5.0, 5.0])  # alarm at sample 10, before transition 40
    late = _score([0.0, 0.0, 5.0, 5.0])  # alarm at sample 20, before transition 25
    missed = _score([0.0, 0.0, 0.0])  # never alarms
    after = _score([0.0, 0.0, 0.0, 5.0, 5.0])  # alarm at 30, after transition 25
    pairs = [(early, 40), (late, 25), (missed, 40), (after, 25)]
    rate, median, leads = evaluate_leads(pairs, threshold=3.0)
    assert leads == [30.0, 5.0]
    assert rate == 0.5
    assert median == 17.5


# --------------------------------------------------------------------------- #
# verdict                                                                      #
# --------------------------------------------------------------------------- #


def _result(
    condition: str,
    *,
    entropy_lead: float,
    slowing_lead: float,
    entropy_rate: float = 1.0,
    slowing_rate: float = 1.0,
) -> ConditionResult:
    """Return a condition result populated only where the verdict reads it."""
    return ConditionResult(
        condition=condition,
        transition_kind="first-order" if condition == "explosive" else "second-order",
        n_realisations=10,
        entropy_false_alarm=0.1,
        slowing_false_alarm=0.1,
        entropy_threshold=1.0,
        slowing_threshold=1.0,
        entropy_detection_rate=entropy_rate,
        slowing_detection_rate=slowing_rate,
        entropy_median_lead=entropy_lead,
        slowing_median_lead=slowing_lead,
        entropy_leads=[],
        slowing_leads=[],
    )


def test_verdict_names_a_win_only_in_the_explosive_niche() -> None:
    explosive = _result("explosive", entropy_lead=200.0, slowing_lead=80.0)
    continuous = _result("continuous", entropy_lead=50.0, slowing_lead=90.0)
    assert verdict(explosive, continuous).startswith("WIN")


def test_verdict_flags_a_win_on_both_as_suspect() -> None:
    explosive = _result("explosive", entropy_lead=200.0, slowing_lead=80.0)
    continuous = _result(
        "continuous", entropy_lead=200.0, slowing_lead=50.0, slowing_rate=0.5
    )
    assert verdict(explosive, continuous).startswith("SUSPECT")


def test_verdict_reports_no_win_when_the_baseline_leads_the_explosive() -> None:
    explosive = _result("explosive", entropy_lead=60.0, slowing_lead=150.0)
    continuous = _result("continuous", entropy_lead=40.0, slowing_lead=120.0)
    assert verdict(explosive, continuous).startswith("NO WIN")


def test_verdict_reports_a_trade_off_when_entropy_leads_but_detects_less() -> None:
    explosive = _result(
        "explosive",
        entropy_lead=1327.0,
        slowing_lead=453.0,
        entropy_rate=0.3,
        slowing_rate=0.8,
    )
    continuous = _result("continuous", entropy_lead=155.0, slowing_lead=115.0)
    assert verdict(explosive, continuous).startswith("TRADE-OFF")


def test_verdict_reports_no_win_without_any_leading_detection() -> None:
    explosive = _result(
        "explosive", entropy_lead=0.0, slowing_lead=453.0, entropy_rate=0.0
    )
    continuous = _result("continuous", entropy_lead=100.0, slowing_lead=90.0)
    assert verdict(explosive, continuous).startswith("NO WIN")


# --------------------------------------------------------------------------- #
# helpers + end-to-end wiring                                                  #
# --------------------------------------------------------------------------- #


def test_coupling_schedule_holds_then_ramps() -> None:
    schedule = coupling_schedule(100, 0.5, 4.0, 0.4)
    assert schedule.shape[0] == 100
    assert np.all(schedule[:40] == 0.5)
    assert schedule[40] == 0.5
    assert schedule[-1] == 4.0
    assert np.all(np.diff(schedule[40:]) > 0.0)


def test_order_parameter_transition_finds_the_upward_crossing() -> None:
    order = np.array([0.1, 0.2, 0.6, 0.9], dtype=np.float64)
    assert order_parameter_transition(order, 0.5) == 2
    assert order_parameter_transition(np.array([0.1, 0.2, 0.3]), 0.5) is None


def test_run_condition_wires_the_pipeline_end_to_end() -> None:
    result = run_condition("continuous", n=12, steps=800, n_realisations=2, seed=7)
    assert result.condition == "continuous"
    assert result.transition_kind == "second-order"
    assert 0.0 <= result.entropy_detection_rate <= 1.0
    assert 0.0 <= result.slowing_false_alarm <= 1.0


def test_run_condition_rejects_an_unknown_kind() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown transition kind"):
        run_condition("bogus", n=12, steps=400, n_realisations=1)
