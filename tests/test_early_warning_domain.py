# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — domain-neutral matched-false-alarm harness tests

"""Tests for the domain-neutral early-warning lead-time harness.

The harness segments a recording, calibrates each detector to a matched
false-alarm rate on a no-transition null, measures the honest lead of each alarm
against an annotated onset, and seals every alarm or silence. None of that is
domain-specific, so every path is exercised here on synthetic
:class:`SuiteObservables` — coherence-rise and incoherent-null fields, and hand
-built detector trajectories for the shared alarm rule — independent of any
domain adapter or corpus.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench.early_warning_domain import (
    DEFAULT_STEP,
    DETECTORS,
    Calibration,
    DetectorTrajectory,
    PermutationSignificance,
    SeizureLeadResult,
    calibrate_detectors,
    calibrate_threshold,
    detector_trajectories,
    domain_verdict,
    evaluate_seizure,
    false_alarm_rate,
    null_trials,
    permutation_significance,
    seizure_lead_samples,
    slice_observables,
)
from scpn_phase_orchestrator.assurance.early_warning_evidence import (
    EARLY_WARNING_FLAGGED,
    NO_EARLY_WARNING,
)
from scpn_phase_orchestrator.monitor.early_warning_suite import (
    ENSEMBLE_WEIGHTED,
    SYNCHRONISATION,
    SuiteObservables,
)

_DESCRIPTIONS = dict.fromkeys(DETECTORS, "synthetic phase observable")


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                           #
# --------------------------------------------------------------------------- #


def _transition_observables(
    *,
    n_nodes: int = 6,
    n_samples: int = 320,
    rise_sample: int = 150,
    fs: float = 32.0,
    seed: int = 0,
) -> SuiteObservables:
    """Return observables whose nodes lock into coherence from ``rise_sample``.

    The baseline is incoherent (independent random phases) and, from
    ``rise_sample`` on, every node follows one shared phase with a little jitter —
    a rising-synchronisation precursor the suite should flag while the labelled
    onset is placed later.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, (n_nodes, n_samples))
    shared = rng.uniform(-np.pi, np.pi, n_samples)
    jitter = 0.05 * rng.standard_normal((n_nodes, n_samples))
    for node in range(n_nodes):
        phases[node, rise_sample:] = shared[rise_sample:] + jitter[node, rise_sample:]
    phases = np.arctan2(np.sin(phases), np.cos(phases))
    order = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return SuiteObservables(
        phases=np.ascontiguousarray(phases, dtype=np.float64),
        phase_field=np.ascontiguousarray(np.sin(phases), dtype=np.float64),
        order_parameter=np.ascontiguousarray(order, dtype=np.float64),
        sampling_rate_hz=fs,
    )


def _incoherent_observables(
    *, n_nodes: int = 6, n_samples: int = 320, fs: float = 32.0, seed: int = 1
) -> SuiteObservables:
    """Return a transition-free (incoherent throughout) null observable."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, (n_nodes, n_samples))
    order = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return SuiteObservables(
        phases=np.ascontiguousarray(phases, dtype=np.float64),
        phase_field=np.ascontiguousarray(np.sin(phases), dtype=np.float64),
        order_parameter=np.ascontiguousarray(order, dtype=np.float64),
        sampling_rate_hz=fs,
    )


def _trajectory(
    values: list[float],
    *,
    relative_gate: float = 0.0,
    n_baseline: int = 0,
    relative: list[float] | None = None,
    name: str = "probe",
) -> DetectorTrajectory:
    """Return a detector trajectory with default-hop windows for the alarm rule."""
    score = np.asarray(values, dtype=np.float64)
    gate = (
        np.ones(score.shape[0], dtype=np.float64)
        if relative is None
        else np.asarray(relative, dtype=np.float64)
    )
    return DetectorTrajectory(
        name=name,
        score=score,
        relative=gate,
        relative_gate=relative_gate,
        window_starts=np.arange(score.shape[0], dtype=np.int64) * DEFAULT_STEP,
        n_baseline=n_baseline,
    )


# --------------------------------------------------------------------------- #
# slice_observables / null_trials                                             #
# --------------------------------------------------------------------------- #


def test_slice_observables_restricts_every_field() -> None:
    observables = _incoherent_observables(n_samples=500, seed=8)
    sliced = slice_observables(observables, start=100, stop=420)
    assert sliced.n_samples == 320
    assert sliced.phases.shape == (observables.n_nodes, 320)
    assert sliced.phase_field.shape == (observables.n_nodes, 320)
    assert sliced.order_parameter.shape == (320,)
    assert sliced.sampling_rate_hz == observables.sampling_rate_hz
    assert np.array_equal(sliced.phases, observables.phases[:, 100:420])


def test_slice_observables_rejects_an_inverted_range() -> None:
    observables = _incoherent_observables(n_samples=400, seed=9)
    with pytest.raises(ValueError, match="must be below stop"):
        slice_observables(observables, start=200, stop=200)


def test_slice_observables_rejects_a_range_past_the_end() -> None:
    observables = _incoherent_observables(n_samples=400, seed=9)
    with pytest.raises(ValueError, match="exceeds the field length"):
        slice_observables(observables, start=0, stop=500)


def test_slice_observables_rejects_a_non_integer_bound() -> None:
    observables = _incoherent_observables(n_samples=400, seed=9)
    with pytest.raises(ValueError, match="non-negative integer"):
        slice_observables(observables, start=1.5, stop=100)


def test_slice_observables_rejects_a_negative_bound() -> None:
    observables = _incoherent_observables(n_samples=400, seed=9)
    with pytest.raises(ValueError, match="non-negative integer"):
        slice_observables(observables, start=-1, stop=100)


def test_slice_observables_rejects_a_boolean_bound() -> None:
    observables = _incoherent_observables(n_samples=400, seed=9)
    with pytest.raises(ValueError, match="non-negative integer"):
        slice_observables(observables, start=True, stop=100)


def test_null_trials_cuts_non_overlapping_equal_length_trials() -> None:
    observables = _incoherent_observables(n_samples=700, seed=10)
    trials = null_trials([observables], segment_samples=320)
    # 700 // 320 == 2 non-overlapping trials; the trailing 60 samples are dropped.
    assert len(trials) == 2
    assert all(trial.n_samples == 320 for trial in trials)
    assert np.array_equal(trials[0].order_parameter, observables.order_parameter[:320])
    assert np.array_equal(
        trials[1].order_parameter, observables.order_parameter[320:640]
    )


def test_null_trials_spans_multiple_recordings() -> None:
    a = _incoherent_observables(n_samples=700, seed=11)
    b = _incoherent_observables(n_samples=340, seed=12)
    trials = null_trials([a, b], segment_samples=320)
    assert len(trials) == 3  # 2 from the 700-sample record, 1 from the 340-sample one


def test_null_trials_rejects_a_non_positive_segment() -> None:
    with pytest.raises(ValueError, match="segment_samples"):
        null_trials([_incoherent_observables(n_samples=400)], segment_samples=0)


def test_null_trials_rejects_a_non_integer_segment() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        null_trials([_incoherent_observables(n_samples=400)], segment_samples=1.5)


def test_null_trials_rejects_a_boolean_segment() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        null_trials([_incoherent_observables(n_samples=400)], segment_samples=True)


# --------------------------------------------------------------------------- #
# detector_trajectories                                                        #
# --------------------------------------------------------------------------- #


def test_detector_trajectories_share_one_window_grid() -> None:
    observables = _transition_observables(seed=5)
    trajectories = detector_trajectories(observables)
    assert set(trajectories) == set(DETECTORS)
    reference = trajectories[DETECTORS[0]].window_starts
    for name in DETECTORS:
        assert np.array_equal(trajectories[name].window_starts, reference)
        assert trajectories[name].score.shape == reference.shape


def test_detector_trajectories_flag_a_coherence_rise() -> None:
    observables = _transition_observables(seed=6)
    trajectories = detector_trajectories(observables)
    synchrony = trajectories[SYNCHRONISATION]
    # The post-baseline coherence lock drives the synchrony z-score up.
    post = synchrony.score[synchrony.n_baseline :]
    assert float(post.max()) > 3.0


# --------------------------------------------------------------------------- #
# shared alarm rule, calibration, lead                                         #
# --------------------------------------------------------------------------- #


def test_alarm_fires_on_a_sustained_post_baseline_breach() -> None:
    lead = seizure_lead_samples(
        _trajectory([0.0, 0.0, 5.0, 5.0, 0.0]),
        onset_sample=100,
        threshold=3.0,
        persistence=2,
    )
    assert lead == 100 - DEFAULT_STEP * 2  # alarm at window_starts[2] = 32


def test_alarm_requires_the_full_persistence_run() -> None:
    lead = seizure_lead_samples(
        _trajectory([0.0, 5.0, 0.0, 5.0]),
        onset_sample=100,
        threshold=3.0,
        persistence=2,
    )
    assert lead is None


def test_alarm_relative_gate_blocks_a_high_score() -> None:
    trajectory = _trajectory(
        [0.0, 5.0, 5.0], relative=[0.0, 0.01, 0.01], relative_gate=0.5
    )
    assert (
        seizure_lead_samples(trajectory, onset_sample=100, threshold=3.0, persistence=2)
        is None
    )


def test_alarm_ignores_breaches_inside_the_baseline() -> None:
    trajectory = _trajectory([5.0, 5.0, 5.0, 5.0], n_baseline=2)
    lead = seizure_lead_samples(
        trajectory, onset_sample=100, threshold=3.0, persistence=2
    )
    assert lead == 100 - DEFAULT_STEP * 2  # first breach at the first post-baseline win


def test_lead_is_none_when_the_alarm_follows_the_onset() -> None:
    lead = seizure_lead_samples(
        _trajectory([0.0, 0.0, 5.0, 5.0]),
        onset_sample=10,  # earlier than the alarm at sample 32
        threshold=3.0,
        persistence=2,
    )
    assert lead is None


def test_false_alarm_rate_counts_alarming_nulls() -> None:
    nulls = [_trajectory([0.0, 5.0, 5.0]) for _ in range(3)]
    nulls += [_trajectory([0.0, 0.0, 0.0]) for _ in range(7)]
    assert false_alarm_rate(nulls, 3.0, persistence=2) == pytest.approx(0.3)


def test_false_alarm_rate_rejects_an_empty_null() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        false_alarm_rate([], 3.0)


def test_calibrate_threshold_matches_the_target_exactly() -> None:
    # Three nulls alarm at score 5, seven never; the tightest threshold holding the
    # rate at or below 10 % (of 10 nulls, one may alarm — but the three are tied, so
    # the conservative choice sits just above 5 and none of them alarm).
    nulls = [_trajectory([0.0, 5.0, 5.0]) for _ in range(3)]
    nulls += [_trajectory([0.0, 0.0, 0.0]) for _ in range(7)]
    threshold = calibrate_threshold(nulls, target_fa=0.1, persistence=2)
    assert threshold > 5.0
    assert threshold == pytest.approx(5.0)  # just above 5, no grid rounding
    assert false_alarm_rate(nulls, threshold, persistence=2) == 0.0
    assert false_alarm_rate(nulls, 5.0, persistence=2) == pytest.approx(0.3)


def test_calibrate_threshold_has_no_ceiling() -> None:
    # Every null alarms only at a score of 100 — far above any fixed grid maximum.
    # The continuous calibration places the threshold just above it, not at a clip.
    nulls = [_trajectory([0.0, 100.0, 100.0]) for _ in range(4)]
    threshold = calibrate_threshold(nulls, target_fa=0.0, persistence=2)
    assert threshold > 100.0
    assert false_alarm_rate(nulls, threshold, persistence=2) == 0.0


def test_calibrate_threshold_opens_the_gate_at_unit_target() -> None:
    # Allowing every null to alarm collapses to the tightest valid gate, 0.
    nulls = [_trajectory([0.0, 5.0, 5.0]) for _ in range(3)]
    assert calibrate_threshold(nulls, target_fa=1.0, persistence=2) == 0.0


def test_calibrate_threshold_is_zero_when_no_null_can_alarm() -> None:
    # Every window sits inside the baseline, so no null can alarm at any threshold;
    # the matched gate is the detectors' tightest valid value, 0.
    nulls = [_trajectory([5.0, 5.0, 5.0], n_baseline=3) for _ in range(3)]
    assert calibrate_threshold(nulls, target_fa=0.0, persistence=2) == 0.0


def test_calibrate_threshold_rejects_an_empty_null() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calibrate_threshold([])


def test_calibrate_detectors_returns_thresholds_and_achieved_rate() -> None:
    nulls = [_incoherent_observables(seed=seed) for seed in (10, 11, 12)]
    calibration = calibrate_detectors(nulls, target_fa=0.5)
    assert isinstance(calibration, Calibration)
    assert set(calibration.thresholds) == set(DETECTORS)
    assert set(calibration.achieved_false_alarm) == set(DETECTORS)
    for rate in calibration.achieved_false_alarm.values():
        assert 0.0 <= rate <= 0.5  # held at or below the target


def test_calibrate_detectors_rejects_an_empty_null() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calibrate_detectors([])


# --------------------------------------------------------------------------- #
# permutation_significance                                                     #
# --------------------------------------------------------------------------- #


def test_permutation_flags_a_detector_that_beats_the_null() -> None:
    # Three transitions all alarm; seven nulls never do — drawing all three alarming
    # slots into the transition label is rare, so the p-value is small.
    transitions = [_trajectory([0.0, 5.0, 5.0]) for _ in range(3)]
    nulls = [_trajectory([0.0, 0.0, 0.0]) for _ in range(7)]
    result = permutation_significance(
        transitions, nulls, threshold=3.0, n_permutations=5000, seed=0
    )
    assert isinstance(result, PermutationSignificance)
    assert result.observed_led == 3
    assert result.n_transitions == 3
    assert result.pooled_alarm_rate == pytest.approx(0.3)
    assert result.expected_led == pytest.approx(0.9)
    assert result.p_value < 0.05  # the lead count beats the matched false alarm


def test_permutation_does_not_flag_chance_level_detection() -> None:
    # Transitions and nulls alarm identically, so the observed count is exactly what a
    # relabelling gives — the p-value is not small.
    transitions = [_trajectory([0.0, 5.0, 5.0])]
    nulls = [_trajectory([0.0, 5.0, 5.0]) for _ in range(5)]
    result = permutation_significance(
        transitions, nulls, threshold=3.0, n_permutations=5000, seed=0
    )
    assert result.observed_led == 1
    assert result.pooled_alarm_rate == pytest.approx(1.0)
    assert result.p_value == pytest.approx(1.0)  # every relabelling reaches the count


def test_permutation_is_reproducible_under_a_fixed_seed() -> None:
    transitions = [_trajectory([0.0, 5.0, 5.0]) for _ in range(2)]
    nulls = [_trajectory([0.0, 0.0, 0.0]) for _ in range(6)]
    first = permutation_significance(transitions, nulls, threshold=3.0, seed=7)
    second = permutation_significance(transitions, nulls, threshold=3.0, seed=7)
    assert first.p_value == second.p_value


def test_permutation_audit_record_round_trips() -> None:
    transitions = [_trajectory([0.0, 5.0, 5.0])]
    nulls = [_trajectory([0.0, 0.0, 0.0]) for _ in range(3)]
    record = json.loads(
        json.dumps(
            permutation_significance(
                transitions, nulls, threshold=3.0, n_permutations=100
            ).to_audit_record()
        )
    )
    assert record["observed_led"] == 1
    assert record["n_permutations"] == 100
    assert 0.0 < record["p_value"] <= 1.0


def test_permutation_rejects_an_empty_transition_set() -> None:
    with pytest.raises(ValueError, match="transition_trajectories must not be empty"):
        permutation_significance([], [_trajectory([0.0, 0.0, 0.0])], threshold=3.0)


def test_permutation_rejects_an_empty_null_set() -> None:
    with pytest.raises(ValueError, match="null_trajectories must not be empty"):
        permutation_significance([_trajectory([0.0, 5.0, 5.0])], [], threshold=3.0)


def test_permutation_rejects_a_non_positive_permutation_count() -> None:
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_significance(
            [_trajectory([0.0, 5.0, 5.0])],
            [_trajectory([0.0, 0.0, 0.0])],
            threshold=3.0,
            n_permutations=0,
        )


# --------------------------------------------------------------------------- #
# evaluate_seizure — sealing at the calibrated thresholds                      #
# --------------------------------------------------------------------------- #


def _evaluate(thresholds: dict[str, float], *, onset_sample: int) -> SeizureLeadResult:
    observables = _transition_observables(rise_sample=150, n_samples=320, seed=7)
    return evaluate_seizure(
        observables,
        record_id="synthetic_01",
        onset_sample=onset_sample,
        signal_source="synthetic coherence-rise fixture",
        captured_at="2009-01-01T12:00:00",
        thresholds=thresholds,
        observable_descriptions=_DESCRIPTIONS,
    )


def test_evaluate_seizure_seals_every_detector_with_a_leading_alarm() -> None:
    thresholds = dict.fromkeys(DETECTORS, 0.0)
    result = _evaluate(thresholds, onset_sample=300)
    assert set(result.evidences) == set(DETECTORS)
    synchrony = result.evidences[SYNCHRONISATION]
    assert synchrony.verdict == EARLY_WARNING_FLAGGED
    assert synchrony.warning_triggered is True
    assert synchrony.lead_is_early is True
    assert synchrony.content_hash  # sealed
    # The synchrony lead is reported in seconds against the 32 Hz analysis rate.
    assert result.lead_seconds()[SYNCHRONISATION] is not None


def test_evaluate_seizure_seals_a_silence_when_no_detector_fires() -> None:
    thresholds = dict.fromkeys(DETECTORS, 1.0e3)
    result = _evaluate(thresholds, onset_sample=300)
    for name in DETECTORS:
        evidence = result.evidences[name]
        assert evidence.verdict == NO_EARLY_WARNING
        assert evidence.warning_triggered is False
        assert result.lead_seconds()[name] is None


def test_evaluate_seizure_requires_every_observable_description() -> None:
    thresholds = dict.fromkeys(DETECTORS, 0.0)
    observables = _transition_observables(rise_sample=150, n_samples=320, seed=7)
    with pytest.raises(KeyError):
        evaluate_seizure(
            observables,
            record_id="synthetic_01",
            onset_sample=300,
            signal_source="synthetic coherence-rise fixture",
            captured_at="2009-01-01T12:00:00",
            thresholds=thresholds,
            observable_descriptions={SYNCHRONISATION: "only one described"},
        )


def test_seizure_lead_result_audit_record_round_trips() -> None:
    result = _evaluate(dict.fromkeys(DETECTORS, 0.0), onset_sample=300)
    record = result.to_audit_record()
    encoded = json.loads(json.dumps(record))
    assert encoded["record_id"] == "synthetic_01"
    assert encoded["onset_sample"] == 300
    assert set(encoded["detectors"]) == set(DETECTORS)
    assert encoded["detectors"][SYNCHRONISATION]["content_hash"]


# --------------------------------------------------------------------------- #
# domain_verdict                                                              #
# --------------------------------------------------------------------------- #


def test_verdict_names_a_fusion_advantage_only_on_more_detections() -> None:
    # The fusion leads three transitions, more than any single member (one each).
    leads = {
        "critical_slowing_down": [10.0],
        "synchronisation": [20.0],
        "transition_entropy": [],
        ENSEMBLE_WEIGHTED: [30.0, 40.0, 50.0],
    }
    assert domain_verdict(leads, 6).startswith("FUSION DETECTS MORE")


def test_verdict_calls_a_single_lead_not_a_robust_advantage() -> None:
    # The real chb01 shape: fusion and synchronisation each lead one transition,
    # the fusion by a longer lead — a longer lead on n=1 is not a robust advantage.
    leads = {
        "critical_slowing_down": [],
        "synchronisation": [441.5],
        "transition_entropy": [],
        ENSEMBLE_WEIGHTED: [450.5],
    }
    assert domain_verdict(leads, 6).startswith("SPARSE DETECTION, NO ROBUST ADVANTAGE")


def test_verdict_reports_no_early_warning_when_nothing_leads() -> None:
    leads: dict[str, list[float]] = {name: [] for name in DETECTORS}
    assert domain_verdict(leads, 6).startswith("NO EARLY WARNING")


def test_verdict_uses_the_domain_noun() -> None:
    leads = {
        "critical_slowing_down": [],
        "synchronisation": [441.5],
        "transition_entropy": [],
        ENSEMBLE_WEIGHTED: [450.5],
    }
    verdict = domain_verdict(leads, 6, noun="seizures", singular="seizure")
    assert "more seizures than the fusion" in verdict
    assert "one seizure is not a robust advantage" in verdict
