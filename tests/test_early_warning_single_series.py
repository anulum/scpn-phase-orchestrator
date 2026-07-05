# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — single-series critical-slowing-down harness tests

"""Tests for the single-series critical-slowing-down lead-time harness.

The harness segments a scalar observable, calibrates the critical-slowing-down
detector to a matched false-alarm rate on a no-transition null, measures the honest
lead of its alarm against an annotated onset, and seals the alarm or silence. None
of that touches a corpus, so every path is exercised here on synthetic series — a
stationary low-variance null and a rising-variance transition whose variance ramps
towards its end — with the labelled onset placed at the segment end so any alarm is
a genuine lead. The calibrate-time trajectory and the evaluate-time seal are checked
to agree on the very same alarm, and a re-run is checked to seal a byte-identical
digest.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench.early_warning_domain import (
    Calibration,
    false_alarm_rate,
    seizure_lead_samples,
)
from bench.early_warning_single_series import (
    DETECTOR,
    SingleSeriesObservable,
    SingleSeriesResult,
    calibrate_single_series,
    critical_slowing_down_trajectory,
    evaluate_single_series,
    null_series_trials,
    single_series_verdict,
    slice_series,
)
from scpn_phase_orchestrator.assurance.early_warning_evidence import (
    EARLY_WARNING_FLAGGED,
    NO_EARLY_WARNING,
)

_WINDOW = 64
_STEP = 8
_FS = 10.0


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                           #
# --------------------------------------------------------------------------- #


def _flat_series(
    *, n_samples: int = 1200, scale: float = 0.5, fs: float = _FS, seed: int = 0
) -> SingleSeriesObservable:
    """Return a stationary low-variance null — no critical slowing down anywhere."""
    rng = np.random.default_rng(seed)
    return SingleSeriesObservable(
        series=rng.standard_normal(n_samples) * scale, sampling_rate_hz=fs
    )


def _ramp_series(
    *, n_samples: int = 1200, fs: float = _FS, seed: int = 3
) -> SingleSeriesObservable:
    """Return a transition series whose variance ramps up towards its end.

    A quiet stationary baseline gives way to a linearly growing amplitude — the
    rising variance and lag-one autocorrelation that critical slowing down reads —
    with the labelled onset placed at the series end so every window is pre-onset.
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.3, 4.5, n_samples)
    return SingleSeriesObservable(
        series=rng.standard_normal(n_samples) * ramp, sampling_rate_hz=fs
    )


# --------------------------------------------------------------------------- #
# SingleSeriesObservable — validation and normalisation                        #
# --------------------------------------------------------------------------- #


def test_observable_normalises_to_a_contiguous_float_series() -> None:
    observable = SingleSeriesObservable(series=[1, 2, 3, 4], sampling_rate_hz=2.0)
    assert observable.series.dtype == np.float64
    assert observable.series.flags["C_CONTIGUOUS"]
    assert observable.n_samples == 4
    assert observable.sampling_rate_hz == 2.0


def test_observable_rejects_a_complex_series() -> None:
    with pytest.raises(ValueError, match="real-valued"):
        SingleSeriesObservable(series=np.array([1j, 2j, 3j]), sampling_rate_hz=1.0)


def test_observable_rejects_a_non_numeric_series() -> None:
    with pytest.raises(ValueError, match="real float array"):
        SingleSeriesObservable(series=["x", "y", "z"], sampling_rate_hz=1.0)


def test_observable_rejects_a_two_dimensional_series() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        SingleSeriesObservable(series=np.zeros((2, 5)), sampling_rate_hz=1.0)


def test_observable_rejects_a_too_short_series() -> None:
    with pytest.raises(ValueError, match="at least three samples"):
        SingleSeriesObservable(series=[1.0, 2.0], sampling_rate_hz=1.0)


def test_observable_rejects_a_non_finite_sample() -> None:
    with pytest.raises(ValueError, match="finite values"):
        SingleSeriesObservable(series=[1.0, np.nan, 3.0], sampling_rate_hz=1.0)


@pytest.mark.parametrize("rate", [0.0, -1.0, np.inf, np.nan])
def test_observable_rejects_a_non_positive_or_infinite_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        SingleSeriesObservable(series=[1.0, 2.0, 3.0], sampling_rate_hz=rate)


# --------------------------------------------------------------------------- #
# slice_series / null_series_trials                                            #
# --------------------------------------------------------------------------- #


def test_slice_series_restricts_the_series() -> None:
    observable = _flat_series(n_samples=500, seed=8)
    sliced = slice_series(observable, start=100, stop=420)
    assert sliced.n_samples == 320
    assert sliced.sampling_rate_hz == observable.sampling_rate_hz
    assert np.array_equal(sliced.series, observable.series[100:420])


def test_slice_series_rejects_an_inverted_range() -> None:
    with pytest.raises(ValueError, match="must be below stop"):
        slice_series(_flat_series(n_samples=400), start=200, stop=200)


def test_slice_series_rejects_a_range_past_the_end() -> None:
    with pytest.raises(ValueError, match="exceeds the series length"):
        slice_series(_flat_series(n_samples=400), start=0, stop=500)


def test_slice_series_rejects_a_non_integer_bound() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        slice_series(_flat_series(n_samples=400), start=1.5, stop=100)


def test_slice_series_rejects_a_negative_bound() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        slice_series(_flat_series(n_samples=400), start=-1, stop=100)


def test_slice_series_rejects_a_boolean_bound() -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        slice_series(_flat_series(n_samples=400), start=True, stop=100)


def test_null_series_trials_cuts_non_overlapping_equal_trials() -> None:
    observable = _flat_series(n_samples=700, seed=10)
    trials = null_series_trials([observable], segment_samples=320)
    # 700 // 320 == 2 non-overlapping trials; the trailing 60 samples are dropped.
    assert len(trials) == 2
    assert all(trial.n_samples == 320 for trial in trials)
    assert np.array_equal(trials[0].series, observable.series[:320])
    assert np.array_equal(trials[1].series, observable.series[320:640])


def test_null_series_trials_spans_multiple_records() -> None:
    a = _flat_series(n_samples=700, seed=11)
    b = _flat_series(n_samples=340, seed=12)
    assert len(null_series_trials([a, b], segment_samples=320)) == 3


def test_null_series_trials_rejects_a_non_positive_segment() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        null_series_trials([_flat_series(n_samples=400)], segment_samples=0)


def test_null_series_trials_rejects_a_non_integer_segment() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        null_series_trials([_flat_series(n_samples=400)], segment_samples=1.5)


def test_null_series_trials_rejects_a_boolean_segment() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        null_series_trials([_flat_series(n_samples=400)], segment_samples=True)


# --------------------------------------------------------------------------- #
# critical_slowing_down_trajectory                                             #
# --------------------------------------------------------------------------- #


def test_trajectory_is_keyed_under_the_detector_on_a_window_grid() -> None:
    observable = _flat_series(seed=5)
    trajectory = critical_slowing_down_trajectory(
        observable, window=_WINDOW, step=_STEP, relative_gate=0.05
    )
    assert trajectory.name == DETECTOR
    assert trajectory.relative_gate == 0.05
    assert trajectory.n_baseline >= 1
    expected_windows = (observable.n_samples - _WINDOW) // _STEP + 1
    assert trajectory.score.shape == (expected_windows,)
    assert np.array_equal(trajectory.window_starts, np.arange(expected_windows) * _STEP)


def test_trajectory_flags_a_variance_ramp() -> None:
    trajectory = critical_slowing_down_trajectory(
        _ramp_series(seed=4), window=_WINDOW, step=_STEP
    )
    post = trajectory.score[trajectory.n_baseline :]
    # The rising variance drives the critical-slowing-down z-score well past 3.
    assert float(post.max()) > 3.0


# --------------------------------------------------------------------------- #
# calibrate_single_series                                                       #
# --------------------------------------------------------------------------- #


def test_calibrate_single_series_returns_a_matched_threshold() -> None:
    trials = null_series_trials(
        [_flat_series(n_samples=2400, seed=seed) for seed in (20, 21, 22)],
        segment_samples=800,
    )
    calibration = calibrate_single_series(
        trials, target_fa=0.2, window=_WINDOW, step=_STEP
    )
    assert isinstance(calibration, Calibration)
    assert set(calibration.thresholds) == {DETECTOR}
    assert set(calibration.achieved_false_alarm) == {DETECTOR}
    assert 0.0 <= calibration.achieved_false_alarm[DETECTOR] <= 0.2


def test_calibrate_single_series_rejects_an_empty_null() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calibrate_single_series([])


# --------------------------------------------------------------------------- #
# evaluate_single_series — sealing at the calibrated threshold                  #
# --------------------------------------------------------------------------- #


def _evaluate(
    observable: SingleSeriesObservable, *, threshold: float
) -> SingleSeriesResult:
    """Seal one transition at ``threshold`` with the onset at the series end."""
    return evaluate_single_series(
        observable,
        record_id="synthetic_ramp",
        onset_sample=observable.n_samples,
        signal_source="synthetic rising-variance fixture",
        captured_at="2008-01-01T00:00:00",
        threshold=threshold,
        observable_description="synthetic scalar critical-slowing-down observable",
        window=_WINDOW,
        step=_STEP,
    )


def test_evaluate_single_series_seals_a_leading_alarm() -> None:
    result = _evaluate(_ramp_series(seed=6), threshold=0.0)
    assert result.evidence.verdict == EARLY_WARNING_FLAGGED
    assert result.evidence.warning_triggered is True
    assert result.evidence.lead_is_early is True
    assert result.evidence.content_hash
    assert result.lead_seconds() is not None
    assert result.lead_seconds() > 0.0


def test_evaluate_single_series_seals_a_silence() -> None:
    result = _evaluate(_flat_series(seed=7), threshold=1.0e3)
    assert result.evidence.verdict == NO_EARLY_WARNING
    assert result.evidence.warning_triggered is False
    assert result.lead_seconds() is None


def test_evaluate_single_series_is_byte_reproducible() -> None:
    first = _evaluate(_ramp_series(seed=6), threshold=0.0)
    second = _evaluate(_ramp_series(seed=6), threshold=0.0)
    assert first.evidence.content_hash == second.evidence.content_hash


def test_calibrate_and_seal_agree_on_the_same_alarm() -> None:
    # Lock-step: the alarm the calibration-time trajectory finds at a threshold is
    # the alarm the evaluate-time seal records, because the detector's per-window
    # fields do not depend on the gate.
    observable = _ramp_series(seed=6)
    onset = observable.n_samples
    trajectory = critical_slowing_down_trajectory(
        observable, window=_WINDOW, step=_STEP
    )
    lead = seizure_lead_samples(
        trajectory, onset_sample=onset, threshold=2.0, persistence=2
    )
    assert lead is not None  # the ramp triggers this fixture below the onset
    result = _evaluate(observable, threshold=2.0)
    assert result.evidence.warning_sample == onset - lead


def test_evaluate_matches_the_calibrated_false_alarm_on_the_null() -> None:
    # A threshold calibrated on the flat null holds the null false-alarm rate at the
    # target the calibration reports — the trajectories the threshold was fitted on
    # alarm no more often than that rate.
    trials = null_series_trials(
        [_flat_series(n_samples=2400, seed=seed) for seed in (30, 31)],
        segment_samples=800,
    )
    calibration = calibrate_single_series(
        trials, target_fa=0.0, window=_WINDOW, step=_STEP
    )
    threshold = calibration.thresholds[DETECTOR]
    trajectories = [
        critical_slowing_down_trajectory(trial, window=_WINDOW, step=_STEP)
        for trial in trials
    ]
    achieved = false_alarm_rate(trajectories, threshold, persistence=2)
    assert achieved == 0.0
    assert achieved == calibration.achieved_false_alarm[DETECTOR]


def test_single_series_result_audit_record_round_trips() -> None:
    result = _evaluate(_ramp_series(seed=6), threshold=0.0)
    record = json.loads(json.dumps(result.to_audit_record()))
    assert record["record_id"] == "synthetic_ramp"
    assert record["onset_sample"] == _ramp_series(seed=6).n_samples
    assert record["detector"]["detector"] == DETECTOR
    assert record["detector"]["content_hash"] == result.evidence.content_hash


# --------------------------------------------------------------------------- #
# single_series_verdict                                                        #
# --------------------------------------------------------------------------- #


def test_verdict_reports_no_early_warning_when_nothing_leads() -> None:
    verdict = single_series_verdict(
        {"a": None, "b": None}, 2, noun="collapses", singular="collapse"
    )
    assert verdict.startswith("NO EARLY WARNING")
    assert "none of the 2 evaluated collapses" in verdict


def test_verdict_names_a_single_lead_in_the_singular() -> None:
    verdict = single_series_verdict(
        {"a": 120.0, "b": None}, 2, noun="collapses", singular="collapse"
    )
    assert verdict.startswith("SINGLE-INDICATOR DETECTION")
    assert "1/2 collapses" in verdict
    assert "on one collapse is evidence" in verdict
    assert "median lead 120 s" in verdict


def test_verdict_names_several_leads_in_the_plural_with_the_median() -> None:
    verdict = single_series_verdict(
        {"a": 100.0, "b": 300.0, "c": 200.0},
        3,
        noun="glaciations",
        singular="glaciation",
    )
    assert "3/3 glaciations" in verdict
    assert "on 3 glaciations is evidence" in verdict
    assert "median lead 200 s" in verdict
