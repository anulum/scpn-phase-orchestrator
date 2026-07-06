# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spectral seizure-specific detector tests

"""Tests for the domain-specific scalp-EEG seizure detector.

Band power, the beta-to-delta ratio trajectory, the rising-trend score, and the
matched-false-alarm significance are exercised on synthetic multichannel signals with a
planted rising-beta preictal build-up, alongside every guard and boundary. The detector
is shown to separate a genuine preictal beta rise from flat interictal activity at a
matched false-alarm rate — the behaviour the honest real-data test then probes.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.seizure_detector import (
    BETA_BAND,
    DELTA_BAND,
    SeizureSignificance,
    _positive_int,
    band_power,
    beta_delta_ratio_trajectory,
    channel_ratio_trajectories,
    segment_rise_score,
    seizure_significance,
    spectral_rise_score,
)

_RATE = 256.0


def _channel(rng: np.random.Generator, *, beta_gain: float, samples: int) -> np.ndarray:
    """Return one channel: a steady 2 Hz delta plus a scaled 20 Hz beta and noise."""
    time = np.arange(samples) / _RATE
    delta = np.sin(2.0 * np.pi * 2.0 * time)
    beta = beta_gain * np.sin(2.0 * np.pi * 20.0 * time + rng.uniform(0.0, 6.0))
    return delta + beta + 0.3 * rng.standard_normal(samples)


def _ramped(rng: np.random.Generator, ramp: np.ndarray) -> np.ndarray:
    """Return a channel whose beta gain follows ``ramp`` sample by sample."""
    samples = ramp.shape[0]
    time = np.arange(samples) / _RATE
    delta = np.sin(2.0 * np.pi * 2.0 * time)
    beta = ramp * np.sin(2.0 * np.pi * 20.0 * time + rng.uniform(0.0, 6.0))
    return delta + beta + 0.3 * rng.standard_normal(samples)


def _preictal(
    rng: np.random.Generator, *, channels: int = 3, seconds: int = 16
) -> np.ndarray:
    """Return a signal whose beta amplitude ramps up toward the end (a preictal rise).

    The beta gain follows a linear ramp, so the beta/delta ratio climbs across the run.
    """
    samples = int(_RATE * seconds)
    ramp = np.linspace(0.2, 2.2, samples)
    return np.stack([_ramped(rng, ramp) for _ in range(channels)])


def _interictal(
    rng: np.random.Generator, *, channels: int = 3, seconds: int = 16
) -> np.ndarray:
    """Return a signal with a steady, non-rising beta amplitude."""
    samples = int(_RATE * seconds)
    return np.stack(
        [_channel(rng, beta_gain=0.8, samples=samples) for _ in range(channels)]
    )


# --------------------------------------------------------------------------- #
# band_power                                                                   #
# --------------------------------------------------------------------------- #


def test_band_power_is_positive_inside_the_band() -> None:
    rng = np.random.default_rng(0)
    window = _channel(rng, beta_gain=1.5, samples=1024)
    assert band_power(window, rate=_RATE, band=BETA_BAND) > 0.0


def test_band_power_is_zero_for_a_band_beyond_nyquist() -> None:
    rng = np.random.default_rng(1)
    window = _channel(rng, beta_gain=1.0, samples=1024)
    # 200–300 Hz lies above the 128 Hz Nyquist, so no bin falls in the band.
    assert band_power(window, rate=_RATE, band=(200.0, 300.0)) == 0.0


def test_band_power_rejects_a_non_vector() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        band_power(np.zeros((2, 8)), rate=_RATE, band=DELTA_BAND)


def test_band_power_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        band_power(np.zeros(1), rate=_RATE, band=DELTA_BAND)


def test_band_power_rejects_a_non_positive_rate() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        band_power(np.zeros(8), rate=0.0, band=DELTA_BAND)


# --------------------------------------------------------------------------- #
# beta_delta_ratio_trajectory                                                 #
# --------------------------------------------------------------------------- #


def test_ratio_trajectory_rises_for_a_preictal_build_up() -> None:
    rng = np.random.default_rng(2)
    trajectory = beta_delta_ratio_trajectory(
        _preictal(rng), rate=_RATE, window=1024, step=256
    )
    assert trajectory.shape[0] > 2
    assert trajectory[-1] > trajectory[0]  # beta power grows toward the seizure


def test_channel_ratio_trajectories_are_per_channel() -> None:
    rng = np.random.default_rng(20)
    trajectories = channel_ratio_trajectories(
        _preictal(rng, channels=4), rate=_RATE, window=1024, step=256
    )
    assert trajectories.shape[0] == 4  # one trajectory per channel
    assert trajectories.shape[1] > 2


def test_ratio_trajectory_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        beta_delta_ratio_trajectory(np.zeros(64), rate=_RATE, window=32, step=8)


def test_ratio_trajectory_rejects_no_channels() -> None:
    with pytest.raises(ValueError, match="at least one channel"):
        beta_delta_ratio_trajectory(np.zeros((0, 64)), rate=_RATE, window=32, step=8)


def test_ratio_trajectory_rejects_a_window_that_does_not_fit() -> None:
    with pytest.raises(ValueError, match=r"window .* must lie in"):
        beta_delta_ratio_trajectory(np.zeros((2, 32)), rate=_RATE, window=64, step=8)


def test_ratio_trajectory_rejects_a_non_integer_step() -> None:
    with pytest.raises(ValueError, match="step must be a positive integer"):
        beta_delta_ratio_trajectory(np.zeros((2, 64)), rate=_RATE, window=32, step=2.5)


# --------------------------------------------------------------------------- #
# spectral_rise_score                                                         #
# --------------------------------------------------------------------------- #


def test_rise_score_is_one_for_a_monotone_rise() -> None:
    assert spectral_rise_score([0.1, 0.3, 0.6, 1.0]) == pytest.approx(1.0)


def test_rise_score_is_zero_below_two_windows() -> None:
    assert spectral_rise_score([0.5]) == 0.0


def test_rise_score_is_zero_for_a_constant_trajectory() -> None:
    assert spectral_rise_score([2.0, 2.0, 2.0]) == 0.0


def test_rise_score_rejects_a_non_vector() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        spectral_rise_score([[1.0, 2.0], [3.0, 4.0]])


# --------------------------------------------------------------------------- #
# segment_rise_score                                                          #
# --------------------------------------------------------------------------- #


def test_segment_rise_score_mean_scores_the_whole_head() -> None:
    rng = np.random.default_rng(21)
    score = segment_rise_score(
        _preictal(rng), rate=_RATE, window=1024, step=256, aggregation="mean"
    )
    assert score > 0.0  # the planted whole-head rise


def test_segment_rise_score_focal_takes_the_most_rising_channel() -> None:
    rng = np.random.default_rng(22)
    focal = segment_rise_score(
        _preictal(rng), rate=_RATE, window=1024, step=256, aggregation="focal"
    )
    mean = segment_rise_score(
        _preictal(rng), rate=_RATE, window=1024, step=256, aggregation="mean"
    )
    # the maximum channel rise is at least as strong as the channel-averaged rise
    assert focal >= mean - 1.0e-9


def test_segment_rise_score_rejects_an_unknown_aggregation() -> None:
    rng = np.random.default_rng(23)
    with pytest.raises(ValueError, match="aggregation must be"):
        segment_rise_score(
            _preictal(rng), rate=_RATE, window=1024, step=256, aggregation="median"
        )


# --------------------------------------------------------------------------- #
# seizure_significance                                                        #
# --------------------------------------------------------------------------- #


def test_seizure_significance_separates_preictal_from_interictal() -> None:
    rng = np.random.default_rng(3)
    transitions = [_preictal(rng) for _ in range(6)]
    nulls = [_interictal(rng) for _ in range(12)]
    result = seizure_significance(
        transitions, nulls, rate=_RATE, window=1024, step=256, n_permutations=2000
    )
    assert isinstance(result, SeizureSignificance)
    assert result.significance.observed_led >= 5  # the planted rise is detected
    assert result.significance.p_value < 0.05  # beats chance when the rise is real
    record = result.to_audit_record()
    assert record["detector"] == "spectral_beta_delta_rise_mean"
    assert record["aggregation"] == "mean"
    assert set(record) == {
        "detector",
        "aggregation",
        "score_threshold",
        "achieved_false_alarm",
        "significance",
    }


def test_seizure_significance_focal_aggregation() -> None:
    rng = np.random.default_rng(6)
    transitions = [_preictal(rng) for _ in range(6)]
    nulls = [_interictal(rng) for _ in range(12)]
    result = seizure_significance(
        transitions,
        nulls,
        rate=_RATE,
        window=1024,
        step=256,
        aggregation="focal",
        n_permutations=2000,
    )
    assert result.aggregation == "focal"
    assert result.to_audit_record()["detector"] == "spectral_beta_delta_rise_focal"


def test_seizure_significance_rejects_empty_transitions() -> None:
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="transition_signals must not be empty"):
        seizure_significance([], [_interictal(rng)], rate=_RATE, window=1024, step=256)


def test_seizure_significance_rejects_empty_nulls() -> None:
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="null_signals must not be empty"):
        seizure_significance([_preictal(rng)], [], rate=_RATE, window=1024, step=256)


# --------------------------------------------------------------------------- #
# _positive_int                                                               #
# --------------------------------------------------------------------------- #


def test_positive_int_accepts_a_positive_integer() -> None:
    assert _positive_int(4, "window") == 4


def test_positive_int_rejects_a_boolean() -> None:
    with pytest.raises(ValueError, match="window must be a positive integer"):
        _positive_int(True, "window")


def test_positive_int_rejects_a_non_integer() -> None:
    with pytest.raises(ValueError, match="step must be a positive integer"):
        _positive_int(2.5, "step")


def test_positive_int_rejects_a_non_positive_integer() -> None:
    with pytest.raises(ValueError, match="window must be a positive integer"):
        _positive_int(0, "window")
