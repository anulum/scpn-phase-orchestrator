# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AR(1)-Kendall-tau competitor detector tests

"""Tests for the AR(1)-Kendall-τ competitor detector.

The detector reduces a segment to the Kendall-τ trend of its windowed lag-one
autocorrelation, calibrates a matched-false-alarm τ threshold on the null trials, and
scores the transition alarm count by the shared permutation core. Every path is
exercised here on synthetic series — a rising-autocorrelation approach, a stationary
null, and degenerate constant windows — independent of any corpus.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench.competitor_ar1_kendall import (
    Ar1KendallSignificance,
    ar1_kendall_significance,
    ar1_trend_tau,
    calibrate_tau_threshold,
)


def _rising_ar1(n_samples: int = 600, seed: int = 0) -> np.ndarray:
    """Return a series whose lag-one autocorrelation rises towards its end.

    A white-noise baseline (near-zero autocorrelation) gives way to a cumulatively
    smoothed tail (high autocorrelation) — the critical-slowing-down signature the
    detector should read as a positive Kendall-τ trend.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)
    smoothed = np.cumsum(rng.standard_normal(n_samples)) * 0.15
    ramp = np.linspace(0.0, 1.0, n_samples)
    return white * (1.0 - ramp) + smoothed * ramp


def _stationary(n_samples: int = 600, seed: int = 1) -> np.ndarray:
    """Return a stationary white-noise series — no rising autocorrelation."""
    return np.random.default_rng(seed).standard_normal(n_samples)


# --------------------------------------------------------------------------- #
# ar1_trend_tau                                                               #
# --------------------------------------------------------------------------- #


def test_tau_is_positive_on_a_rising_autocorrelation() -> None:
    tau = ar1_trend_tau(_rising_ar1(seed=2), window=50, step=5)
    assert tau > 0.3  # a clear rising-AR(1) trend


def test_tau_is_near_zero_on_a_stationary_series() -> None:
    tau = ar1_trend_tau(_stationary(seed=3), window=50, step=5)
    assert abs(tau) < 0.3


def test_tau_is_zero_on_a_constant_series() -> None:
    # Every window has zero variance, so each autocorrelation is 0 and the trend is
    # undefined (Kendall-τ of a constant); the detector reads no warning.
    assert ar1_trend_tau(np.full(200, 3.0), window=50, step=5) == 0.0


def test_tau_is_zero_when_fewer_than_two_windows_fit() -> None:
    # A single window admits no trend across windows.
    assert ar1_trend_tau(np.arange(50.0), window=50, step=5) == 0.0


def test_tau_rejects_a_two_dimensional_series() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        ar1_trend_tau(np.zeros((2, 60)), window=10, step=2)


def test_tau_rejects_a_window_below_three() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        ar1_trend_tau(np.arange(60.0), window=2, step=1)


def test_tau_rejects_a_window_longer_than_the_series() -> None:
    with pytest.raises(ValueError, match="exceeds the series length"):
        ar1_trend_tau(np.arange(20.0), window=50, step=5)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_tau_rejects_a_non_positive_window(bad: object) -> None:
    with pytest.raises(ValueError, match="window"):
        ar1_trend_tau(np.arange(100.0), window=bad, step=5)


def test_tau_rejects_a_non_positive_step() -> None:
    with pytest.raises(ValueError, match="step"):
        ar1_trend_tau(np.arange(100.0), window=20, step=0)


# --------------------------------------------------------------------------- #
# calibrate_tau_threshold                                                     #
# --------------------------------------------------------------------------- #


def test_calibrate_tau_threshold_holds_the_false_alarm_at_target() -> None:
    nulls = [0.9, 0.8, 0.1, 0.0, -0.2, -0.5, 0.05, 0.02, -0.1, -0.3]  # 10 nulls
    threshold = calibrate_tau_threshold(nulls, target_fa=0.1)
    # One null (0.9) may alarm within the 10 % budget, so the threshold sits just above
    # the second-largest (0.8): 0.9 alarms, 0.8 does not.
    alarms = sum(tau >= threshold for tau in nulls)
    assert alarms <= 1


def test_calibrate_tau_threshold_opens_the_gate_when_all_may_alarm() -> None:
    assert calibrate_tau_threshold([0.5, 0.5], target_fa=1.0) == float(-np.inf)


def test_calibrate_tau_threshold_rejects_an_empty_null() -> None:
    with pytest.raises(ValueError, match="null_taus must not be empty"):
        calibrate_tau_threshold([])


# --------------------------------------------------------------------------- #
# ar1_kendall_significance                                                    #
# --------------------------------------------------------------------------- #


def test_significance_flags_rising_transitions_against_stationary_nulls() -> None:
    transitions = [_rising_ar1(seed=s) for s in range(5)]
    nulls = [_stationary(seed=s) for s in range(100, 112)]
    result = ar1_kendall_significance(
        transitions, nulls, window=50, step=5, target_fa=0.1, n_permutations=5000
    )
    assert isinstance(result, Ar1KendallSignificance)
    assert result.achieved_false_alarm <= 0.1 + 1.0e-9
    assert result.significance.observed_led >= 3
    assert result.significance.p_value < 0.05


def test_significance_is_at_chance_when_transitions_match_nulls() -> None:
    stationary = [_stationary(seed=s) for s in range(20)]
    result = ar1_kendall_significance(
        stationary[:6], stationary[6:], window=50, step=5, n_permutations=5000
    )
    assert result.significance.p_value > 0.05


def test_significance_audit_record_round_trips() -> None:
    transitions = [_rising_ar1(seed=9)]
    nulls = [_stationary(seed=s) for s in range(200, 205)]
    record = json.loads(
        json.dumps(
            ar1_kendall_significance(
                transitions, nulls, window=50, step=5, n_permutations=200
            ).to_audit_record()
        )
    )
    assert record["detector"] == "ar1_kendall_tau"
    assert "tau_threshold" in record
    assert record["significance"]["n_transitions"] == 1


def test_significance_rejects_empty_sets() -> None:
    nulls = [_stationary(seed=1)]
    with pytest.raises(ValueError, match="transition_series must not be empty"):
        ar1_kendall_significance([], nulls, window=50, step=5)
    transitions = [_rising_ar1(seed=1)]
    with pytest.raises(ValueError, match="null_series must not be empty"):
        ar1_kendall_significance(transitions, [], window=50, step=5)
