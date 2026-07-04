# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — critical-slowing-down monitor tests

"""Tests for the critical-slowing-down early-warning baseline monitor.

The monitor is the literature baseline (rising variance + lag-one
autocorrelation) for the fair head-to-head against the ordinal-transition-entropy
detector. The tests exercise the alarm on a genuine slowing-down ramp, silence on
a stationary null, the shared window sweep and validation surface, the degenerate
constant-baseline path, and the summary export — so the baseline that decides the
comparison is itself trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.critical_slowing_down import (
    CriticalSlowingDownWarning,
    critical_slowing_down_warning,
)


def _slowing_down_signal(
    n_nodes: int = 4, length: int = 2000, seed: int = 0
) -> np.ndarray:
    """Return an AR(1) signal whose recovery slows over the second half.

    The lag-one coefficient ramps from 0.1 to 0.97 across the second half, so
    variance and autocorrelation both rise ahead of the notional transition at
    the midpoint — the textbook critical-slowing-down precursor.
    """
    rng = np.random.default_rng(seed)
    half = length // 2
    coefficients = np.concatenate(
        [np.full(half, 0.1), np.linspace(0.1, 0.97, length - half)]
    )
    signal = np.zeros((n_nodes, length), dtype=np.float64)
    for node in range(n_nodes):
        for t in range(1, length):
            signal[node, t] = (
                coefficients[t] * signal[node, t - 1] + rng.standard_normal() * 0.2
            )
    return signal


def test_alarms_on_a_critical_slowing_down_ramp() -> None:
    warning = critical_slowing_down_warning(
        _slowing_down_signal(), window=128, step=16, z_threshold=3.0
    )
    assert warning.warning_triggered is True
    assert warning.warning_sample is not None
    # The alarm lands after the ramp begins (midpoint) and before the series ends.
    assert 1000 <= warning.warning_sample < 2000
    assert warning.combined_z.max() > 3.0


def test_silent_on_a_stationary_null() -> None:
    # The "either indicator" combination is deliberately sensitive, so a
    # stationary null is asserted silent under a stringent gate (a genuinely
    # stationary signal must not raise a strong, sustained slowing-down warning);
    # false-alarm control at looser gates is the calibration harness's job.
    rng = np.random.default_rng(1)
    stationary = rng.standard_normal((4, 2000)) * 0.2
    warning = critical_slowing_down_warning(
        stationary, window=128, step=16, z_threshold=6.0, persistence=3
    )
    assert warning.warning_triggered is False
    assert warning.warning_window is None
    assert warning.warning_sample is None


def test_one_dimensional_signal_is_treated_as_a_single_node() -> None:
    warning = critical_slowing_down_warning(
        _slowing_down_signal(n_nodes=1)[0], window=128, step=16
    )
    assert warning.variance_index.ndim == 1
    assert warning.warning_triggered is True


def test_echoes_analysis_parameters() -> None:
    warning = critical_slowing_down_warning(
        _slowing_down_signal(),
        window=100,
        step=20,
        z_threshold=2.5,
        rise_threshold=0.2,
        persistence=3,
    )
    assert warning.window == 100
    assert warning.step == 20
    assert warning.z_threshold == 2.5
    assert warning.rise_threshold == 0.2
    assert warning.persistence == 3


def test_constant_baseline_blocks_the_relative_gate() -> None:
    # A perfectly constant leading segment gives a zero baseline median, so the
    # relative-rise gate cannot be evaluated and the alarm is suppressed even
    # though the robust z-score explodes on the noisy tail.
    rng = np.random.default_rng(2)
    signal = np.zeros((3, 1200), dtype=np.float64)
    signal[:, 600:] = rng.standard_normal((3, 600)) * 0.5
    warning = critical_slowing_down_warning(signal, window=128, step=16)
    assert warning.baseline_variance == 0.0
    assert warning.baseline_autocorrelation == 0.0
    assert np.all(warning.relative_rise == 0.0)
    assert warning.warning_triggered is False


def test_window_below_three_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 3 for autocorrelation"):
        critical_slowing_down_warning(np.zeros((2, 100)), window=2)


def test_window_larger_than_series_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds the series length"):
        critical_slowing_down_warning(np.zeros((2, 50)), window=64)


@pytest.mark.parametrize(
    "signals",
    [
        np.zeros((2, 4), dtype=bool),
        np.zeros((2, 4), dtype=complex),
        np.zeros((2, 2, 4)),
        np.array([[0.0, np.nan, 1.0, 2.0]]),
    ],
)
def test_malformed_signals_are_rejected(signals: np.ndarray) -> None:
    with pytest.raises(ValueError):
        critical_slowing_down_warning(signals, window=3, step=1)


def test_object_signals_are_rejected() -> None:
    with pytest.raises(ValueError, match="real float array"):
        critical_slowing_down_warning(np.array([[object(), object()]]), window=3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"window": 0}, "window"),
        ({"window": 1.5}, "window"),
        ({"step": 0}, "step"),
        ({"min_baseline_windows": 0}, "min_baseline_windows"),
        ({"baseline_fraction": 0.0}, "baseline_fraction"),
        ({"baseline_fraction": 1.0}, "baseline_fraction"),
        ({"baseline_fraction": "x"}, "baseline_fraction"),
        ({"z_threshold": -1.0}, "z_threshold"),
        ({"z_threshold": "x"}, "z_threshold"),
        ({"rise_threshold": -0.1}, "rise_threshold"),
        ({"persistence": 0}, "persistence"),
        ({"window": True}, "window"),
    ],
)
def test_invalid_controls_are_rejected(kwargs: dict[str, object], match: str) -> None:
    base = np.zeros((2, 300), dtype=np.float64)
    with pytest.raises(ValueError, match=match):
        critical_slowing_down_warning(base, **kwargs)  # type: ignore[arg-type]


def test_summary_reports_the_verdict() -> None:
    warning = critical_slowing_down_warning(_slowing_down_signal(), window=128, step=16)
    summary = warning.summary()
    assert summary["warning_triggered"] is True
    assert summary["max_combined_z"] == pytest.approx(float(warning.combined_z.max()))
    assert summary["n_windows"] == int(warning.combined_z.shape[0])


def test_summary_handles_an_empty_result() -> None:
    empty = np.empty(0, dtype=np.float64)
    warning = CriticalSlowingDownWarning(
        window_starts=np.empty(0, dtype=np.int64),
        variance_index=empty,
        autocorrelation_index=empty,
        combined_z=empty,
        robust_z_variance=empty,
        robust_z_autocorrelation=empty,
        relative_rise=empty,
        baseline_variance=0.0,
        baseline_autocorrelation=0.0,
        baseline_scale_variance=0.0,
        baseline_scale_autocorrelation=0.0,
        n_baseline_windows=0,
        warning_triggered=False,
        warning_window=None,
        warning_sample=None,
        window=128,
        step=16,
        z_threshold=3.0,
        rise_threshold=0.1,
        persistence=2,
    )
    summary = warning.summary()
    assert summary["max_combined_z"] == 0.0
    assert summary["max_relative_rise"] == 0.0
