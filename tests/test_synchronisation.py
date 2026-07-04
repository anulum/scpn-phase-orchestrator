# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — rising-synchronisation monitor tests

"""Tests for the rising-synchronisation early-warning monitor.

The monitor is the first-moment (order-parameter) member of the early-warning
detector suite — the indicator that carries the leading precursor on a real
scalp-EEG seizure. The tests exercise the alarm on a genuine coherence rise,
silence on an incoherent null, the degenerate zero-coherence baseline, the phase
validation surface, and the summary export.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.synchronisation import (
    SynchronisationWarning,
    synchronisation_warning,
)


def _rising_coherence_phases(
    n_nodes: int = 8, length: int = 2000, seed: int = 0
) -> np.ndarray:
    """Return phases that drift incoherently then lock to a common phase.

    A ``lock`` ramp over the second half blends each node's independent drift
    into a shared phase, so the Kuramoto order parameter climbs toward one — the
    synchronisation precursor.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    lock = np.clip((t - length // 2) / (length // 2), 0.0, 1.0)
    common = 0.01 * t
    phases = np.empty((n_nodes, length), dtype=np.float64)
    for node in range(n_nodes):
        individual = (
            rng.uniform(0.0, 2.0 * np.pi) + 0.02 * t + rng.standard_normal(length) * 0.3
        )
        phases[node] = (1.0 - lock) * individual + lock * common
    return phases


def test_alarms_on_rising_coherence() -> None:
    warning = synchronisation_warning(
        _rising_coherence_phases(), window=128, step=16, z_threshold=3.0
    )
    assert warning.warning_triggered is True
    assert warning.warning_sample is not None
    assert 1000 <= warning.warning_sample < 2000
    assert warning.synchrony_index.max() > 0.9
    assert warning.baseline_median < 0.6


def test_silent_on_incoherent_phases() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(2000)
    incoherent = rng.uniform(0.0, 2.0 * np.pi, (8, 2000)) + 0.02 * t
    warning = synchronisation_warning(incoherent, window=128, step=16, z_threshold=3.0)
    assert warning.warning_triggered is False
    assert warning.warning_window is None
    assert warning.warning_sample is None


def test_echoes_analysis_parameters() -> None:
    warning = synchronisation_warning(
        _rising_coherence_phases(),
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


def test_zero_coherence_baseline_blocks_the_relative_gate() -> None:
    # Two anti-phase nodes give a vanishing order parameter across the baseline
    # (R within a rounding epsilon of zero), so the relative-rise gate cannot be
    # evaluated and the alarm is suppressed.
    length = 1200
    phases = np.zeros((2, length), dtype=np.float64)
    phases[1, :600] = np.pi  # anti-phase pair -> R ~ 0 in the baseline
    phases[:, 600:] = 0.5  # both lock -> R = 1 on the tail
    warning = synchronisation_warning(phases, window=128, step=16)
    assert warning.baseline_median < 1.0e-10
    assert np.all(warning.relative_rise == 0.0)
    assert warning.warning_triggered is False


def test_window_larger_than_series_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds the series length"):
        synchronisation_warning(np.zeros((3, 40)), window=64)


def test_single_node_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two nodes"):
        synchronisation_warning(np.zeros((1, 200)), window=64)


@pytest.mark.parametrize(
    "phases",
    [
        np.zeros((2, 4), dtype=bool),
        np.zeros((2, 4), dtype=complex),
        np.zeros(200),
        np.zeros((2, 2, 4)),
        np.array([[0.0, np.nan, 1.0, 2.0], [0.0, 0.0, 0.0, 0.0]]),
    ],
)
def test_malformed_phases_are_rejected(phases: np.ndarray) -> None:
    with pytest.raises(ValueError):
        synchronisation_warning(phases, window=3, step=1)


def test_object_phases_are_rejected() -> None:
    with pytest.raises(ValueError, match="real float array"):
        synchronisation_warning(
            np.array([[object(), object()], [object(), object()]]), window=2
        )


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
        ({"persistence": True}, "persistence"),
    ],
)
def test_invalid_controls_are_rejected(kwargs: dict[str, object], match: str) -> None:
    base = np.zeros((3, 300), dtype=np.float64)
    with pytest.raises(ValueError, match=match):
        synchronisation_warning(base, **kwargs)  # type: ignore[arg-type]


def test_summary_reports_the_verdict() -> None:
    warning = synchronisation_warning(_rising_coherence_phases(), window=128, step=16)
    summary = warning.summary()
    assert summary["warning_triggered"] is True
    assert summary["max_robust_z"] == pytest.approx(float(warning.robust_z.max()))
    assert summary["max_synchrony_index"] == pytest.approx(
        float(warning.synchrony_index.max())
    )


def test_summary_handles_an_empty_result() -> None:
    empty = np.empty(0, dtype=np.float64)
    warning = SynchronisationWarning(
        window_starts=np.empty(0, dtype=np.int64),
        synchrony_index=empty,
        robust_z=empty,
        relative_rise=empty,
        baseline_median=0.0,
        baseline_scale=0.0,
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
    assert summary["max_synchrony_index"] == 0.0
    assert summary["max_robust_z"] == 0.0
    assert summary["max_relative_rise"] == 0.0
