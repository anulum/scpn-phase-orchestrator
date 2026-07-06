# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid modal-growth stream replay tests

"""Tests for replaying a recorded per-bus voltage stream through the live monitor.

The replay driver feeds a recorded stream sample by sample into the causal monitor and,
for a known onset, reports whether the transition was led and by how long. A growing
stream leads its onset with a positive lead time; a damped stream stays silent; and
every guard is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.grid_modal_stream import GridModalStreamMonitor
from scpn_phase_orchestrator.runtime.grid_modal_replay import (
    ReplayResult,
    replay,
    replay_lead_time,
)


def _stream(*, sigma: float, n: int, buses: int = 4) -> np.ndarray:
    """Return a deterministic (buses, n) stream whose deviation grows at sigma.

    Each bus carries a distinct constant offset scaled by ``exp(sigma·t)`` — a monotone
    envelope with no oscillation, so the cross-bus deviation is exactly ``exp(sigma·t)``
    and every bus's log-envelope slope is the planted ``sigma``. This exercises the
    replay plumbing and the growing/damped separation cleanly, without the log-flooring
    edge case a synthetic sinusoid starting at a zero crossing would introduce (real PMU
    data is noisy and never lands exactly on a zero, so the detector never sees it).
    """
    time = np.arange(n) / 100.0
    envelope = np.exp(sigma * time)
    offsets = np.linspace(-0.2, 0.2, buses)
    return np.stack([1.0 + envelope * offset for offset in offsets])


def _monitor() -> GridModalStreamMonitor:
    return GridModalStreamMonitor(
        rate=100.0, threshold=0.1, window_seconds=1.0, step_seconds=0.25
    )


def test_replay_collects_alarms_on_a_growing_stream() -> None:
    alarms = replay(_stream(sigma=0.6, n=160), _monitor())
    assert len(alarms) == 1  # one lead event per episode (the monitor latches)
    assert alarms[0].score >= 0.1


def test_replay_is_silent_on_a_damped_stream() -> None:
    assert replay(_stream(sigma=-0.6, n=160), _monitor()) == []


def test_replay_rejects_a_non_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        replay(np.zeros(8), _monitor())


def test_replay_lead_time_leads_a_growing_onset() -> None:
    segment = _stream(sigma=0.6, n=160)
    result = replay_lead_time(segment, _monitor(), onset_sample=160)
    assert isinstance(result, ReplayResult)
    assert result.led is True
    assert result.first_alarm is not None
    assert result.lead_time_s > 0.0  # the alarm fired before the onset
    # lead time is the onset-to-alarm gap over the rate
    assert result.lead_time_s == pytest.approx(
        (160 - result.first_alarm.sample_index) / 100.0
    )


def test_replay_lead_time_reports_no_lead_on_a_damped_onset() -> None:
    result = replay_lead_time(_stream(sigma=-0.6, n=160), _monitor(), onset_sample=160)
    assert result.led is False
    assert result.first_alarm is None
    assert np.isnan(result.lead_time_s)


def test_replay_lead_time_ignores_alarms_after_the_onset() -> None:
    segment = _stream(sigma=0.6, n=160)
    # an onset one sample in means every alarm fires after it -> not a lead
    result = replay_lead_time(segment, _monitor(), onset_sample=1)
    assert result.led is False
    assert result.first_alarm is None
    assert result.alarms  # alarms were still raised, just none before the onset


def test_replay_lead_time_rejects_a_non_positive_onset() -> None:
    with pytest.raises(ValueError, match="onset_sample must be a positive"):
        replay_lead_time(_stream(sigma=0.6, n=160), _monitor(), onset_sample=0)
