# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — replay lead time needs an integer onset and fresh monitor

"""Lead time is measured on whole samples against a monitor counting from one.

A float onset gave fractional lead times, NaN silently discarded a real lead,
and a monitor that had already consumed a stream numbered its alarms from where
it stopped, so a genuine 0.5 s lead was reported as "not led".
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.grid_modal_stream import GridModalStreamMonitor
from scpn_phase_orchestrator.runtime.grid_modal_replay import replay_lead_time


def _growing_stream(n: int = 160) -> np.ndarray:
    time = np.arange(n) / 100.0
    offsets = np.linspace(-0.3, 0.3, 4)
    return 1.0 + offsets[:, None] * np.exp(0.6 * time)[None, :]


def _monitor() -> GridModalStreamMonitor:
    return GridModalStreamMonitor(
        rate=100.0, threshold=0.1, window_seconds=1.0, step_seconds=0.25
    )


def test_integer_onset_reports_the_lead() -> None:
    result = replay_lead_time(_growing_stream(), _monitor(), onset_sample=150)
    assert result.led is True
    assert result.first_alarm is not None
    assert result.lead_time_s == pytest.approx(
        (150 - result.first_alarm.sample_index) / 100.0
    )


def test_numpy_integer_onset_is_accepted() -> None:
    result = replay_lead_time(_growing_stream(), _monitor(), onset_sample=np.int64(150))
    assert result.led is True


@pytest.mark.parametrize("onset", [150.9, 150.0, math.nan, True, 0, -5, "150"])
def test_non_integer_or_non_positive_onset_is_refused(onset: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        replay_lead_time(_growing_stream(), _monitor(), onset_sample=onset)  # type: ignore[arg-type]


def test_used_monitor_is_refused_and_reset_restores_the_lead() -> None:
    monitor = _monitor()
    first = replay_lead_time(_growing_stream(), monitor, onset_sample=150)
    assert monitor.samples_seen == 160
    with pytest.raises(ValueError, match="already consumed 160 samples"):
        replay_lead_time(_growing_stream(), monitor, onset_sample=150)
    monitor.reset()
    assert monitor.samples_seen == 0
    again = replay_lead_time(_growing_stream(), monitor, onset_sample=150)
    assert again.led is True
    assert again.lead_time_s == pytest.approx(first.lead_time_s)
