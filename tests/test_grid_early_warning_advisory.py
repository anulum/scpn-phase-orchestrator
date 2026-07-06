# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid early-warning advisory tests

"""Tests for the review-only grid early-warning advisory.

The neutral seal primitive and the live-monitor adapter are exercised: the record is
hash-sealed and recomputes, it is structurally non-actuating, it carries the honest
recall, and the lead is honest (early, late, coincident, or undefined). Every guard is
covered, and the adapter is driven end-to-end from a real monitor alarm on a growing
stream, so the streaming-to-decision path is pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.assurance.grid_early_warning_advisory import (
    GRID_ADVISORY_RAISED,
    GridEarlyWarningAdvisory,
    advise_from_stream_alarm,
    seal_grid_early_warning_advisory,
)
from scpn_phase_orchestrator.monitor.grid_modal_stream import (
    WHOLE_NETWORK_BUS,
    GridModalStreamMonitor,
)

_OK: dict[str, object] = {
    "detector": "grid_modal_growth_stream",
    "observable": "growth rate of the most unstable bus",
    "signal_source": "psml/scenario_42",
    "captured_at": "2026-07-06T00:00:00Z",
    "sampling_rate_hz": 238.0,
    "window_seconds": 2.0,
    "step_seconds": 0.5,
    "persistence": 2,
    "aggregation": "focal",
    "recency_top": 3.0,
    "r2_gate": 0.5,
    "warning_sample": 595,
    "warning_time_s": 2.5,
    "growth_rate": 0.8,
    "growth_rate_threshold": 0.3,
    "most_unstable_bus": 1,
    "certified_recall": 0.24,
    "certified_false_alarm": 0.10,
    "certified_operating_point": "grid_modal_stream_operating_point.json#3e2d74b7",
}


def _seal(**overrides: object) -> GridEarlyWarningAdvisory:
    return seal_grid_early_warning_advisory(**{**_OK, **overrides})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# seal_grid_early_warning_advisory                                            #
# --------------------------------------------------------------------------- #


def test_advisory_is_sealed_and_recomputes() -> None:
    advisory = _seal()
    record = advisory.to_audit_record()
    stored = record.pop("content_hash")
    from scpn_phase_orchestrator.assurance._hashing import canonical_record_hash

    assert stored == canonical_record_hash(record)
    assert advisory.verdict == GRID_ADVISORY_RAISED
    assert advisory.certified_recall == 0.24


def test_advisory_is_structurally_non_actuating() -> None:
    advisory = _seal()
    assert advisory.non_actuating is True
    assert advisory.actuating is False


def test_advisory_lead_is_early_when_onset_follows_the_alarm() -> None:
    advisory = _seal(transition_onset_sample=760)
    assert advisory.lead_samples == 760 - 595
    assert advisory.lead_seconds == pytest.approx((760 - 595) / 238.0)
    assert advisory.lead_is_early is True


def test_advisory_lead_is_not_early_when_alarm_is_late() -> None:
    advisory = _seal(warning_sample=800, transition_onset_sample=760)
    assert advisory.lead_samples == 760 - 800
    assert advisory.lead_is_early is False  # a non-positive lead is reported honestly


def test_advisory_lead_is_not_early_when_coincident() -> None:
    advisory = _seal(warning_sample=760, transition_onset_sample=760)
    assert advisory.lead_samples == 0
    assert advisory.lead_is_early is False


def test_advisory_lead_is_undefined_without_an_onset() -> None:
    advisory = _seal()
    assert advisory.lead_samples is None
    assert advisory.lead_seconds is None
    assert advisory.lead_is_early is False


def test_advisory_allows_the_whole_network_bus() -> None:
    advisory = _seal(aggregation="mean", most_unstable_bus=WHOLE_NETWORK_BUS)
    assert advisory.most_unstable_bus == WHOLE_NETWORK_BUS


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"detector": " "}, "detector must be a non-empty"),
        ({"sampling_rate_hz": 0.0}, "sampling_rate_hz must be positive"),
        ({"window_seconds": 0.0}, "window_seconds must be positive"),
        ({"persistence": 0}, "persistence must be a positive integer"),
        ({"aggregation": "median"}, "aggregation must be"),
        ({"recency_top": 0.5}, "recency_top must be at least one"),
        ({"r2_gate": 1.5}, r"r2_gate must be in \[0, 1\]"),
        ({"certified_recall": -0.1}, r"certified_recall must be in \[0, 1\]"),
        ({"warning_sample": -1}, "warning_sample must be non-negative"),
        ({"most_unstable_bus": -2}, "most_unstable_bus must be >="),
        ({"growth_rate": float("nan")}, "growth_rate must be finite"),
        ({"growth_rate": "fast"}, "growth_rate must be a finite real"),
        ({"persistence": 2.5}, "persistence must be a positive integer"),
        ({"warning_sample": 1.5}, "warning_sample must be an integer"),
        ({"transition_onset_sample": -3}, "must be non-negative"),
    ],
)
def test_advisory_guards(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _seal(**overrides)


# --------------------------------------------------------------------------- #
# advise_from_stream_alarm — the live streaming-to-decision path               #
# --------------------------------------------------------------------------- #


def _growing_stream(sigma: float, *, rate: float, n: int, buses: int = 4) -> np.ndarray:
    time = np.arange(n) / rate
    offsets = np.linspace(-0.3, 0.3, buses)
    return 1.0 + offsets[:, None] * np.exp(sigma * time)[None, :]


def test_advise_from_stream_alarm_reads_the_monitor_and_alarm() -> None:
    rate = 238.0
    monitor = GridModalStreamMonitor(
        rate=rate,
        threshold=0.3,
        window_seconds=2.0,
        step_seconds=0.5,
        persistence=2,
        r2_gate=0.5,
    )
    stream = _growing_stream(0.8, rate=rate, n=800)
    alarm = None
    for index in range(stream.shape[1]):
        alarm = monitor.update(stream[:, index])
        if alarm is not None:
            break
    assert alarm is not None

    advisory = advise_from_stream_alarm(
        alarm,
        monitor,
        signal_source="psml/scenario_42",
        captured_at="2026-07-06T00:00:00Z",
        certified_recall=0.24,
        certified_false_alarm=0.10,
        certified_operating_point="grid_modal_stream_operating_point.json#3e2d74b7",
        transition_onset_sample=alarm.sample_index + 100,
    )
    # the advisory records exactly what the monitor and alarm carried
    assert advisory.growth_rate == alarm.score
    assert advisory.growth_rate_threshold == alarm.threshold
    assert advisory.most_unstable_bus == alarm.bus
    assert advisory.warning_sample == alarm.sample_index
    assert advisory.window_seconds == monitor.window_seconds
    assert advisory.step_seconds == monitor.step_seconds
    assert advisory.persistence == monitor.persistence
    assert advisory.recency_top == monitor.recency_top
    assert advisory.r2_gate == monitor.r2_gate
    assert advisory.non_actuating is True
    assert advisory.lead_is_early is True
