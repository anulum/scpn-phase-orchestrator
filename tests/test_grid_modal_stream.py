# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — grid modal streaming monitor tests

"""Tests for the causal real-time grid modal-growth streaming monitor.

The monitor's core correctness property is that its streaming score on a full window is
bit-for-bit the offline :func:`modal_growth_score` on that same window, so the
offline-certified threshold is valid online. That identity is pinned here, alongside the
sliding-window warm-up and step gating, the alarm on a growing stream and the silence on
a damped one, the persistence debounce, the latch-and-re-arm episode logic, construction
from a sealed certification, and every guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.grid_modal_stream import (
    WHOLE_NETWORK_BUS,
    GridModalStreamMonitor,
    StreamAlarm,
)
from bench.grid_oscillation_detector import modal_growth_score

_REPO = Path(__file__).resolve().parents[1]
_EVIDENCE = _REPO / "examples/real_data/psml_modal_growth/grid_modal_head_to_head.json"


def _oscillation(rng: np.random.Generator, *, sigma: float, n: int, buses: int = 4):
    """Return a (buses, n) multi-bus oscillation whose envelope grows at rate sigma."""
    rate = 100.0
    time = np.arange(n) / rate
    envelope = np.exp(sigma * time)
    wave = np.sin(2.0 * np.pi * 1.0 * time)
    return np.stack(
        [
            1.0
            + envelope * wave * rng.uniform(0.8, 1.2)
            + 1e-3 * rng.standard_normal(n)
            for _ in range(buses)
        ]
    )


def _feed(monitor: GridModalStreamMonitor, segment: np.ndarray) -> list[StreamAlarm]:
    """Feed every column of ``segment`` and collect the alarms raised."""
    alarms = []
    for index in range(segment.shape[1]):
        alarm = monitor.update(segment[:, index])
        if alarm is not None:
            alarms.append(alarm)
    return alarms


# --------------------------------------------------------------------------- #
# the core identity: streaming score == offline score on the same window       #
# --------------------------------------------------------------------------- #


def test_streaming_score_matches_the_offline_detector_on_the_window() -> None:
    rng = np.random.default_rng(0)
    segment = _oscillation(rng, sigma=0.5, n=100)
    monitor = GridModalStreamMonitor(
        rate=100.0, threshold=0.0, window_seconds=1.0, step_seconds=1.0
    )
    _feed(monitor, segment)  # 100 samples => window full, scored once
    offline = modal_growth_score(
        segment, rate=100.0, aggregation="focal", recency_top=3.0
    )
    assert monitor.latest_score == pytest.approx(offline, rel=1e-12)


def test_mean_aggregation_matches_the_offline_mean_score() -> None:
    rng = np.random.default_rng(1)
    segment = _oscillation(rng, sigma=0.4, n=100)
    monitor = GridModalStreamMonitor(
        rate=100.0,
        threshold=0.0,
        window_seconds=1.0,
        step_seconds=1.0,
        aggregation="mean",
    )
    _feed(monitor, segment)
    offline = modal_growth_score(
        segment, rate=100.0, aggregation="mean", recency_top=3.0
    )
    assert monitor.latest_score == pytest.approx(offline, rel=1e-12)


# --------------------------------------------------------------------------- #
# streaming behaviour                                                          #
# --------------------------------------------------------------------------- #


def test_warm_up_returns_no_alarm_before_the_window_is_full() -> None:
    rng = np.random.default_rng(2)
    segment = _oscillation(rng, sigma=0.5, n=60)
    monitor = GridModalStreamMonitor(
        rate=100.0, threshold=0.1, window_seconds=1.0, step_seconds=0.5
    )
    # only 60 < 100 window samples fed, so the monitor is still warming up
    assert _feed(monitor, segment) == []
    assert np.isnan(monitor.latest_score)


def test_growing_stream_raises_a_lead_alarm() -> None:
    rng = np.random.default_rng(3)
    segment = _oscillation(rng, sigma=0.6, n=160)
    monitor = GridModalStreamMonitor(
        rate=100.0, threshold=0.1, window_seconds=1.0, step_seconds=0.25
    )
    alarms = _feed(monitor, segment)
    assert len(alarms) == 1  # latched: one lead event per episode
    alarm = alarms[0]
    assert isinstance(alarm, StreamAlarm)
    assert alarm.score >= monitor.threshold
    assert 0 <= alarm.bus < 4  # a focal bus is attributed
    assert alarm.time_s == pytest.approx(alarm.sample_index / 100.0)


def test_damped_stream_stays_silent() -> None:
    rng = np.random.default_rng(4)
    segment = _oscillation(rng, sigma=-0.6, n=160)
    monitor = GridModalStreamMonitor(
        rate=100.0, threshold=0.1, window_seconds=1.0, step_seconds=0.25
    )
    assert _feed(monitor, segment) == []
    assert monitor.latest_score < monitor.threshold


def test_persistence_delays_the_alarm_until_sustained() -> None:
    rng = np.random.default_rng(5)
    segment = _oscillation(rng, sigma=0.6, n=200)
    lax = GridModalStreamMonitor(
        rate=100.0, threshold=0.1, window_seconds=1.0, step_seconds=0.25, persistence=1
    )
    strict = GridModalStreamMonitor(
        rate=100.0, threshold=0.1, window_seconds=1.0, step_seconds=0.25, persistence=3
    )
    first_lax = _feed(lax, segment)[0]
    first_strict = _feed(strict, segment)[0]
    # requiring three sustained windows fires no earlier than requiring one
    assert first_strict.sample_index >= first_lax.sample_index


def test_alarm_latches_then_re_arms_across_episodes() -> None:
    rng = np.random.default_rng(6)
    grow = _oscillation(rng, sigma=0.6, n=140)
    damp = _oscillation(rng, sigma=-0.9, n=140)
    monitor = GridModalStreamMonitor(
        rate=100.0, threshold=0.1, window_seconds=1.0, step_seconds=0.25
    )
    first = _feed(monitor, grow)
    assert len(first) == 1  # fires once, then latched while still growing
    quiet = _feed(monitor, damp)  # σ falls below threshold -> unlatch
    assert quiet == []
    second = _feed(monitor, grow)  # a fresh episode can fire again
    assert len(second) == 1


def test_reset_clears_window_and_alarm_state() -> None:
    rng = np.random.default_rng(7)
    segment = _oscillation(rng, sigma=0.6, n=140)
    monitor = GridModalStreamMonitor(
        rate=100.0, threshold=0.1, window_seconds=1.0, step_seconds=0.25
    )
    assert _feed(monitor, segment)  # fires and latches
    monitor.reset()
    assert np.isnan(monitor.latest_score)
    assert len(_feed(monitor, segment)) == 1  # fires again from a clean slate


# --------------------------------------------------------------------------- #
# construction from a sealed certification                                     #
# --------------------------------------------------------------------------- #


def test_from_evidence_carries_the_certified_operating_point() -> None:
    monitor = GridModalStreamMonitor.from_evidence(_EVIDENCE, rate=238.095)
    payload = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    modal = payload["modal"]
    assert monitor.threshold == pytest.approx(modal["score_threshold"])
    assert monitor.aggregation == modal["aggregation"]


def test_mean_aggregation_attributes_the_whole_network() -> None:
    rng = np.random.default_rng(8)
    segment = _oscillation(rng, sigma=0.6, n=140)
    monitor = GridModalStreamMonitor(
        rate=100.0,
        threshold=0.1,
        window_seconds=1.0,
        step_seconds=0.25,
        aggregation="mean",
    )
    alarm = _feed(monitor, segment)[0]
    assert alarm.bus == WHOLE_NETWORK_BUS


# --------------------------------------------------------------------------- #
# guards                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rate": 0.0}, "rate must be a positive"),
        ({"window_seconds": 0.0}, "window_seconds must be a positive"),
        ({"step_seconds": 0.0}, "step_seconds must be a positive"),
        ({"aggregation": "median"}, "aggregation must be"),
        ({"persistence": 0}, "persistence must be a positive"),
        ({"rate": 1.0, "window_seconds": 0.001}, "too short"),
    ],
)
def test_construction_guards(kwargs: dict[str, object], match: str) -> None:
    base: dict[str, object] = {"rate": 100.0, "threshold": 0.1}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        GridModalStreamMonitor(**base)  # type: ignore[arg-type]
