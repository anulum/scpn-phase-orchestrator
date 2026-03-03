# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.supervisor.events import (
    VALID_EVENT_KINDS,
    EventBus,
    RegimeEvent,
)


def test_event_creation():
    e = RegimeEvent(kind="manual", step=5, detail="test")
    assert e.kind == "manual"
    assert e.step == 5
    assert e.detail == "test"


def test_event_invalid_kind():
    with pytest.raises(ValueError, match="invalid event kind"):
        RegimeEvent(kind="bogus", step=0)


def test_event_default_detail():
    e = RegimeEvent(kind="r_threshold", step=0)
    assert e.detail == ""


def test_event_frozen():
    e = RegimeEvent(kind="manual", step=0)
    with pytest.raises(AttributeError):
        e.kind = "other"  # type: ignore[misc]


def test_valid_event_kinds_complete():
    for kind in VALID_EVENT_KINDS:
        e = RegimeEvent(kind=kind, step=0)
        assert e.kind == kind


def test_bus_post_and_history():
    bus = EventBus()
    e1 = RegimeEvent(kind="manual", step=0, detail="a")
    e2 = RegimeEvent(kind="manual", step=1, detail="b")
    bus.post(e1)
    bus.post(e2)
    assert bus.history == [e1, e2]
    assert bus.count == 2


def test_bus_subscribe_callback():
    bus = EventBus()
    received: list[RegimeEvent] = []
    bus.subscribe(received.append)
    e = RegimeEvent(kind="boundary_breach", step=3)
    bus.post(e)
    assert received == [e]


def test_bus_unsubscribe():
    bus = EventBus()
    received: list[RegimeEvent] = []
    bus.subscribe(received.append)
    bus.unsubscribe(received.append)
    bus.post(RegimeEvent(kind="manual", step=0))
    assert received == []


def test_bus_multiple_subscribers():
    bus = EventBus()
    a: list[RegimeEvent] = []
    b: list[RegimeEvent] = []
    bus.subscribe(a.append)
    bus.subscribe(b.append)
    e = RegimeEvent(kind="regime_transition", step=10)
    bus.post(e)
    assert a == [e]
    assert b == [e]


def test_bus_history_bounded():
    bus = EventBus(maxlen=3)
    for i in range(5):
        bus.post(RegimeEvent(kind="manual", step=i))
    assert bus.count == 3
    assert bus.history[0].step == 2
    assert bus.history[-1].step == 4


def test_bus_clear():
    bus = EventBus()
    bus.post(RegimeEvent(kind="manual", step=0))
    bus.clear()
    assert bus.count == 0
    assert bus.history == []


def test_petri_transition_kind():
    e = RegimeEvent(kind="petri_transition", step=7, detail="warmup->nominal")
    assert e.kind == "petri_transition"


def test_r_threshold_kind():
    e = RegimeEvent(kind="r_threshold", step=2, detail="R < 0.3")
    assert e.kind == "r_threshold"
