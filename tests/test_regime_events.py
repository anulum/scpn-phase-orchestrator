# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Regime event tests

from __future__ import annotations

from collections.abc import Callable

import pytest

from scpn_phase_orchestrator.supervisor.events import (
    VALID_EVENT_KINDS,
    EventBus,
    RegimeEvent,
)

# The refusal tests pass deliberately mistyped arguments, so they construct
# and post through unchecked aliases.
_event_unchecked: Callable[..., RegimeEvent] = RegimeEvent
_bus_unchecked: Callable[..., EventBus] = EventBus


def test_event_creation() -> None:
    e = RegimeEvent(kind="manual", step=5, detail="test")
    assert e.kind == "manual"
    assert e.step == 5
    assert e.detail == "test"


def test_event_invalid_kind() -> None:
    with pytest.raises(ValueError, match="invalid event kind"):
        RegimeEvent(kind="bogus", step=0)


@pytest.mark.parametrize("step", [True, -1, 1.5, "1"])
def test_event_invalid_step(step: object) -> None:
    with pytest.raises(ValueError, match="step"):
        _event_unchecked(kind="manual", step=step)


@pytest.mark.parametrize("detail", [None, 1, object()])
def test_event_invalid_detail(detail: object) -> None:
    with pytest.raises(ValueError, match="detail"):
        _event_unchecked(kind="manual", step=0, detail=detail)


def test_event_default_detail() -> None:
    e = RegimeEvent(kind="r_threshold", step=0)
    assert e.detail == ""


def test_event_frozen() -> None:
    e = RegimeEvent(kind="manual", step=0)
    # Assign through setattr: a direct assignment to a frozen field is a
    # static error, and the runtime refusal is what this test checks.
    field_name = "kind"
    with pytest.raises(AttributeError):
        setattr(e, field_name, "other")


def test_valid_event_kinds_complete() -> None:
    for kind in VALID_EVENT_KINDS:
        e = RegimeEvent(kind=kind, step=0)
        assert e.kind == kind


def test_bus_post_and_history() -> None:
    bus = EventBus()
    e1 = RegimeEvent(kind="manual", step=0, detail="a")
    e2 = RegimeEvent(kind="manual", step=1, detail="b")
    bus.post(e1)
    bus.post(e2)
    assert bus.history == [e1, e2]
    assert bus.count == 2


def test_bus_subscribe_callback() -> None:
    bus = EventBus()
    received: list[RegimeEvent] = []
    bus.subscribe(received.append)
    e = RegimeEvent(kind="boundary_breach", step=3)
    bus.post(e)
    assert received == [e]


def test_bus_unsubscribe() -> None:
    bus = EventBus()
    received: list[RegimeEvent] = []
    bus.subscribe(received.append)
    bus.unsubscribe(received.append)
    bus.post(RegimeEvent(kind="manual", step=0))
    assert received == []


def test_bus_multiple_subscribers() -> None:
    bus = EventBus()
    a: list[RegimeEvent] = []
    b: list[RegimeEvent] = []
    bus.subscribe(a.append)
    bus.subscribe(b.append)
    e = RegimeEvent(kind="regime_transition", step=10)
    bus.post(e)
    assert a == [e]
    assert b == [e]


def test_bus_dispatch_uses_subscriber_snapshot() -> None:
    bus = EventBus()
    first: list[RegimeEvent] = []
    late: list[RegimeEvent] = []

    def subscribe_late(event: RegimeEvent) -> None:
        first.append(event)
        bus.subscribe(late.append)

    initial = RegimeEvent(kind="manual", step=0)
    next_event = RegimeEvent(kind="manual", step=1)
    bus.subscribe(subscribe_late)

    bus.post(initial)
    assert first == [initial]
    assert late == []

    bus.post(next_event)
    assert first == [initial, next_event]
    assert late == [next_event]


def test_bus_history_bounded() -> None:
    bus = EventBus(maxlen=3)
    for i in range(5):
        bus.post(RegimeEvent(kind="manual", step=i))
    assert bus.count == 3
    assert bus.history[0].step == 2
    assert bus.history[-1].step == 4


@pytest.mark.parametrize("maxlen", [True, 1.5, "3"])
def test_bus_rejects_invalid_maxlen_type(maxlen: object) -> None:
    with pytest.raises(ValueError, match="maxlen"):
        _bus_unchecked(maxlen=maxlen)


@pytest.mark.parametrize("callback", [None, object(), "not-callable"])
def test_bus_rejects_non_callable_subscriber(callback: object) -> None:
    with pytest.raises(ValueError, match="callback"):
        EventBus().subscribe(callback)


@pytest.mark.parametrize("event", [None, object(), "manual"])
def test_bus_rejects_non_event_post(event: object) -> None:
    post_unchecked: Callable[..., None] = EventBus().post
    with pytest.raises(ValueError, match="event"):
        post_unchecked(event)


def test_bus_clear() -> None:
    bus = EventBus()
    bus.post(RegimeEvent(kind="manual", step=0))
    bus.clear()
    assert bus.count == 0
    assert bus.history == []


def test_petri_transition_kind() -> None:
    e = RegimeEvent(kind="petri_transition", step=7, detail="warmup->nominal")
    assert e.kind == "petri_transition"


def test_r_threshold_kind() -> None:
    e = RegimeEvent(kind="r_threshold", step=2, detail="R < 0.3")
    assert e.kind == "r_threshold"


class TestEventBusPipelineWiring:
    """Pipeline wiring: EventBus fires when RegimeManager transitions."""

    def test_regime_transition_posts_event(self) -> None:
        """RegimeManager → EventBus: transition fires event to subscriber."""
        import numpy as np

        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        bus = EventBus()
        received: list[RegimeEvent] = []
        bus.subscribe(received.append)

        mgr = RegimeManager(cooldown_steps=0)
        # Drive into DEGRADED
        state = UPDEState(
            layers=[LayerState(R=0.4, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.4,
            regime_id="nominal",
        )
        regime = mgr.evaluate(state, BoundaryState())
        new_regime = mgr.transition(regime)
        bus.post(
            RegimeEvent(
                kind="regime_transition",
                step=0,
                detail=f"→{new_regime.value}",
            )
        )

        assert len(received) == 1
        assert received[0].kind == "regime_transition"
        assert new_regime.value in received[0].detail
