# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Supervisor event types

"""Validated supervisor event records and an in-process bounded event bus.

``RegimeEvent`` restricts event kinds and step/detail fields before publication,
and ``EventBus`` records a bounded chronological history while notifying
callable subscribers synchronously. The bus is process-local and passive: it
does not spawn threads, persist logs, retry subscriber failures, or emit network
traffic.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

__all__ = ["RegimeEvent", "EventBus"]

VALID_EVENT_KINDS = frozenset(
    {
        "boundary_breach",
        "r_threshold",
        "regime_transition",
        "manual",
        "petri_transition",
    }
)


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class RegimeEvent:
    """Immutable event emitted on regime transitions or boundary breaches."""

    kind: str
    step: int
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_EVENT_KINDS:
            raise ValueError(
                f"invalid event kind {self.kind!r}, "
                f"expected one of {sorted(VALID_EVENT_KINDS)}"
            )
        _validate_nonnegative_int(self.step, name="step")
        if not isinstance(self.detail, str):
            raise ValueError(f"detail must be a string, got {self.detail!r}")


class EventBus:
    """Pub/sub bus for regime events with bounded history."""

    def __init__(self, maxlen: int = 200) -> None:
        maxlen = _validate_positive_int(maxlen, name="maxlen")
        self._subscribers: list[Callable[[RegimeEvent], None]] = []
        self._history: deque[RegimeEvent] = deque(maxlen=maxlen)

    def subscribe(self, callback: object) -> None:
        """Register a callback to receive future events."""
        if not callable(callback):
            raise ValueError(f"callback must be callable, got {callback!r}")
        self._subscribers.append(callback)

    def unsubscribe(self, callback: object) -> None:
        """Remove a previously registered callback."""
        self._subscribers = [s for s in self._subscribers if s != callback]

    def post(self, event: RegimeEvent) -> None:
        """Record *event* in history and notify all subscribers."""
        if not isinstance(event, RegimeEvent):
            raise ValueError(f"event must be a RegimeEvent, got {event!r}")
        self._history.append(event)
        for cb in tuple(self._subscribers):
            cb(event)

    @property
    def history(self) -> list[RegimeEvent]:
        """Chronological list of all posted events."""
        return list(self._history)

    @property
    def count(self) -> int:
        """Number of events in history."""
        return len(self._history)

    def clear(self) -> None:
        """Discard all recorded events."""
        self._history.clear()
