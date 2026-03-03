# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RegimeEvent:
    kind: str
    step: int
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in VALID_EVENT_KINDS:
            raise ValueError(
                f"invalid event kind {self.kind!r}, "
                f"expected one of {sorted(VALID_EVENT_KINDS)}"
            )


class EventBus:
    """Pub/sub bus for regime events with bounded history."""

    def __init__(self, maxlen: int = 200) -> None:
        self._subscribers: list = []
        self._history: deque[RegimeEvent] = deque(maxlen=maxlen)

    def subscribe(self, callback: object) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: object) -> None:
        self._subscribers = [s for s in self._subscribers if s != callback]

    def post(self, event: RegimeEvent) -> None:
        self._history.append(event)
        for cb in self._subscribers:
            cb(event)

    @property
    def history(self) -> list[RegimeEvent]:
        return list(self._history)

    @property
    def count(self) -> int:
        return len(self._history)

    def clear(self) -> None:
        self._history.clear()
