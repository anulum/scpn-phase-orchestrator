# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Petri net engine

from __future__ import annotations

import operator
from dataclasses import dataclass, field

from scpn_phase_orchestrator.exceptions import PolicyError

__all__ = ["Place", "Arc", "Transition", "Marking", "PetriNet"]

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


@dataclass(frozen=True)
class Place:
    name: str


@dataclass(frozen=True)
class Arc:
    place: str
    weight: int = 1


@dataclass(frozen=True)
class Guard:
    metric: str
    op: str
    threshold: float

    def evaluate(self, ctx: dict[str, float]) -> bool:
        val = ctx.get(self.metric)
        if val is None:
            return False
        fn = _OPS.get(self.op)
        if fn is None:
            return False
        return bool(fn(val, self.threshold))


@dataclass(frozen=True)
class Transition:
    name: str
    inputs: list[Arc]
    outputs: list[Arc]
    guard: Guard | None = None


@dataclass
class Marking:
    tokens: dict[str, int] = field(default_factory=dict)

    def __getitem__(self, place: str) -> int:
        return self.tokens.get(place, 0)

    def __setitem__(self, place: str, count: int) -> None:
        if count < 0:
            raise PolicyError(f"negative token count for {place!r}")
        if count == 0:
            self.tokens.pop(place, None)
        else:
            self.tokens[place] = count

    def active_places(self) -> list[str]:
        return [p for p, n in self.tokens.items() if n > 0]

    def copy(self) -> Marking:
        return Marking(tokens=dict(self.tokens))


def parse_guard(text: str) -> Guard:
    """Parse guard string like 'stability_proxy > 0.6'."""
    parts = text.split()
    if len(parts) != 3:
        raise PolicyError(f"guard must be 'metric op threshold', got {text!r}")
    return Guard(metric=parts[0], op=parts[1], threshold=float(parts[2]))


class PetriNet:
    """Classical Petri net with guard-gated transitions.

    step() fires at most one enabled transition per call (first-match priority).
    """

    def __init__(
        self,
        places: list[Place],
        transitions: list[Transition],
    ) -> None:
        self._place_names = frozenset(p.name for p in places)
        self._transitions = transitions
        self._validate()

    def _validate(self) -> None:
        for t in self._transitions:
            for arc in t.inputs + t.outputs:
                if arc.place not in self._place_names:
                    raise PolicyError(
                        f"transition {t.name!r} references unknown place {arc.place!r}"
                    )

    @property
    def place_names(self) -> frozenset[str]:
        return self._place_names

    @property
    def transitions(self) -> list[Transition]:
        return list(self._transitions)

    def enabled(self, marking: Marking, ctx: dict[str, float]) -> list[Transition]:
        result = []
        for t in self._transitions:
            if t.guard is not None and not t.guard.evaluate(ctx):
                continue
            if all(marking[arc.place] >= arc.weight for arc in t.inputs):
                result.append(t)
        return result

    def fire(self, marking: Marking, transition: Transition) -> Marking:
        new = marking.copy()
        for arc in transition.inputs:
            new[arc.place] = new[arc.place] - arc.weight
        for arc in transition.outputs:
            new[arc.place] = new[arc.place] + arc.weight
        return new

    def step(
        self, marking: Marking, ctx: dict[str, float]
    ) -> tuple[Marking, Transition | None]:
        """Fire the first enabled transition, return (new_marking, fired_transition)."""
        for t in self._transitions:
            if t.guard is not None and not t.guard.evaluate(ctx):
                continue
            if all(marking[arc.place] >= arc.weight for arc in t.inputs):
                return self.fire(marking, t), t
        return marking, None
