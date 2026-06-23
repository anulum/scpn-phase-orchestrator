# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Petri net engine

"""Guarded Petri-net primitives for deterministic regime transition modeling.

The module defines validated places, weighted arcs, guards, transitions,
markings, and a first-match-priority Petri net. Marking updates are local and
non-negative, guard metrics must be finite, and net construction rejects arcs to
unknown places. The engine performs no event emission or policy action mapping;
adapter modules own those boundaries.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real

from scpn_phase_orchestrator.exceptions import PolicyError

__all__ = ["Place", "Arc", "Transition", "Marking", "PetriNet"]

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


def _validate_name(value: object, *, kind: str) -> str:
    """Return the validated place/transition name, else raise."""
    if not isinstance(value, str) or not value:
        raise PolicyError(f"{kind} names must not be empty, got {value!r}")
    return value


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise PolicyError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise PolicyError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _validate_finite_real(value: object, *, name: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PolicyError(f"{name} must be finite, got {value!r}")
    out = float(value)
    if not isfinite(out):
        raise PolicyError(f"{name} must be finite, got {value!r}")
    return out


@dataclass(frozen=True)
class Place:
    """Named place (state) in the Petri net."""

    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name, kind="place"))


@dataclass(frozen=True)
class Arc:
    """Weighted arc connecting a place to a transition."""

    place: str
    weight: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "place", _validate_name(self.place, kind="place"))
        object.__setattr__(
            self,
            "weight",
            _validate_positive_int(self.weight, name="weight"),
        )


@dataclass(frozen=True)
class Guard:
    """Boolean guard condition on a named metric (e.g. 'stability_proxy > 0.6')."""

    metric: str
    op: str
    threshold: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _validate_name(self.metric, kind="metric"))
        if not isinstance(self.op, str) or self.op not in _OPS:
            raise PolicyError(
                f"operator must be one of {sorted(_OPS)}, got {self.op!r}"
            )
        object.__setattr__(
            self,
            "threshold",
            _validate_finite_real(self.threshold, name="threshold"),
        )

    def evaluate(self, ctx: Mapping[str, float]) -> bool:
        """Return True if the guard condition is satisfied by *ctx*."""
        val = ctx.get(self.metric)
        if val is None:
            return False
        val = _validate_finite_real(val, name=f"context metric {self.metric!r}")
        fn = _OPS.get(self.op)
        if fn is None:
            return False
        return bool(fn(val, self.threshold))


@dataclass(frozen=True)
class Transition:
    """Petri net transition with input/output arcs and optional guard."""

    name: str
    inputs: list[Arc]
    outputs: list[Arc]
    guard: Guard | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _validate_name(self.name, kind="transition"),
        )


@dataclass
class Marking:
    """Token distribution across places in a Petri net."""

    tokens: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        initial = dict(self.tokens)
        self.tokens.clear()
        for place, count in initial.items():
            self[place] = count

    def __getitem__(self, place: str) -> int:
        return self.tokens.get(place, 0)

    def __setitem__(self, place: str, count: int) -> None:
        place = _validate_name(place, kind="place")
        count = _validate_nonnegative_int(count, name="token count")
        if count == 0:
            self.tokens.pop(place, None)
        else:
            self.tokens[place] = count

    def active_places(self) -> list[str]:
        """Return names of places that hold at least one token.

        Returns
        -------
        list[str]
            Return names of places that hold at least one token.
        """
        return [p for p, n in self.tokens.items() if n > 0]

    def copy(self) -> Marking:
        """Return a shallow copy of this marking.

        Returns
        -------
        Marking
            Return a shallow copy of this marking.
        """
        return Marking(tokens=dict(self.tokens))


def parse_guard(text: str) -> Guard:
    """Parse guard string like 'stability_proxy > 0.6'."""
    parts = text.split()
    if len(parts) != 3:
        raise PolicyError(f"guard must be 'metric op threshold', got {text!r}")
    try:
        threshold = float(parts[2])
    except ValueError as exc:
        raise PolicyError(f"threshold must be finite, got {parts[2]!r}") from exc
    return Guard(metric=parts[0], op=parts[1], threshold=threshold)


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
        self._guard_metrics = frozenset(
            t.guard.metric for t in transitions if t.guard is not None
        )
        self._validate()

    def _validate(self) -> None:
        """Validate and normalise the Petri-net definition, else raise."""
        for t in self._transitions:
            for arc in t.inputs + t.outputs:
                if arc.place not in self._place_names:
                    raise PolicyError(
                        f"transition {t.name!r} references unknown place {arc.place!r}"
                    )

    @property
    def place_names(self) -> frozenset[str]:
        """All place names registered in this net.

        Returns
        -------
        frozenset[str]
            All place names registered in this net.
        """
        return self._place_names

    @property
    def transitions(self) -> list[Transition]:
        """All transitions in firing-priority order.

        Returns
        -------
        list[Transition]
            All transitions in firing-priority order.
        """
        return list(self._transitions)

    @property
    def guard_metrics(self) -> frozenset[str]:
        """Whitelisted context metric names used by transition guards.

        Returns
        -------
        frozenset[str]
            Whitelisted context metric names used by transition guards.
        """
        return self._guard_metrics

    def _validated_context(self, ctx: Mapping[str, float]) -> dict[str, float]:
        """Return the validated transition-firing context, else raise."""
        if not isinstance(ctx, Mapping):
            raise PolicyError(f"ctx must be a mapping, got {ctx!r}")
        validated: dict[str, float] = {}
        for metric, value in ctx.items():
            metric = _validate_name(metric, kind="metric")
            if metric not in self._guard_metrics:
                raise PolicyError(f"unknown guard context metric {metric!r}")
            validated[metric] = _validate_finite_real(
                value,
                name=f"context metric {metric!r}",
            )
        return validated

    def enabled(self, marking: Marking, ctx: Mapping[str, float]) -> list[Transition]:
        """Return all transitions whose input arcs and guards are satisfied.

        Parameters
        ----------
        marking : Marking
            The Petri net marking (token distribution).
        ctx : Mapping[str, float]
            Context metric values keyed by guard-metric name.

        Returns
        -------
        list[Transition]
            The transitions whose input arcs and guards are satisfied.
        """
        ctx = self._validated_context(ctx)
        result = []
        for t in self._transitions:
            if t.guard is not None and not t.guard.evaluate(ctx):
                continue
            if all(marking[arc.place] >= arc.weight for arc in t.inputs):
                result.append(t)
        return result

    def fire(self, marking: Marking, transition: Transition) -> Marking:
        """Fire *transition*, consuming input tokens and producing output tokens.

        Parameters
        ----------
        marking : Marking
            The Petri net marking (token distribution).
        transition : Transition
            The transition to fire.

        Returns
        -------
        Marking
            The marking after firing the transition.
        """
        new = marking.copy()
        for arc in transition.inputs:
            new[arc.place] = new[arc.place] - arc.weight
        for arc in transition.outputs:
            new[arc.place] = new[arc.place] + arc.weight
        return new

    def step(
        self, marking: Marking, ctx: Mapping[str, float]
    ) -> tuple[Marking, Transition | None]:
        """Fire the first enabled transition, return (new_marking, fired_transition).

        Parameters
        ----------
        marking : Marking
            The Petri net marking (token distribution).
        ctx : Mapping[str, float]
            Context metric values keyed by guard-metric name.

        Returns
        -------
        tuple[Marking, Transition | None]
            The new marking and the fired transition (or ``None``).
        """
        ctx = self._validated_context(ctx)
        for t in self._transitions:
            if t.guard is not None and not t.guard.evaluate(ctx):
                continue
            if all(marking[arc.place] >= arc.weight for arc in t.inputs):
                return self.fire(marking, t), t
        return marking, None
