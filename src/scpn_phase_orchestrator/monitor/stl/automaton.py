# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL monitoring automaton synthesis

"""Monitoring automaton states, transitions, and always/eventually synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .monitor import (
    FloatArray,
    _format_threshold,
    _parse_simple_spec,
    _pointwise_robustness,
    _validate_trace,
)


@dataclass(frozen=True)
class STLAutomatonState:
    """State in a synthesized STL monitoring automaton."""

    name: str
    accepting: bool
    violation: bool
    first_hit_index: int | None = None

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable automaton-state payload.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable automaton-state payload.
        """
        return {
            "name": self.name,
            "accepting": self.accepting,
            "violation": self.violation,
            "first_hit_index": self.first_hit_index,
        }


@dataclass(frozen=True)
class STLAutomatonTransition:
    """Trace-indexed transition taken by a runtime STL automaton."""

    source: str
    target: str
    time_index: int
    guard: str
    robustness: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable automaton-transition payload.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable automaton-transition payload.
        """
        return {
            "source": self.source,
            "target": self.target,
            "time_index": self.time_index,
            "guard": self.guard,
            "robustness": self.robustness,
        }


@dataclass(frozen=True)
class STLMonitoringAutomaton:
    """Audit-ready runtime automaton synthesized from a simple STL monitor."""

    spec: str
    temporal_op: str
    signals: tuple[str, ...]
    states: tuple[STLAutomatonState, ...]
    transitions: tuple[STLAutomatonTransition, ...]
    robustness: float
    satisfied: bool
    backend: str = "builtin"

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable STL automaton audit payload.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable STL automaton audit payload.
        """
        return {
            "spec": self.spec,
            "temporal_op": self.temporal_op,
            "signals": list(self.signals),
            "states": [state.to_audit_record() for state in self.states],
            "transitions": [
                transition.to_audit_record() for transition in self.transitions
            ],
            "robustness": self.robustness,
            "satisfied": self.satisfied,
            "backend": self.backend,
        }


def synthesise_stl_monitoring_automaton(
    spec: str,
    trace: dict[str, list[float]],
) -> STLMonitoringAutomaton:
    """Synthesize a trace automaton for builtin simple STL safety formulas.

    The synthesized automaton is intentionally conservative and audit-oriented:
    it records the state sequence taken by the monitor over the supplied trace
    for supported ``always (...)`` and ``eventually (...)`` conjunctions. More
    expressive STL remains delegated to ``rtamt`` for robustness evaluation.

    Parameters
    ----------
    spec : str
        STL specification string.
    trace : dict[str, list[float]]
        Signal trace keyed by variable name, each a list of floats.

    Returns
    -------
    STLMonitoringAutomaton
        The trace automaton for the STL safety formula.

    Raises
    ------
    ValueError
        If the spec is not a supported builtin STL formula.
    """
    _validate_trace(trace)
    parsed = _parse_simple_spec(spec)
    if parsed is None:
        raise ValueError(
            "monitoring automata synthesis supports builtin simple STL syntax only"
        )

    temporal_op, predicates = parsed
    pointwise = _pointwise_robustness(predicates, trace)
    guard = _format_predicate_guard(predicates)
    signals = tuple(dict.fromkeys(signal for signal, _, _ in predicates))

    if temporal_op == "always":
        robustness = float(np.min(pointwise))
        return _synthesise_always_automaton(
            spec=spec,
            signals=signals,
            pointwise=pointwise,
            guard=guard,
            robustness=robustness,
        )
    if temporal_op == "eventually":
        robustness = float(np.max(pointwise))
        return _synthesise_eventually_automaton(
            spec=spec,
            signals=signals,
            pointwise=pointwise,
            guard=guard,
            robustness=robustness,
        )
    raise ValueError(f"unsupported STL temporal operator {temporal_op!r}")


synthesize_stl_monitoring_automaton = synthesise_stl_monitoring_automaton


def _validate_horizon_steps(horizon_steps: int) -> None:
    """Return the horizon length as a positive integer, else raise."""
    if (
        isinstance(horizon_steps, bool)
        or not isinstance(horizon_steps, int)
        or horizon_steps <= 0
    ):
        raise ValueError("closed-loop horizon_steps must be a positive integer")


def _format_predicate_guard(predicates: list[tuple[str, str, float]]) -> str:
    """Return a stable string rendering of a predicate guard."""
    return " and ".join(
        f"{signal} {op} {_format_threshold(threshold)}"
        for signal, op, threshold in predicates
    )


def _synthesise_always_automaton(
    *,
    spec: str,
    signals: tuple[str, ...],
    pointwise: FloatArray,
    guard: str,
    robustness: float,
) -> STLMonitoringAutomaton:
    """Build the monitoring automaton for an 'always' STL operator."""
    transitions: list[STLAutomatonTransition] = []
    state = "holding"
    first_violation_index: int | None = None
    for time_index, value in enumerate(pointwise):
        target = "violated" if value < 0.0 else state
        if target == "violated" and first_violation_index is None:
            first_violation_index = time_index
        transitions.append(
            STLAutomatonTransition(
                source=state,
                target=target,
                time_index=time_index,
                guard=guard,
                robustness=float(value),
            )
        )
        state = target

    states = (
        STLAutomatonState(
            name="holding",
            accepting=first_violation_index is None,
            violation=False,
        ),
        STLAutomatonState(
            name="violated",
            accepting=False,
            violation=first_violation_index is not None,
            first_hit_index=first_violation_index,
        ),
    )
    return STLMonitoringAutomaton(
        spec=spec,
        temporal_op="always",
        signals=signals,
        states=states,
        transitions=tuple(transitions),
        robustness=robustness,
        satisfied=robustness >= 0.0,
    )


def _synthesise_eventually_automaton(
    *,
    spec: str,
    signals: tuple[str, ...],
    pointwise: FloatArray,
    guard: str,
    robustness: float,
) -> STLMonitoringAutomaton:
    """Build the monitoring automaton for an 'eventually' STL operator."""
    transitions: list[STLAutomatonTransition] = []
    state = "pending"
    first_satisfaction_index: int | None = None
    for time_index, value in enumerate(pointwise):
        target = "satisfied" if value >= 0.0 else state
        if target == "satisfied" and first_satisfaction_index is None:
            first_satisfaction_index = time_index
        transitions.append(
            STLAutomatonTransition(
                source=state,
                target=target,
                time_index=time_index,
                guard=guard,
                robustness=float(value),
            )
        )
        state = target

    states = (
        STLAutomatonState(
            name="pending",
            accepting=False,
            violation=first_satisfaction_index is None,
        ),
        STLAutomatonState(
            name="satisfied",
            accepting=first_satisfaction_index is not None,
            violation=False,
            first_hit_index=first_satisfaction_index,
        ),
    )
    return STLMonitoringAutomaton(
        spec=spec,
        temporal_op="eventually",
        signals=signals,
        states=states,
        transitions=tuple(transitions),
        robustness=robustness,
        satisfied=robustness >= 0.0,
    )
