# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL controller candidate synthesis

"""Controller candidate and synthesis generation from STL predicates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .automaton import STLMonitoringAutomaton
from .monitor import (
    _format_threshold,
    _parse_simple_spec,
    _pointwise_robustness,
    _predicate_robustness,
    _validate_trace,
)


@dataclass(frozen=True)
class STLControllerCandidate:
    """Non-actuating controller candidate derived from an STL automaton."""

    signal: str
    action: str
    direction: str
    time_index: int
    robustness: float
    rationale: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable controller-candidate payload.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable controller-candidate payload.
        """
        return {
            "signal": self.signal,
            "action": self.action,
            "direction": self.direction,
            "time_index": self.time_index,
            "robustness": self.robustness,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class STLControllerSynthesis:
    """Audit-ready, non-actuating controller synthesis proposal."""

    spec: str
    satisfied: bool
    actuating: bool
    source_backend: str
    candidates: tuple[STLControllerCandidate, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable controller-synthesis payload.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable controller-synthesis payload.
        """
        return {
            "spec": self.spec,
            "satisfied": self.satisfied,
            "actuating": self.actuating,
            "source_backend": self.source_backend,
            "candidates": [
                candidate.to_audit_record() for candidate in self.candidates
            ],
        }


def synthesise_stl_controller_candidates(
    automaton: STLMonitoringAutomaton,
    trace: dict[str, list[float]],
    *,
    action_map: dict[str, str] | None = None,
) -> STLControllerSynthesis:
    """Synthesize non-actuating controller candidates from an STL automaton.

    The result is a review artefact, not a controller. It identifies the
    weakest predicate margin and proposes signal-level adjustment directions for
    supported builtin ``always`` and ``eventually`` monitors. Callers must still
    map candidates through policy, projection, safety, and actuation gates.

    Parameters
    ----------
    automaton : STLMonitoringAutomaton
        The STL monitoring automaton.
    trace : dict[str, list[float]]
        Signal trace keyed by variable name, each a list of floats.
    action_map : dict[str, str] | None
        Mapping of automaton state to action name, or ``None``.

    Returns
    -------
    STLControllerSynthesis
        The non-actuating controller-candidate synthesis.

    Raises
    ------
    ValueError
        If the automaton or trace is invalid.
    """
    _validate_trace(trace)
    parsed = _parse_simple_spec(automaton.spec)
    if parsed is None:
        raise ValueError("controller synthesis supports builtin simple STL syntax only")
    temporal_op, predicates = parsed
    if temporal_op != automaton.temporal_op:
        raise ValueError("automaton temporal operator does not match its STL spec")
    index = _controller_focus_index(automaton, predicates, trace)
    candidates = tuple(
        candidate
        for predicate in predicates
        if (
            candidate := _candidate_for_predicate(
                predicate,
                trace,
                time_index=index,
                action_map=action_map or {},
            )
        )
        is not None
    )
    if automaton.satisfied:
        candidates = ()
    return STLControllerSynthesis(
        spec=automaton.spec,
        satisfied=automaton.satisfied,
        actuating=False,
        source_backend=automaton.backend,
        candidates=candidates,
    )


synthesize_stl_controller_candidates = synthesise_stl_controller_candidates


def _controller_focus_index(
    automaton: STLMonitoringAutomaton,
    predicates: list[tuple[str, str, float]],
    trace: dict[str, list[float]],
) -> int:
    pointwise = _pointwise_robustness(predicates, trace)
    if automaton.temporal_op == "always":
        return int(np.argmin(pointwise))
    if automaton.temporal_op == "eventually":
        return int(np.argmax(pointwise))
    raise ValueError(f"unsupported STL temporal operator {automaton.temporal_op!r}")


def _candidate_for_predicate(
    predicate: tuple[str, str, float],
    trace: dict[str, list[float]],
    *,
    time_index: int,
    action_map: dict[str, str],
) -> STLControllerCandidate | None:
    signal, op, threshold = predicate
    robustness = float(_predicate_robustness(signal, op, threshold, trace)[time_index])
    if robustness >= 0.0:
        return None
    direction = _controller_direction(op)
    action = action_map.get(signal, f"{direction}_{signal}")
    rationale = (
        f"{signal} {op} {_format_threshold(threshold)} violated at "
        f"t={time_index} with robustness {robustness:g}"
    )
    return STLControllerCandidate(
        signal=signal,
        action=action,
        direction=direction,
        time_index=time_index,
        robustness=robustness,
        rationale=rationale,
    )


def _controller_direction(op: str) -> str:
    if op in {">=", ">"}:
        return "increase"
    if op in {"<=", "<"}:
        return "decrease"
    if op == "==":
        return "restore"
    raise ValueError(f"unsupported STL comparison operator {op!r}")
