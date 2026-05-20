# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Offline evolutionary Petri-net mutation grammar

"""Review-only offline evolutionary mutation grammar for Petri-net topologies.

This module produces deterministic mutation candidates and plans from a simple,
normalised net descriptor. It performs no execution, no actuation, and never
commits graph changes itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Literal

__all__ = [
    "EvolutionaryPetriMutationCandidate",
    "EvolutionaryPetriMutationConfig",
    "EvolutionaryPetriMutationPlan",
    "run_offline_evolutionary_petri_mutation_grammar",
]

MutationType = Literal["add_arc", "guard_weight", "token_bound"]


def _validate_non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _validate_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _validate_finite_real(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be finite")
    value_f = float(value)
    if not (value_f == value_f and value_f not in (float("inf"), float("-inf"))):
        raise ValueError(f"{field} must be finite")
    return value_f


def _validate_name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty string")
    return value


def _validate_guard_metric(value: object) -> str:
    return _validate_name(value, field="guard metric")


def _build_stable_hash(payload: Mapping[str, Any] | object) -> str:
    clean: object
    if isinstance(payload, Mapping):
        clean = dict(payload)
        clean.pop("candidate_hash", None)
        clean.pop("plan_hash", None)
    else:
        clean = payload
    return hashlib.sha256(
        json.dumps(clean, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _PetriNetPlace:
    """Internal normalised representation of one Petri place."""

    name: str
    token_bound: int

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "token_bound": self.token_bound,
        }


@dataclass(frozen=True)
class _PetriNetTransition:
    """Internal normalised Petri transition with guard weights."""

    name: str
    guard_weights: tuple[tuple[str, float], ...]

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "guard_weights": [
                {"metric": m, "weight": w} for m, w in self.guard_weights
            ],
        }


@dataclass(frozen=True)
class _PetriNetArc:
    """Internal normalised arc in the net grammar."""

    place: str
    transition: str
    direction: Literal["input", "output"]
    weight: int

    def to_record(self) -> dict[str, object]:
        return {
            "place": self.place,
            "transition": self.transition,
            "direction": self.direction,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class _PetriNetSpec:
    """Normalised net snapshot for mutation generation."""

    places: tuple[_PetriNetPlace, ...]
    transitions: tuple[_PetriNetTransition, ...]
    arcs: tuple[_PetriNetArc, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "places": [place.to_record() for place in self.places],
            "transitions": [transition.to_record() for transition in self.transitions],
            "arcs": [arc.to_record() for arc in self.arcs],
        }


@dataclass(frozen=True)
class EvolutionaryPetriMutationConfig:
    """Mutation generation configuration for the offline Petri grammar."""

    generation_count: int = 2
    candidates_per_generation: int = 6
    mutation_step: float = 0.1
    max_arc_weight: int = 4
    max_token_bound: int = 128

    def __post_init__(self) -> None:
        _validate_positive_int(self.generation_count, field="generation_count")
        _validate_positive_int(
            self.candidates_per_generation,
            field="candidates_per_generation",
        )
        _validate_finite_real(self.mutation_step, field="mutation_step")
        if self.mutation_step <= 0.0:
            raise ValueError("mutation_step must be positive")
        _validate_positive_int(self.max_arc_weight, field="max_arc_weight")
        _validate_positive_int(self.max_token_bound, field="max_token_bound")


@dataclass(frozen=True)
class EvolutionaryPetriMutationCandidate:
    """One offline-only Petri-net mutation candidate."""

    candidate_id: str
    generation: int
    mutation_type: MutationType
    mutation_target: str
    mutation_kind: str
    blocked_reasons: tuple[str, ...]
    before: dict[str, object]
    after: dict[str, object]
    mutation_delta: float
    candidate_hash: str
    operator_review_required: bool = True
    execution_disabled: bool = True
    live_merge_permitted: bool = False
    hot_patch_permitted: bool = False
    actuation_permitted: bool = False

    @property
    def accepted(self) -> bool:
        """Return whether this candidate is accepted for review."""
        return not self.blocked_reasons

    @property
    def status(self) -> str:
        """Return the review status label for this candidate."""
        return "accepted" if self.accepted else "blocked"

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record."""
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "mutation_type": self.mutation_type,
            "mutation_target": self.mutation_target,
            "mutation_kind": self.mutation_kind,
            "blocked_reasons": list(self.blocked_reasons),
            "before": dict(self.before),
            "after": dict(self.after),
            "mutation_delta": self.mutation_delta,
            "status": self.status,
            "candidate_hash": self.candidate_hash,
            "operator_review_required": self.operator_review_required,
            "execution_disabled": self.execution_disabled,
            "live_merge_permitted": self.live_merge_permitted,
            "hot_patch_permitted": self.hot_patch_permitted,
            "actuation_permitted": self.actuation_permitted,
        }


@dataclass(frozen=True)
class EvolutionaryPetriMutationPlan:
    """Deterministic, offline review plan for Petri-net grammar search."""

    schema_name: str
    schema_version: str
    config: EvolutionaryPetriMutationConfig
    source_net_hash: str
    candidate_count: int
    accepted_count: int
    rejected_count: int
    candidates: tuple[EvolutionaryPetriMutationCandidate, ...]
    best_candidate_id: str | None
    source_net: dict[str, object]
    operator_review_required: bool
    execution_disabled: bool
    live_merge_permitted: bool
    hot_patch_permitted: bool
    actuation_permitted: bool
    non_actuating: bool
    plan_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record."""
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "generation_count": self.config.generation_count,
            "candidates_per_generation": self.config.candidates_per_generation,
            "mutation_step": self.config.mutation_step,
            "max_arc_weight": self.config.max_arc_weight,
            "max_token_bound": self.config.max_token_bound,
            "source_net_hash": self.source_net_hash,
            "source_net": self.source_net,
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "best_candidate_id": self.best_candidate_id,
            "candidates": [
                candidate.to_audit_record() for candidate in self.candidates
            ],
            "operator_review_required": self.operator_review_required,
            "execution_disabled": self.execution_disabled,
            "live_merge_permitted": self.live_merge_permitted,
            "hot_patch_permitted": self.hot_patch_permitted,
            "actuation_permitted": self.actuation_permitted,
            "non_actuating": self.non_actuating,
            "claim_boundary": "offline_petri_mutation_review_only",
            "plan_hash": self.plan_hash,
        }


def run_offline_evolutionary_petri_mutation_grammar(
    net_like: Mapping[str, object] | Sequence[object],
    *,
    generation_count: int = 2,
    candidates_per_generation: int = 6,
    mutation_step: float = 0.1,
    max_arc_weight: int = 4,
    max_token_bound: int = 128,
) -> EvolutionaryPetriMutationPlan:
    """Build a deterministic review-only mutation plan from a net-like payload."""

    config = EvolutionaryPetriMutationConfig(
        generation_count=generation_count,
        candidates_per_generation=candidates_per_generation,
        mutation_step=mutation_step,
        max_arc_weight=max_arc_weight,
        max_token_bound=max_token_bound,
    )

    spec = _normalise_net_like(net_like)
    base_record = spec.to_record()
    source_net_hash = _build_stable_hash(base_record)

    candidates: list[EvolutionaryPetriMutationCandidate] = []
    for generation in range(config.generation_count):
        for local_index in range(config.candidates_per_generation):
            mutation_type: MutationType = _mutation_type_at_index(
                generation=generation,
                local_index=local_index,
            )
            candidate_index = (
                generation * config.candidates_per_generation + local_index
            )
            if mutation_type == "add_arc":
                candidate = _build_add_arc_candidate(
                    spec,
                    config,
                    generation,
                    local_index,
                )
            elif mutation_type == "guard_weight":
                candidate = _build_guard_weight_candidate(
                    spec,
                    config,
                    generation,
                    local_index,
                    candidate_index,
                )
            elif mutation_type == "token_bound":
                candidate = _build_token_bound_candidate(
                    spec,
                    config,
                    generation,
                    local_index,
                    candidate_index,
                )
            else:  # pragma: no cover - defensive guard for typing completeness
                raise ValueError(f"unsupported mutation type {mutation_type}")
            candidates.append(
                replace(
                    candidate,
                    candidate_id=f"g{generation + 1:03d}-c{local_index + 1:03d}",
                    candidate_hash=_build_stable_hash(candidate.to_audit_record()),
                )
            )

    accepted = [candidate for candidate in candidates if candidate.accepted]
    best_candidate_id = max(
        accepted,
        key=lambda candidate: _candidate_score(candidate),
        default=None,
    )
    best_id = best_candidate_id.candidate_id if best_candidate_id else None

    report = EvolutionaryPetriMutationPlan(
        schema_name="evolutionary_petri_mutation_grammar",
        schema_version="0.1.0",
        config=config,
        source_net_hash=source_net_hash,
        candidate_count=len(candidates),
        accepted_count=len(accepted),
        rejected_count=len(candidates) - len(accepted),
        candidates=tuple(candidates),
        best_candidate_id=best_id,
        source_net=base_record,
        operator_review_required=True,
        execution_disabled=True,
        live_merge_permitted=False,
        hot_patch_permitted=False,
        actuation_permitted=False,
        non_actuating=True,
        plan_hash="",
    )

    return EvolutionaryPetriMutationPlan(
        schema_name=report.schema_name,
        schema_version=report.schema_version,
        config=report.config,
        source_net_hash=report.source_net_hash,
        candidate_count=report.candidate_count,
        accepted_count=report.accepted_count,
        rejected_count=report.rejected_count,
        candidates=report.candidates,
        best_candidate_id=report.best_candidate_id,
        source_net=report.source_net,
        operator_review_required=report.operator_review_required,
        execution_disabled=report.execution_disabled,
        live_merge_permitted=report.live_merge_permitted,
        hot_patch_permitted=report.hot_patch_permitted,
        actuation_permitted=report.actuation_permitted,
        non_actuating=report.non_actuating,
        plan_hash=_build_stable_hash(report.to_audit_record()),
    )


def _mutation_type_at_index(*, generation: int, local_index: int) -> MutationType:
    index = generation % 3 * 2 + local_index % 3
    if index % 3 == 0:
        return "add_arc"
    if index % 3 == 1:
        return "guard_weight"
    return "token_bound"


def _candidate_score(candidate: EvolutionaryPetriMutationCandidate) -> float:
    if not candidate.accepted:
        return float("-inf")
    return -abs(candidate.mutation_delta)


def _normalise_net_like(
    net_like: Mapping[str, object] | Sequence[object],
) -> _PetriNetSpec:
    if not isinstance(net_like, (Mapping, Sequence)) or isinstance(
        net_like, (str, bytes, bytearray)
    ):
        raise ValueError("net_like must be a mapping or a sequence of net records")

    if isinstance(net_like, Mapping):
        if "places" in net_like or "transitions" in net_like or "arcs" in net_like:
            return _parse_net_mapping(net_like)
        return _parse_simple_record(net_like)

    return _parse_net_sequence(list(net_like))


def _parse_net_mapping(mapping: Mapping[str, object]) -> _PetriNetSpec:
    places_raw = mapping.get("places", ())
    transitions_raw = mapping.get("transitions", ())
    arcs_raw = mapping.get("arcs", ())

    places = _parse_places(places_raw)
    transitions = _parse_transitions(transitions_raw)
    place_names = {p.name for p in places}
    transition_names = {t.name for t in transitions}
    arcs = _parse_arcs(
        arcs_raw,
        place_names=place_names,
        transition_names=transition_names,
    )
    return _PetriNetSpec(
        places=tuple(places),
        transitions=tuple(transitions),
        arcs=tuple(arcs),
    )


def _parse_net_sequence(records: Sequence[object]) -> _PetriNetSpec:
    if not records:
        raise ValueError("net-like sequence must not be empty")

    places: list[_PetriNetPlace] = []
    transitions: list[_PetriNetTransition] = []
    arcs: list[_PetriNetArc] = []

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each net record must be a mapping with kind")
        kind = record.get("kind")
        if kind == "place":
            places.append(_parse_place_record(record))
        elif kind == "transition":
            transitions.append(_parse_transition_record(record))
        elif kind == "arc":
            # Temporary store with empty sets until we can validate references.
            arc = _parse_arc_record(record)
            arcs.append(arc)
        else:
            raise ValueError(f"unknown record kind {kind!r}")

    place_names = {place.name for place in places}
    transition_names = {transition.name for transition in transitions}
    _validate_arc_references(
        arcs,
        place_names=place_names,
        transition_names=transition_names,
    )
    return _PetriNetSpec(
        places=tuple(places),
        transitions=tuple(transitions),
        arcs=tuple(arcs),
    )


def _parse_simple_record(record: Mapping[str, object]) -> _PetriNetSpec:
    kind = record.get("kind")
    if kind == "place":
        place = _parse_place_record(record)
        return _PetriNetSpec(
            places=(place,),
            transitions=(),
            arcs=(),
        )
    if kind == "transition":
        transition = _parse_transition_record(record)
        return _PetriNetSpec(
            places=(),
            transitions=(transition,),
            arcs=(),
        )
    if kind == "arc":
        arc = _parse_arc_record(record)
        return _PetriNetSpec(
            places=(),
            transitions=(),
            arcs=(arc,),
        )

    raise ValueError(
        "simple net record must include kind 'place', 'transition' or 'arc'"
    )


def _parse_places(raw: object) -> list[_PetriNetPlace]:
    if isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("places must be a sequence")
    if not isinstance(raw, Sequence):
        raise ValueError("places must be a sequence")
    places = [_parse_place_record(item) for item in raw]
    if not places:
        raise ValueError("places must be non-empty")
    return sorted(places, key=lambda place: place.name)


def _parse_place_record(record: object) -> _PetriNetPlace:
    if isinstance(record, str):
        name = _validate_name(record, field="place name")
        return _PetriNetPlace(name=name, token_bound=1)
    if not isinstance(record, Mapping):
        raise ValueError("place record must be a string name or a mapping")
    name = _validate_name(record.get("name"), field="place name")
    token_bound = _validate_non_negative_int(
        record.get("token_bound", 1),
        field="token_bound",
    )
    return _PetriNetPlace(name=name, token_bound=token_bound)


def _parse_transitions(raw: object) -> list[_PetriNetTransition]:
    if isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("transitions must be a sequence")
    if not isinstance(raw, Sequence):
        raise ValueError("transitions must be a sequence")
    transitions = [_parse_transition_record(item) for item in raw]
    if not transitions:
        raise ValueError("transitions must be non-empty")
    return sorted(transitions, key=lambda transition: transition.name)


def _parse_transition_record(record: object) -> _PetriNetTransition:
    if isinstance(record, str):
        name = _validate_name(record, field="transition name")
        return _PetriNetTransition(name=name, guard_weights=())
    if not isinstance(record, Mapping):
        raise ValueError("transition record must be a string name or a mapping")

    name = _validate_name(record.get("name"), field="transition name")
    guards_raw = record.get("guard_weights", ())

    if isinstance(guards_raw, Mapping):
        guard_weights = [
            (
                _validate_guard_metric(metric),
                _validate_finite_real(weight, field="guard weight"),
            )
            for metric, weight in guards_raw.items()
        ]
    elif guards_raw:
        if not isinstance(guards_raw, Sequence) or isinstance(
            guards_raw, (str, bytes, bytearray)
        ):
            raise ValueError("guard_weights must be a mapping or a sequence")
        parsed: list[tuple[str, float]] = []
        for item in guards_raw:
            if not isinstance(item, Mapping):
                raise ValueError("guard entry must be a mapping")
            parsed.append(
                (
                    _validate_guard_metric(item.get("metric")),
                    _validate_finite_real(item.get("weight"), field="guard weight"),
                )
            )
        guard_weights = parsed
    else:
        guard_weights = []

    guard_weights = [
        (metric, float(weight)) for metric, weight in guard_weights if metric.strip()
    ]
    # Preserve deterministic order for hashing and mutation schedule.
    return _PetriNetTransition(name=name, guard_weights=tuple(sorted(guard_weights)))


def _parse_arcs(
    raw: object,
    *,
    place_names: set[str],
    transition_names: set[str],
) -> list[_PetriNetArc]:
    if isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("arcs must be a sequence")
    if not isinstance(raw, Sequence):
        raise ValueError("arcs must be a sequence")
    arcs = [_parse_arc_record(item) for item in raw]
    _validate_arc_references(
        arcs,
        place_names=place_names,
        transition_names=transition_names,
    )
    return sorted(
        arcs,
        key=lambda arc: (arc.place, arc.transition, arc.direction, arc.weight),
    )


def _parse_arc_record(item: object) -> _PetriNetArc:
    if isinstance(item, Mapping):
        place = _validate_name(item.get("place"), field="arc.place")
        transition = _validate_name(item.get("transition"), field="arc.transition")
        direction = _validate_arc_direction(item.get("direction"))
        weight = _validate_positive_int(item.get("weight", 1), field="arc.weight")
        return _PetriNetArc(
            place=place,
            transition=transition,
            direction=direction,
            weight=weight,
        )

    if isinstance(item, Sequence):
        if len(item) < 3:
            raise ValueError(
                "arc sequence record needs at least place, transition, direction"
            )
        place = _validate_name(item[0], field="arc.place")
        transition = _validate_name(item[1], field="arc.transition")
        direction = _validate_arc_direction(item[2])
        weight = _validate_positive_int(
            item[3] if len(item) > 3 else 1,
            field="arc.weight",
        )
        return _PetriNetArc(
            place=place,
            transition=transition,
            direction=direction,
            weight=weight,
        )

    raise ValueError("arc record must be a mapping or sequence")


def _validate_arc_direction(value: object) -> Literal["input", "output"]:
    if value not in {"input", "output"}:
        raise ValueError("arc.direction must be 'input' or 'output'")
    if value == "input":
        return "input"
    return "output"


def _validate_arc_references(
    arcs: Sequence[_PetriNetArc],
    *,
    place_names: set[str],
    transition_names: set[str],
) -> None:
    for arc in arcs:
        if arc.place not in place_names:
            raise ValueError(f"arc references unknown place {arc.place!r}")
        if arc.transition not in transition_names:
            raise ValueError(f"arc references unknown transition {arc.transition!r}")


def _mutation_delta(index: int, *, step: float, max_delta: int) -> float:
    unit = step if step <= 1.0 else 1.0
    offset = (index % max_delta) + 1
    return unit * float(offset / max_delta)


def _build_add_arc_candidate(
    spec: _PetriNetSpec,
    config: EvolutionaryPetriMutationConfig,
    generation: int,
    local_index: int,
) -> EvolutionaryPetriMutationCandidate:
    if not spec.transitions or not spec.places:
        return EvolutionaryPetriMutationCandidate(
            candidate_id="",
            generation=generation + 1,
            mutation_type="add_arc",
            mutation_target="network",
            mutation_kind="structural",
            blocked_reasons=("net_missing_places_or_transitions",),
            before={"arcs": [arc.to_record() for arc in spec.arcs]},
            after={"arcs": [arc.to_record() for arc in spec.arcs]},
            mutation_delta=0.0,
            candidate_hash="",
        )

    transitions = [transition.name for transition in spec.transitions]
    places = [place.name for place in spec.places]
    existing = {(arc.place, arc.transition, arc.direction) for arc in spec.arcs}

    selected_transition = transitions[
        (generation * config.candidates_per_generation + local_index) % len(transitions)
    ]
    selected_place = places[(generation + local_index * 2) % len(places)]
    direction: Literal["input", "output"] = (
        "input" if (generation + local_index) % 2 == 0 else "output"
    )
    weight = 1 + (
        (generation * config.candidates_per_generation + local_index)
        % config.max_arc_weight
    )

    # Deterministic fallback: skip taken arcs by cycling over places and transitions.
    candidate_tuple: tuple[str, str, Literal["input", "output"]] | None = (
        selected_place,
        selected_transition,
        direction,
    )
    if candidate_tuple in existing:
        candidate_tuple = _first_free_arc_tuple(
            spec,
            start=candidate_tuple,
            config=config,
            prefer_direction=direction,
        )

    if candidate_tuple is None:
        return EvolutionaryPetriMutationCandidate(
            candidate_id="",
            generation=generation + 1,
            mutation_type="add_arc",
            mutation_target="network",
            mutation_kind="structural",
            blocked_reasons=("add_arc_no_available_target",),
            before={"arcs": [arc.to_record() for arc in spec.arcs]},
            after={"arcs": [arc.to_record() for arc in spec.arcs]},
            mutation_delta=0.0,
            candidate_hash="",
        )

    after_arc = _PetriNetArc(
        place=candidate_tuple[0],
        transition=candidate_tuple[1],
        direction=candidate_tuple[2],
        weight=weight,
    )
    after_arc_dict = after_arc.to_record()
    new_arcs = [arc.to_record() for arc in spec.arcs]
    new_arcs.append(after_arc_dict)
    return EvolutionaryPetriMutationCandidate(
        candidate_id="",
        generation=generation + 1,
        mutation_type="add_arc",
        mutation_target=after_arc.place,
        mutation_kind="add_arc",
        blocked_reasons=(),
        before={
            "place": selected_place,
            "transition": selected_transition,
            "direction": direction,
            "arc_weight": None,
            "arcs": [arc.to_record() for arc in spec.arcs],
        },
        after={
            "place": selected_place,
            "transition": selected_transition,
            "direction": direction,
            "arc_weight": weight,
            "arcs": sorted(
                new_arcs,
                key=lambda item: (
                    item["place"],
                    item["transition"],
                    item["direction"],
                    item["weight"],
                ),
            ),
        },
        mutation_delta=float(weight),
        candidate_hash="",
    )


def _first_free_arc_tuple(
    spec: _PetriNetSpec,
    *,
    start: tuple[str, str, Literal["input", "output"]],
    config: EvolutionaryPetriMutationConfig,
    prefer_direction: Literal["input", "output"],
) -> tuple[str, str, Literal["input", "output"]] | None:
    transitions = [transition.name for transition in spec.transitions]
    places = [place.name for place in spec.places]
    if prefer_direction == "input":
        direction_cycle: tuple[Literal["input", "output"], ...] = (
            prefer_direction,
            "output",
            "input",
        )
    else:
        direction_cycle = (prefer_direction, "input", "output")
    occupied = {(arc.place, arc.transition, arc.direction): None for arc in spec.arcs}

    for base in range(len(places) * len(transitions) * 2):
        direction = direction_cycle[base % len(direction_cycle)]
        place = places[(base + len(start[0])) % len(places)]
        transition = transitions[(base * 2 + len(start[1])) % len(transitions)]
        key = (place, transition, direction)
        if key not in occupied:
            return place, transition, direction
    return None


def _build_guard_weight_candidate(
    spec: _PetriNetSpec,
    config: EvolutionaryPetriMutationConfig,
    generation: int,
    local_index: int,
    candidate_index: int,
) -> EvolutionaryPetriMutationCandidate:
    if not spec.transitions:
        return EvolutionaryPetriMutationCandidate(
            candidate_id="",
            generation=generation + 1,
            mutation_type="guard_weight",
            mutation_target="network",
            mutation_kind="guard",
            blocked_reasons=("net_missing_transitions",),
            before={
                "transitions": [
                    transition.to_record() for transition in spec.transitions
                ]
            },
            after={
                "transitions": [
                    transition.to_record() for transition in spec.transitions
                ]
            },
            mutation_delta=0.0,
            candidate_hash="",
        )

    transitions = list(spec.transitions)
    transition = transitions[candidate_index % len(transitions)]

    guard_candidates = list(transition.guard_weights)
    if guard_candidates:
        metric, previous = guard_candidates[candidate_index % len(guard_candidates)]
        metric_id = metric
    else:
        metric_id = "safety_margin"
        previous = 0.0

    direction = -1.0 if ((generation + local_index) % 2) else 1.0
    delta = direction * _mutation_delta(
        candidate_index,
        step=config.mutation_step,
        max_delta=config.max_arc_weight,
    )
    after = previous + delta
    return EvolutionaryPetriMutationCandidate(
        candidate_id="",
        generation=generation + 1,
        mutation_type="guard_weight",
        mutation_target=transition.name,
        mutation_kind="guard",
        blocked_reasons=(),
        before={
            "transition": transition.name,
            "metric": metric_id,
            "weight": previous,
            "all_guards": [
                {"metric": m, "weight": w} for m, w in transition.guard_weights
            ],
        },
        after={
            "transition": transition.name,
            "metric": metric_id,
            "weight": after,
            "all_guards": [
                {"metric": m, "weight": float(w) if m != metric_id else float(after)}
                for m, w in transition.guard_weights
                if guard_candidates
            ]
            or [{"metric": metric_id, "weight": after}]
            if not guard_candidates
            else [],
        },
        mutation_delta=float(after - previous),
        candidate_hash="",
    )


def _build_token_bound_candidate(
    spec: _PetriNetSpec,
    config: EvolutionaryPetriMutationConfig,
    generation: int,
    local_index: int,
    candidate_index: int,
) -> EvolutionaryPetriMutationCandidate:
    if not spec.places:
        return EvolutionaryPetriMutationCandidate(
            candidate_id="",
            generation=generation + 1,
            mutation_type="token_bound",
            mutation_target="network",
            mutation_kind="bounds",
            blocked_reasons=("net_missing_places",),
            before={"places": [place.to_record() for place in spec.places]},
            after={"places": [place.to_record() for place in spec.places]},
            mutation_delta=0.0,
            candidate_hash="",
        )

    places = list(spec.places)
    place = places[candidate_index % len(places)]
    delta = 1 if ((generation + local_index) % 2 == 0) else -1
    magnitude = max(1, int(round(config.mutation_step * config.max_token_bound)))
    after_bound = place.token_bound + delta * magnitude
    if after_bound < 0:
        after_bound = 0
    new_place = _PetriNetPlace(name=place.name, token_bound=after_bound)

    new_places = [p.to_record() for p in places if p.name != place.name]
    if new_place.to_record() not in new_places:
        new_places.append(new_place.to_record())

    blocked_reasons: tuple[str, ...] = ()
    if after_bound > config.max_token_bound:
        blocked_reasons = ("token_bound_exceeds_policy_max",)

    return EvolutionaryPetriMutationCandidate(
        candidate_id="",
        generation=generation + 1,
        mutation_type="token_bound",
        mutation_target=place.name,
        mutation_kind="bounds",
        blocked_reasons=blocked_reasons,
        before={"place": place.name, "token_bound": place.token_bound},
        after={"place": new_place.name, "token_bound": new_place.token_bound},
        mutation_delta=float(after_bound - place.token_bound),
        candidate_hash="",
    )
