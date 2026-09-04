# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Diagnostic-plan 1.2 declaration-depth validation
"""Validate declaration-only signal, frame, and clock depth for plan 1.2."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import NoReturn, TypeGuard

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.]*$")
_SIGNAL_KEYS = frozenset({"description", "identifier", "quantity", "role", "unit"})
_TRANSFORMATION_KEYS = frozenset(
    {
        "equilibrium_dependent",
        "evidence_claimed",
        "kind",
        "method",
        "source_identifier",
        "target_identifier",
    }
)
_DOMAIN_KEYS = frozenset(
    {"identifier", "member_clock_identifiers", "root_clock_identifier", "scope"}
)
_TOPOLOGY_KEYS = frozenset({"domains", "reference_domain_identifier"})
_SIGNAL_ROLES = frozenset({"amplitude", "auxiliary", "carrier", "timing_marker"})
_TRANSFORMATION_KINDS = frozenset({"flux_mapping", "projection", "rigid"})
_ADMISSIBLE_TRANSFORMATIONS = {
    frozenset({"machine_cylindrical", "flux_surface"}): "flux_mapping",
    frozenset({"flux_surface", "boozer"}): "flux_mapping",
    frozenset({"field_line", "machine_cylindrical"}): "flux_mapping",
    frozenset({"blanket_zone", "machine_cylindrical"}): "projection",
    frozenset({"chamber_cartesian", "beamline"}): "rigid",
}


class DiagnosticPlanDepthRefusalKind(StrEnum):
    """Boundary category to preserve the parent intake's refusal vocabulary."""

    PLAN = "plan"
    CARRIER = "carrier"
    CLOCK = "clock"
    AUTHORITY = "authority"


class DiagnosticPlanDepthError(ValueError):
    """Raised when a 1.2 declaration violates its exact depth contract."""

    def __init__(self, kind: DiagnosticPlanDepthRefusalKind, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


def validate_diagnostic_plan_depth(
    plan: Mapping[str, object],
    *,
    candidate_classes: Mapping[str, str],
    clock_kinds: Mapping[str, str],
    frame_kinds: Mapping[str, str],
) -> None:
    """Validate all members added by producer envelope version 1.2.0.

    The declarations remain synthetic design metadata. In particular, signal
    quantity and unit text never select a candidate, change its registered
    carrier, create an observation, or establish a physical phase.
    """
    _validate_signal_inventories(plan, candidate_classes)
    _validate_frame_transformations(plan, frame_kinds)
    _validate_clock_topology(plan, clock_kinds)


def _validate_signal_inventories(
    plan: Mapping[str, object], candidate_classes: Mapping[str, str]
) -> None:
    """Validate plan signal inventories against candidate classes."""
    for channel in _objects(plan, "channels"):
        channel_id = _identifier(channel, "identifier")
        candidate_id = _identifier(channel, "candidate_id")
        try:
            observability_class = candidate_classes[candidate_id]
        except KeyError:
            _fail(
                DiagnosticPlanDepthRefusalKind.CARRIER,
                f"channel {channel_id!r} names an unknown candidate",
            )
        signals = _objects(channel, "signals", exact_keys=_SIGNAL_KEYS)
        if not signals:
            _fail(
                DiagnosticPlanDepthRefusalKind.CARRIER,
                f"channel {channel_id!r} must declare at least one signal",
            )
        signal_ids: list[str] = []
        for signal in signals:
            signal_ids.append(_identifier(signal, "identifier"))
            _nonempty_text(signal, "quantity")
            unit = _nonempty_text(signal, "unit")
            if any(character.isspace() for character in unit):
                _fail(
                    DiagnosticPlanDepthRefusalKind.PLAN,
                    f"channel {channel_id!r} signal unit must not contain whitespace",
                )
            role = _nonempty_text(signal, "role")
            if role not in _SIGNAL_ROLES:
                _fail(
                    DiagnosticPlanDepthRefusalKind.PLAN,
                    f"channel {channel_id!r} has unsupported signal role {role!r}",
                )
            _nonempty_text(signal, "description")
        _sorted_unique_identifiers(signal_ids, f"channel {channel_id!r} signals")
        carriers = [signal for signal in signals if signal["role"] == "carrier"]
        if len(carriers) != 1:
            _fail(
                DiagnosticPlanDepthRefusalKind.CARRIER,
                f"channel {channel_id!r} must declare exactly one carrier signal",
            )
        markers = [signal for signal in signals if signal["role"] == "timing_marker"]
        if observability_class == "event_relative":
            if len(markers) != 1 or markers[0]["unit"] != "s":
                _fail(
                    DiagnosticPlanDepthRefusalKind.CLOCK,
                    f"event-relative channel {channel_id!r} requires one "
                    "seconds marker",
                )
        elif markers:
            _fail(
                DiagnosticPlanDepthRefusalKind.CLOCK,
                f"non-event channel {channel_id!r} cannot declare a timing marker",
            )
        if observability_class == "numerical_only" and (
            len(signals) != 1
            or carriers[0]["quantity"] != "phase"
            or carriers[0]["unit"] != "rad"
        ):
            _fail(
                DiagnosticPlanDepthRefusalKind.CARRIER,
                f"numerical-only channel {channel_id!r} requires one phase/rad carrier",
            )


def _validate_frame_transformations(
    plan: Mapping[str, object], frame_kinds: Mapping[str, str]
) -> None:
    """Validate every declared frame transformation."""
    transformations = _objects(
        plan, "frame_transformations", exact_keys=_TRANSFORMATION_KEYS
    )
    ordered_keys: list[tuple[str, str]] = []
    seen_pairs: set[frozenset[str]] = set()
    adjacency: dict[str, set[str]] = {identifier: set() for identifier in frame_kinds}
    for transformation in transformations:
        source = _identifier(transformation, "source_identifier")
        target = _identifier(transformation, "target_identifier")
        if source == target:
            _fail(
                DiagnosticPlanDepthRefusalKind.PLAN,
                "a frame transformation cannot map a frame to itself",
            )
        ordered_keys.append((source, target))
        pair = frozenset({source, target})
        if pair in seen_pairs:
            _fail(
                DiagnosticPlanDepthRefusalKind.PLAN,
                f"duplicate transformation pair {sorted(pair)!r}",
            )
        seen_pairs.add(pair)
        try:
            kinds = frozenset({frame_kinds[source], frame_kinds[target]})
        except KeyError:
            _fail(
                DiagnosticPlanDepthRefusalKind.PLAN,
                f"transformation pair {sorted(pair)!r} names an undeclared frame",
            )
        kind = _nonempty_text(transformation, "kind")
        if kind not in _TRANSFORMATION_KINDS:
            _fail(
                DiagnosticPlanDepthRefusalKind.PLAN,
                f"unsupported transformation kind {kind!r}",
            )
        if _ADMISSIBLE_TRANSFORMATIONS.get(kinds) != kind:
            _fail(
                DiagnosticPlanDepthRefusalKind.PLAN,
                f"transformation kind {kind!r} is inadmissible for {sorted(kinds)!r}",
            )
        dependent = _boolean(transformation, "equilibrium_dependent")
        if dependent is not (kind == "flux_mapping"):
            _fail(
                DiagnosticPlanDepthRefusalKind.PLAN,
                "equilibrium dependency does not match transformation kind",
            )
        _nonempty_text(transformation, "method")
        if _boolean(transformation, "evidence_claimed") is not False:
            _fail(
                DiagnosticPlanDepthRefusalKind.AUTHORITY,
                "frame transformation cannot claim mapping evidence",
            )
        adjacency[source].add(target)
        adjacency[target].add(source)
    if tuple(sorted(ordered_keys)) != tuple(ordered_keys):
        _fail(
            DiagnosticPlanDepthRefusalKind.PLAN,
            "frame transformations must be sorted by source and target",
        )
    if len(frame_kinds) >= 2:
        start = next(iter(frame_kinds))
        reached = {start}
        frontier = [start]
        while frontier:
            for neighbour in adjacency[frontier.pop()]:
                if neighbour not in reached:
                    reached.add(neighbour)
                    frontier.append(neighbour)
        if reached != set(frame_kinds):
            _fail(
                DiagnosticPlanDepthRefusalKind.PLAN,
                "declared frames are disconnected; "
                f"unreachable={sorted(set(frame_kinds) - reached)!r}",
            )


def _validate_clock_topology(
    plan: Mapping[str, object], clock_kinds: Mapping[str, str]
) -> None:
    """Validate clock identities, relations, and timing topology."""
    topology = _mapping(plan, "clock_topology", exact_keys=_TOPOLOGY_KEYS)
    domains = _objects(topology, "domains", exact_keys=_DOMAIN_KEYS)
    if not domains:
        _fail(DiagnosticPlanDepthRefusalKind.CLOCK, "clock topology has no domains")
    domain_ids: list[str] = []
    domain_members: dict[str, tuple[str, ...]] = {}
    domain_roots: dict[str, str] = {}
    membership: dict[str, str] = {}
    for domain in domains:
        domain_id = _identifier(domain, "identifier")
        domain_ids.append(domain_id)
        root = _identifier(domain, "root_clock_identifier")
        members = _string_tuple(domain, "member_clock_identifiers")
        if not members:
            _fail(
                DiagnosticPlanDepthRefusalKind.CLOCK,
                f"clock domain {domain_id!r} has no members",
            )
        _sorted_unique_identifiers(members, f"clock domain {domain_id!r} members")
        if root not in members:
            _fail(
                DiagnosticPlanDepthRefusalKind.CLOCK,
                f"clock domain {domain_id!r} root is not a member",
            )
        _nonempty_text(domain, "scope")
        for member in members:
            try:
                kind = clock_kinds[member]
            except KeyError:
                _fail(
                    DiagnosticPlanDepthRefusalKind.CLOCK,
                    f"clock topology names undeclared clock {member!r}",
                )
            if kind == "simulation":
                _fail(
                    DiagnosticPlanDepthRefusalKind.CLOCK,
                    "simulation clock cannot belong to a physical clock domain",
                )
            if member in membership:
                _fail(
                    DiagnosticPlanDepthRefusalKind.CLOCK,
                    f"clock {member!r} belongs to more than one domain",
                )
            membership[member] = domain_id
        expected_root_kind = (
            "facility_monotonic"
            if any(clock_kinds[member] == "facility_monotonic" for member in members)
            else "shot_event_epoch"
        )
        if clock_kinds[root] != expected_root_kind:
            _fail(
                DiagnosticPlanDepthRefusalKind.CLOCK,
                f"clock domain {domain_id!r} has invalid root kind",
            )
        domain_members[domain_id] = members
        domain_roots[domain_id] = root
    _sorted_unique_identifiers(domain_ids, "clock topology domains")
    reference_domain = _identifier(topology, "reference_domain_identifier")
    if reference_domain not in domain_roots:
        _fail(
            DiagnosticPlanDepthRefusalKind.CLOCK,
            "clock topology reference domain is not declared",
        )
    unassigned = sorted(
        identifier
        for identifier, kind in clock_kinds.items()
        if kind != "simulation" and identifier not in membership
    )
    if unassigned:
        _fail(
            DiagnosticPlanDepthRefusalKind.CLOCK,
            f"non-simulation clocks belong to no domain: {unassigned!r}",
        )
    parents: dict[str, set[str]] = {}
    for relation in _objects(plan, "clock_relations"):
        child = _identifier(relation, "child_identifier")
        parent = _identifier(relation, "parent_identifier")
        parents.setdefault(child, set()).add(parent)
    reference_root = domain_roots[reference_domain]
    for domain_id, members in domain_members.items():
        root = domain_roots[domain_id]
        for member in members:
            if member != root and root not in parents.get(member, set()):
                _fail(
                    DiagnosticPlanDepthRefusalKind.CLOCK,
                    f"clock {member!r} lacks a relation to domain root {root!r}",
                )
        if domain_id != reference_domain and reference_root not in parents.get(
            root, set()
        ):
            _fail(
                DiagnosticPlanDepthRefusalKind.CLOCK,
                f"domain root {root!r} lacks a relation to reference root "
                f"{reference_root!r}",
            )
    _refuse_relation_cycles(parents, clock_kinds)


def _refuse_relation_cycles(
    parents: Mapping[str, set[str]], clock_kinds: Mapping[str, str]
) -> None:
    """Reject cycles in the declared clock relations."""
    visiting: set[str] = set()
    finished: set[str] = set()

    def visit(identifier: str) -> None:
        """Visit one clock relation while detecting dependency cycles."""
        if identifier in finished:
            return
        if identifier in visiting:
            _fail(
                DiagnosticPlanDepthRefusalKind.CLOCK,
                "clock relations must not form a cycle",
            )
        visiting.add(identifier)
        for parent in sorted(parents.get(identifier, set())):
            visit(parent)
        visiting.remove(identifier)
        finished.add(identifier)

    for identifier in sorted(clock_kinds):
        visit(identifier)


def _objects(
    parent: Mapping[str, object],
    name: str,
    *,
    exact_keys: frozenset[str] | None = None,
) -> list[Mapping[str, object]]:
    """Require an array of object records from the parent mapping."""
    value = parent.get(name)
    if not isinstance(value, list):
        _fail(DiagnosticPlanDepthRefusalKind.PLAN, f"{name} must be an array")
    records: list[Mapping[str, object]] = []
    for item in value:
        if not _is_string_mapping(item):
            _fail(DiagnosticPlanDepthRefusalKind.PLAN, f"{name}[] must be an object")
        if exact_keys is not None and set(item) != exact_keys:
            _fail(
                DiagnosticPlanDepthRefusalKind.PLAN,
                f"{name}[] key mismatch",
            )
        records.append(item)
    return records


def _mapping(
    parent: Mapping[str, object], name: str, *, exact_keys: frozenset[str]
) -> Mapping[str, object]:
    """Require a mapping field from the parent mapping."""
    value = parent.get(name)
    if not _is_string_mapping(value) or set(value) != exact_keys:
        _fail(
            DiagnosticPlanDepthRefusalKind.PLAN,
            f"{name} must be an exact object",
        )
    return value


def _identifier(parent: Mapping[str, object], name: str) -> str:
    """Require a canonical identifier field."""
    value = parent.get(name)
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        _fail(
            DiagnosticPlanDepthRefusalKind.PLAN,
            f"{name} must be a valid identifier",
        )
    return value


def _nonempty_text(parent: Mapping[str, object], name: str) -> str:
    """Require a non-empty text field."""
    value = parent.get(name)
    if not isinstance(value, str) or not value:
        _fail(DiagnosticPlanDepthRefusalKind.PLAN, f"{name} must be non-empty text")
    return value


def _boolean(parent: Mapping[str, object], name: str) -> bool:
    """Require a strict boolean field."""
    value = parent.get(name)
    if not isinstance(value, bool):
        _fail(DiagnosticPlanDepthRefusalKind.PLAN, f"{name} must be a boolean")
    return value


def _string_tuple(parent: Mapping[str, object], name: str) -> tuple[str, ...]:
    """Require a tuple of text values."""
    value = parent.get(name)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _fail(
            DiagnosticPlanDepthRefusalKind.PLAN,
            f"{name} must be an array of strings",
        )
    return tuple(value)


def _sorted_unique_identifiers(values: list[str] | tuple[str, ...], name: str) -> None:
    """Require sorted unique canonical identifiers."""
    if tuple(sorted(set(values))) != tuple(values) or any(
        _IDENTIFIER.fullmatch(value) is None for value in values
    ):
        _fail(
            DiagnosticPlanDepthRefusalKind.PLAN,
            f"{name} must contain unique sorted identifiers",
        )


def _is_string_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    """Report whether a value is a string-keyed mapping."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def _fail(kind: DiagnosticPlanDepthRefusalKind, detail: str) -> NoReturn:
    """Raise a typed diagnostic-plan-depth refusal."""
    raise DiagnosticPlanDepthError(kind, detail)
