# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for offline evolutionary Petri-net mutation grammar

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar

import pytest

from scpn_phase_orchestrator.supervisor.evolutionary_petri_grammar import (
    run_offline_evolutionary_petri_mutation_grammar,
)

_R_CALL = TypeVar("_R_CALL")


def _call(fn: Callable[..., _R_CALL], *args: object, **kwargs: object) -> _R_CALL:
    """Call ``fn`` with arguments mypy cannot check: invalid or dynamic kwargs."""
    return fn(*args, **kwargs)


def _net_like_mapping() -> dict[str, object]:
    return {
        "places": [
            {"name": "idle", "token_bound": 2},
            {"name": "nominal", "token_bound": 4},
        ],
        "transitions": [
            {"name": "to_nominal", "guard_weights": {"R": 0.8}},
            {"name": "to_degraded", "guard_weights": {"R": 0.2}},
        ],
        "arcs": [
            {
                "place": "idle",
                "transition": "to_nominal",
                "direction": "input",
                "weight": 1,
            }
        ],
    }


def _net_like_records() -> list[dict[str, object]]:
    return [
        {"kind": "place", "name": "idle", "token_bound": 2},
        {"kind": "place", "name": "nominal", "token_bound": 3},
        {"kind": "transition", "name": "to_nominal", "guard_weights": {"R": 0.5}},
        {
            "kind": "arc",
            "place": "idle",
            "transition": "to_nominal",
            "direction": "input",
            "weight": 1,
        },
    ]


def test_plan_generation_is_deterministic() -> None:
    first = run_offline_evolutionary_petri_mutation_grammar(
        _net_like_mapping(),
        generation_count=2,
        candidates_per_generation=3,
        mutation_step=0.12,
        max_arc_weight=3,
        max_token_bound=32,
    )
    second = run_offline_evolutionary_petri_mutation_grammar(
        _net_like_mapping(),
        generation_count=2,
        candidates_per_generation=3,
        mutation_step=0.12,
        max_arc_weight=3,
        max_token_bound=32,
    )

    assert first == second
    assert first.plan_hash == second.plan_hash
    assert first.candidate_count == 6
    assert first.execution_disabled is True
    assert first.operator_review_required is True
    assert first.live_merge_permitted is False
    assert first.hot_patch_permitted is False
    assert first.actuation_permitted is False
    assert all(
        candidate.candidate_hash and len(candidate.candidate_hash) == 64
        for candidate in first.candidates
    )
    assert {candidate.mutation_type for candidate in first.candidates} == {
        "add_arc",
        "guard_weight",
        "token_bound",
    }


def test_simple_record_input_is_accepted() -> None:
    first = run_offline_evolutionary_petri_mutation_grammar(
        _net_like_records(),
        generation_count=1,
        candidates_per_generation=2,
        mutation_step=0.1,
    )
    second = run_offline_evolutionary_petri_mutation_grammar(
        _net_like_records(),
        generation_count=1,
        candidates_per_generation=2,
        mutation_step=0.1,
    )

    assert first == second
    assert first.candidate_count == 2
    assert first.source_net["arcs"]


def test_malformed_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="place name"):
        run_offline_evolutionary_petri_mutation_grammar(
            {"places": [{"name": ""}], "transitions": ["to_nominal"]},
        )

    with pytest.raises(ValueError, match="token_bound"):
        run_offline_evolutionary_petri_mutation_grammar(
            {
                "places": [{"name": "idle", "token_bound": -1}],
                "transitions": ["to_nominal"],
            },
        )

    with pytest.raises(ValueError, match="kind"):
        run_offline_evolutionary_petri_mutation_grammar(
            {"name": "idle"},
        )

    with pytest.raises(ValueError, match="arc.direction"):
        run_offline_evolutionary_petri_mutation_grammar(
            {
                "places": ["idle"],
                "transitions": ["to_nominal"],
                "arcs": [
                    {
                        "place": "idle",
                        "transition": "to_nominal",
                        "direction": "sideways",
                    }
                ],
            },
        )


def test_audit_record_is_json_safe_and_non_actuating() -> None:
    plan = run_offline_evolutionary_petri_mutation_grammar(
        _net_like_mapping(),
        generation_count=1,
        candidates_per_generation=4,
        mutation_step=0.15,
    )
    record = plan.to_audit_record()

    dumped = json.dumps(record, sort_keys=True, allow_nan=False)
    loaded = json.loads(dumped)

    assert loaded["schema_name"] == "evolutionary_petri_mutation_grammar"
    assert loaded["non_actuating"] is True
    assert loaded["execution_disabled"] is True
    assert loaded["operator_review_required"] is True
    assert loaded["live_merge_permitted"] is False
    assert loaded["hot_patch_permitted"] is False
    assert loaded["actuation_permitted"] is False
    assert all(
        isinstance(candidate, dict) and candidate["candidate_id"].startswith("g")
        for candidate in loaded["candidates"]
    )
    assert all(
        len(candidate["candidate_hash"]) == 64 for candidate in loaded["candidates"]
    )
    assert loaded["plan_hash"] == plan.plan_hash
    assert isinstance(loaded["source_net"]["places"], list)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"generation_count": 0}, "generation_count must be a positive integer"),
        (
            {"candidates_per_generation": 0},
            "candidates_per_generation must be a positive integer",
        ),
        ({"mutation_step": "x"}, "mutation_step must be finite"),
        ({"mutation_step": float("inf")}, "mutation_step must be finite"),
        ({"mutation_step": 0.0}, "mutation_step must be positive"),
        ({"max_arc_weight": 0}, "max_arc_weight must be a positive integer"),
        ({"max_token_bound": 0}, "max_token_bound must be a positive integer"),
    ],
)
def test_grammar_rejects_invalid_config(config: dict[str, object], match: str) -> None:
    # Some configs deliberately carry values outside the declared parameter
    # types, so the guard is reached through an unparameterised reference.
    build: Callable[..., object] = run_offline_evolutionary_petri_mutation_grammar
    with pytest.raises(ValueError, match=match):
        build(_net_like_mapping(), **config)


@pytest.mark.parametrize(
    ("net_like", "match"),
    [
        (5, "net_like must be a mapping or a sequence"),
        ([], "sequence must not be empty"),
        ([5], "each net record must be a mapping"),
        ([{"kind": "weird"}], "unknown record kind"),
        ({"places": "abc", "transitions": ["t"]}, "places must be a sequence"),
        ({"places": 5, "transitions": ["t"]}, "places must be a sequence"),
        ({"places": ["p"], "transitions": 5}, "transitions must be a sequence"),
        (
            {"places": ["p"], "transitions": ["t"], "arcs": 5},
            "arcs must be a sequence",
        ),
        ({"places": [], "transitions": ["t"]}, "places must be non-empty"),
        (
            {"places": [5], "transitions": ["t"]},
            "place record must be a string name or a mapping",
        ),
        ({"places": ["p"], "transitions": "abc"}, "transitions must be a sequence"),
        ({"places": ["p"], "transitions": []}, "transitions must be non-empty"),
        (
            {"places": ["p"], "transitions": [5]},
            "transition record must be a string name or a mapping",
        ),
        (
            {"places": ["p"], "transitions": [{"name": "t", "guard_weights": 5}]},
            "guard_weights must be a mapping or a sequence",
        ),
        (
            {"places": ["p"], "transitions": [{"name": "t", "guard_weights": [5]}]},
            "guard entry must be a mapping",
        ),
        (
            {"places": ["p"], "transitions": ["t"], "arcs": "abc"},
            "arcs must be a sequence",
        ),
        (
            {"places": ["p"], "transitions": ["t"], "arcs": [5]},
            "arc record must be a mapping or sequence",
        ),
        (
            {"places": ["p"], "transitions": ["t"], "arcs": [["p", "t"]]},
            "arc sequence record needs at least",
        ),
        (
            {
                "places": ["p"],
                "transitions": ["t"],
                "arcs": [{"place": "ghost", "transition": "t", "direction": "input"}],
            },
            "arc references unknown place",
        ),
        (
            {
                "places": ["p"],
                "transitions": ["t"],
                "arcs": [{"place": "p", "transition": "ghost", "direction": "input"}],
            },
            "arc references unknown transition",
        ),
    ],
)
def test_grammar_rejects_malformed_net(net_like: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _call(run_offline_evolutionary_petri_mutation_grammar, net_like)


def test_grammar_accepts_sequence_guards_and_arcs() -> None:
    net = {
        "places": [
            {"name": "p1", "token_bound": 2},
            {"name": "p2", "token_bound": 2},
        ],
        "transitions": [
            {"name": "t1", "guard_weights": [{"metric": "R", "weight": 0.5}]},
        ],
        "arcs": [
            ["p1", "t1", "input", 1],
            ["p2", "t1", "output", 1],
        ],
    }
    plan = run_offline_evolutionary_petri_mutation_grammar(
        net,
        generation_count=1,
        candidates_per_generation=2,
    )
    assert plan.candidate_count == 2
    assert plan.execution_disabled is True


def test_grammar_accepts_single_simple_records() -> None:
    place_plan = run_offline_evolutionary_petri_mutation_grammar(
        {"kind": "place", "name": "idle", "token_bound": 2},
        generation_count=1,
        candidates_per_generation=1,
    )
    assert place_plan.source_net["places"]
    transition_plan = run_offline_evolutionary_petri_mutation_grammar(
        {"kind": "transition", "name": "to_nominal", "guard_weights": {"R": 0.5}},
        generation_count=1,
        candidates_per_generation=1,
    )
    assert transition_plan.source_net["transitions"]


def test_arc_only_record_blocks_every_mutation_type() -> None:
    """A lone arc record has no place or transition to mutate, so all are blocked."""
    plan = run_offline_evolutionary_petri_mutation_grammar(
        {
            "kind": "arc",
            "place": "p",
            "transition": "t",
            "direction": "input",
            "weight": 1,
        },
        generation_count=1,
        candidates_per_generation=3,
    )
    reasons = {c.mutation_type: c.blocked_reasons for c in plan.candidates}
    assert reasons == {
        "add_arc": ("net_missing_places_or_transitions",),
        "guard_weight": ("net_missing_transitions",),
        "token_bound": ("net_missing_places",),
    }
    assert plan.accepted_count == 0
    assert plan.best_candidate_id is None
    assert plan.source_net["arcs"]


def test_transition_without_guards_mutates_the_default_safety_metric() -> None:
    """A transition with no guard weights is mutated on ``safety_margin`` from 0."""
    plan = run_offline_evolutionary_petri_mutation_grammar(
        {
            "places": [{"name": "p", "token_bound": 2}],
            "transitions": [{"name": "t"}],
            "arcs": [],
        },
        generation_count=1,
        candidates_per_generation=3,
    )
    guard = next(c for c in plan.candidates if c.mutation_type == "guard_weight")
    assert guard.accepted
    assert guard.before["metric"] == "safety_margin"
    assert guard.before["weight"] == 0.0
    assert guard.before["all_guards"] == []
    assert guard.after["metric"] == "safety_margin"
    assert guard.after["weight"] != 0.0


def test_fully_connected_net_has_no_free_arc_target() -> None:
    """When every place-transition-direction arc exists, add_arc is blocked."""
    plan = run_offline_evolutionary_petri_mutation_grammar(
        {
            "places": [{"name": "p", "token_bound": 2}],
            "transitions": [{"name": "t", "guard_weights": {"R": 0.5}}],
            "arcs": [
                {"place": "p", "transition": "t", "direction": "input", "weight": 1},
                {"place": "p", "transition": "t", "direction": "output", "weight": 1},
            ],
        },
        generation_count=2,
        candidates_per_generation=3,
    )
    add_arcs = [c for c in plan.candidates if c.mutation_type == "add_arc"]
    assert len(add_arcs) == 2
    assert all(c.blocked_reasons == ("add_arc_no_available_target",) for c in add_arcs)


def test_preferred_output_arc_search_proposes_only_a_free_arc() -> None:
    """A taken preferred output arc falls back to the one free arc, never a copy."""
    existing = [
        {"place": "p1", "transition": "t1", "direction": "input", "weight": 1},
        {"place": "p1", "transition": "t1", "direction": "output", "weight": 1},
        {"place": "p2", "transition": "t1", "direction": "output", "weight": 1},
    ]
    plan = run_offline_evolutionary_petri_mutation_grammar(
        {
            "places": [
                {"name": "p1", "token_bound": 2},
                {"name": "p2", "token_bound": 2},
            ],
            "transitions": [{"name": "t1", "guard_weights": {"R": 0.5}}],
            "arcs": existing,
        },
        generation_count=1,
        candidates_per_generation=4,
    )
    add_arcs = [c for c in plan.candidates if c.mutation_type == "add_arc"]
    assert len(add_arcs) == 2
    for candidate in add_arcs:
        assert candidate.accepted
        arcs = candidate.after["arcs"]
        assert isinstance(arcs, list)
        keys = [(a["place"], a["transition"], a["direction"]) for a in arcs]
        assert len(keys) == len(set(keys)) == len(existing) + 1
        assert ("p2", "t1", "input") in keys


def test_token_bound_above_policy_max_is_blocked() -> None:
    """A token-bound mutation past ``max_token_bound`` is proposed but blocked."""
    plan = run_offline_evolutionary_petri_mutation_grammar(
        {
            "places": [{"name": "p", "token_bound": 128}],
            "transitions": [{"name": "t", "guard_weights": {"R": 0.5}}],
            "arcs": [],
        },
        generation_count=2,
        candidates_per_generation=6,
        max_token_bound=128,
    )
    blocked = [
        c
        for c in plan.candidates
        if c.blocked_reasons == ("token_bound_exceeds_policy_max",)
    ]
    assert blocked
    assert all(c.mutation_type == "token_bound" for c in blocked)
    assert all(not c.accepted for c in blocked)
