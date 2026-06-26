# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio evolutionary panel contract tests

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

import pytest

from scpn_phase_orchestrator.studio import (
    build_evolutionary_supervisor_policy_search_studio_panel,
)

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64
HEX_F = "f" * 64


def _candidate(candidate_id: str, *, accepted: bool) -> dict[str, object]:
    status = "accepted_for_review" if accepted else "rejected"
    blocked_reasons: tuple[str, ...] = () if accepted else ("stl_margin_breach",)
    return {
        "candidate_id": candidate_id,
        "generation": 1,
        "knob": "K",
        "parent_value": 0.42,
        "candidate_value": 0.45 if accepted else 0.62,
        "mutation_delta": 0.03 if accepted else 0.20,
        "genome": (("K", 0.45), ("alpha", 0.18)),
        "replay_fitness": 0.91 if accepted else 0.47,
        "stl_robustness": 0.08 if accepted else -0.03,
        "stl_satisfied": accepted,
        "replay_violation_count": 0 if accepted else 2,
        "blocked_reasons": blocked_reasons,
        "status": status,
        "review_required": True,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "actuation_permitted": False,
        "candidate_hash": HEX_B if accepted else HEX_C,
    }


def _search_report() -> dict[str, object]:
    return {
        "schema_name": "evolutionary_supervisor_policy_search",
        "schema_version": "1.0",
        "generation_count": 2,
        "population_size": 4,
        "mutation_step": 0.04,
        "minimum_replay_reward": 0.70,
        "minimum_safety_margin": 0.04,
        "parent_policy_hash": HEX_A,
        "replay_summary": {
            "replay_count": 3,
            "mean_reward": 0.88,
            "min_reward": 0.81,
            "mean_safety_margin": 0.09,
            "min_safety_margin": 0.05,
            "violation_count": 0,
        },
        "stl_spec": "always (R >= 0.82)",
        "stl_monitoring": {
            "satisfied": True,
            "minimum_margin": 0.05,
            "status": "pass",
            "margin_trace": (0.05, 0.07, "reviewed", True),
        },
        "candidate_count": 2,
        "accepted_count": 1,
        "rejected_count": 1,
        "best_candidate": None,
        "candidates": (
            _candidate("accepted_grid_guard", accepted=True),
            _candidate("rejected_grid_guard", accepted=False),
        ),
        "claim_boundary": "offline_evolutionary_supervisor_review_not_live_actuation",
        "non_actuating": True,
        "execution_disabled": True,
        "hot_patch_permitted": False,
        "live_merge_permitted": False,
        "operator_review_required": True,
        "report_hash": HEX_D,
    }


def _dsl_candidate(candidate_id: str, *, status: str = "accepted") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "generation": 1,
        "mutation_index": 0,
        "source_rule_name": "grid_guard",
        "blocked_reasons": () if status == "accepted" else ("parse_rejected",),
        "status": status,
        "candidate_hash": HEX_E,
        "operator_review_required": True,
        "execution_disabled": True,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "actuation_permitted": False,
    }


def _dsl_report() -> dict[str, object]:
    return {
        "schema_name": "policy_dsl_evolution",
        "schema_version": "1.0",
        "generation_count": 1,
        "population_size": 2,
        "mutation_step": 0.01,
        "source_policy_hash": HEX_F,
        "candidate_count": 1,
        "accepted_count": 1,
        "rejected_count": 0,
        "candidates": (_dsl_candidate("dsl_grid_guard"),),
        "execution_disabled": True,
        "hot_patch_permitted": False,
        "live_merge_permitted": False,
        "actuation_permitted": False,
        "operator_review_required": True,
        "non_actuating": True,
        "report_hash": HEX_A,
    }


def _example(*, enriched: bool = True) -> dict[str, object]:
    record: dict[str, object] = {
        "domain": "power_grid",
        "scenario_id": "wide_area_phase_review",
        "scenario_hash": HEX_B,
        "claim_boundary": "evolutionary_supervisor_search_not_live_actuation",
        "operator_review_required": True,
        "execution_disabled": True,
        "hot_patch_permitted": False,
        "live_merge_permitted": False,
        "actuation_permitted": False,
    }
    if enriched:
        record.update(
            {
                "candidate_count": 2,
                "accepted_candidate_count": 1,
                "rejected_candidate_count": 1,
                "report_hash": HEX_C,
            }
        )
    return record


def _with_path(
    record: Mapping[str, object],
    path: Sequence[str | int],
    value: object,
) -> dict[str, object]:
    updated = deepcopy(dict(record))
    cursor: Any = updated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return updated


def test_evolutionary_panel_accepts_review_only_search_dsl_and_examples() -> None:
    panel = build_evolutionary_supervisor_policy_search_studio_panel(
        [_search_report()],
        examples=[_example(enriched=False), _example()],
        dsl_reports=[_dsl_report()],
    )

    assert panel["panel_kind"] == "studio_evolutionary_supervisor_policy_search_panel"
    assert panel["search_report_count"] == 1
    assert panel["dsl_report_count"] == 1
    assert panel["example_count"] == 2
    assert panel["best_candidate_rows"] == ()
    assert panel["candidate_count_range"] == {"minimum": 1, "maximum": 2}
    assert panel["accepted_candidate_total"] == 2
    assert panel["rejected_candidate_total"] == 1
    assert panel["replay_reward_range"] == {"minimum": 0.88, "maximum": 0.88}
    assert panel["search_reports"][0]["stl_monitoring"]["margin_trace"] == [
        0.05,
        0.07,
        "reviewed",
        True,
    ]
    assert panel["example_rows"][0]["candidate_count"] is None
    assert "actions_to_apply" not in panel
    assert "control_actions" not in panel


@pytest.mark.parametrize(
    ("reports", "message"),
    [
        (
            {"schema_name": "evolutionary_supervisor_policy_search"},
            "non-empty sequence",
        ),
        ((), "non-empty sequence"),
        ((object(),), "report must be a mapping"),
        (
            (_with_path(_search_report(), ("schema_name",), "unsupported"),),
            "schema_name",
        ),
        (
            (_with_path(_search_report(), ("claim_boundary",), "live_policy_merge"),),
            "claim_boundary",
        ),
        (
            (_with_path(_search_report(), ("candidate_count",), 3),),
            "sum to candidate_count",
        ),
        (
            (
                {
                    **_search_report(),
                    "accepted_count": 0,
                    "rejected_count": 2,
                },
            ),
            "accepted_count",
        ),
        (
            (
                {
                    **_search_report(),
                    "accepted_count": 2,
                    "rejected_count": 0,
                },
            ),
            "rejected_count",
        ),
        (
            (
                {
                    **_search_report(),
                    "candidate_count": 1,
                    "accepted_count": 1,
                    "rejected_count": 0,
                },
            ),
            "candidates length",
        ),
        (
            (_with_path(_search_report(), ("candidates",), "candidate"),),
            "must be a sequence",
        ),
        ((_with_path(_search_report(), ("candidates",), ()),), "must not be empty"),
        (
            (_with_path(_search_report(), ("candidates",), (object(),)),),
            "entries must be mappings",
        ),
        (
            (
                _with_path(
                    _search_report(),
                    ("candidates", 1, "candidate_id"),
                    "accepted_grid_guard",
                ),
            ),
            "unique",
        ),
        (
            (_with_path(_search_report(), ("candidates", 0, "status"), "queued"),),
            "status",
        ),
        (
            (
                _with_path(
                    _search_report(), ("candidates", 0, "review_required"), False
                ),
            ),
            "review_required",
        ),
        (
            (
                _with_path(
                    _search_report(),
                    ("candidates", 0, "blocked_reasons"),
                    ("should_not_block",),
                ),
            ),
            "must not have blocked_reasons",
        ),
        (
            (_with_path(_search_report(), ("candidates", 1, "blocked_reasons"), ()),),
            "require blocked_reasons",
        ),
        (
            (_with_path(_search_report(), ("replay_summary",), object()),),
            "replay_summary",
        ),
        (
            (_with_path(_search_report(), ("stl_monitoring",), object()),),
            "stl_monitoring",
        ),
        (
            (
                _with_path(
                    _search_report(), ("stl_monitoring", "margin_trace"), (object(),)
                ),
            ),
            "sequence values",
        ),
        (
            (_with_path(_search_report(), ("stl_monitoring", "raw"), object()),),
            "JSON-safe",
        ),
        (
            (_with_path(_search_report(), ("best_candidate",), object()),),
            "best_candidate",
        ),
        ((_with_path(_search_report(), ("non_actuating",), False),), "non_actuating"),
    ],
)
def test_evolutionary_panel_rejects_malformed_search_reports(
    reports: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_evolutionary_supervisor_policy_search_studio_panel(
            cast("Sequence[Mapping[str, object]]", reports)
        )


@pytest.mark.parametrize(
    ("examples", "message"),
    [
        ({"domain": "power_grid"}, "must be a sequence"),
        ((object(),), "example must be a mapping"),
        (
            (_with_path(_example(), ("claim_boundary",), "live_merge"),),
            "claim_boundary",
        ),
        (
            (_with_path(_example(), ("execution_disabled",), False),),
            "execution_disabled",
        ),
        (
            (_with_path(_example(), ("accepted_candidate_count",), None),),
            "require report_hash",
        ),
        ((_with_path(_example(), ("report_hash",), None),), "require report_hash"),
        (
            (_with_path(_example(), ("accepted_candidate_count",), 2),),
            "accepted/rejected counts",
        ),
    ],
)
def test_evolutionary_panel_rejects_malformed_examples(
    examples: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_evolutionary_supervisor_policy_search_studio_panel(
            [_search_report()],
            examples=cast("Sequence[Mapping[str, object]]", examples),
        )


@pytest.mark.parametrize(
    ("dsl_reports", "message"),
    [
        ({"schema_name": "policy_dsl_evolution"}, "must be a sequence"),
        ((object(),), "DSL report must be a mapping"),
        ((_with_path(_dsl_report(), ("schema_name",), "unsupported"),), "schema_name"),
        (
            (_with_path(_dsl_report(), ("execution_disabled",), False),),
            "execution_disabled",
        ),
        (
            (_with_path(_dsl_report(), ("accepted_count",), 2),),
            "accepted/rejected counts",
        ),
        (
            (
                {
                    **_dsl_report(),
                    "candidate_count": 2,
                    "rejected_count": 1,
                },
            ),
            "candidate_count",
        ),
        (
            (_with_path(_dsl_report(), ("candidates",), "candidate"),),
            "must be a sequence",
        ),
        (
            (_with_path(_dsl_report(), ("candidates",), (object(),)),),
            "entries must be mappings",
        ),
        (
            (
                {
                    **_dsl_report(),
                    "candidate_count": 2,
                    "accepted_count": 2,
                    "candidates": (
                        _dsl_candidate("dsl_grid_guard"),
                        _dsl_candidate("dsl_grid_guard"),
                    ),
                },
            ),
            "unique",
        ),
        (
            (
                _with_path(
                    _dsl_report(), ("candidates", 0, "operator_review_required"), False
                ),
            ),
            "operator_review_required",
        ),
        ((_with_path(_dsl_report(), ("candidates", 0, "status"), "queued"),), "status"),
        ((_with_path(_dsl_report(), ("candidates",), ()),), "must not be empty"),
    ],
)
def test_evolutionary_panel_rejects_malformed_dsl_reports(
    dsl_reports: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_evolutionary_supervisor_policy_search_studio_panel(
            [_search_report()],
            dsl_reports=cast("Sequence[Mapping[str, object]]", dsl_reports),
        )
