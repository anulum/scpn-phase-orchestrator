# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for offline policy DSL mutation search

from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.supervisor.evolutionary_policy_dsl import (
    PolicyMutationCandidate,
    parse_policy_dsl,
    run_offline_evolutionary_policy_dsl_search,
)


def _sample_dsl() -> str:
    return (
        "rule throttle_guard: if R < 0.95 and K > 0.10 then set K += 0.04\n"
        "rule recovery_guard: if R >= 0.20 then set K -= 0.03\n"
        "rule safety_guard: if R < 0.40 then set K = 0.12"
    )


def test_mutation_search_is_deterministic_for_same_inputs() -> None:
    first = run_offline_evolutionary_policy_dsl_search(
        _sample_dsl(),
        generation_count=2,
        population_size=4,
        mutation_step=0.02,
    )
    second = run_offline_evolutionary_policy_dsl_search(
        _sample_dsl(),
        generation_count=2,
        population_size=4,
        mutation_step=0.02,
    )

    assert first == second
    assert first.report_hash == second.report_hash
    assert first.candidate_count == 8
    assert first.accepted_count >= 1
    assert first.rejected_count >= 0


def test_parse_and_mutate_cover_rules_conditions_and_actions() -> None:
    rules = parse_policy_dsl(_sample_dsl())
    assert len(rules) == 3

    report = run_offline_evolutionary_policy_dsl_search(
        _sample_dsl(),
        generation_count=1,
        population_size=4,
        mutation_step=0.01,
    )

    assert len(report.candidates) == 4
    components = {candidate.mutation_plan.component for candidate in report.candidates}
    assert "condition" in components
    assert "action" in components
    assert all(
        isinstance(candidate, PolicyMutationCandidate)
        for candidate in report.candidates
    )


def test_to_audit_record_is_json_safe() -> None:
    report = run_offline_evolutionary_policy_dsl_search(
        _sample_dsl(),
        generation_count=1,
        population_size=2,
        mutation_step=0.01,
    )

    encoded_report = json.dumps(report.to_audit_record(), allow_nan=False)
    recovered_report = json.loads(encoded_report)
    assert isinstance(recovered_report["report_hash"], str)

    for candidate in report.candidates:
        encoded_candidate = json.dumps(candidate.to_audit_record(), allow_nan=False)
        assert isinstance(json.loads(encoded_candidate)["candidate_id"], str)


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="policy_dsl must be a non-empty string"):
        run_offline_evolutionary_policy_dsl_search(
            "",
            generation_count=1,
            population_size=1,
        )

    with pytest.raises(ValueError, match="Malformed rule line"):
        run_offline_evolutionary_policy_dsl_search(
            "bad-rule-format",
            generation_count=1,
            population_size=1,
        )

    with pytest.raises(ValueError, match="Malformed condition"):
        run_offline_evolutionary_policy_dsl_search(
            "rule test: if R <> 0.5 then set K += 0.1",
            generation_count=1,
            population_size=1,
        )

    with pytest.raises(ValueError, match="Malformed action"):
        run_offline_evolutionary_policy_dsl_search(
            "rule test: if R < 0.5 then raise K += 0.1",
            generation_count=1,
            population_size=1,
        )

    with pytest.raises(ValueError, match="generation_count must be a positive integer"):
        run_offline_evolutionary_policy_dsl_search(
            _sample_dsl(),
            generation_count=0,
            population_size=1,
        )


def test_search_results_are_review_only_and_non_actuating() -> None:
    report = run_offline_evolutionary_policy_dsl_search(
        _sample_dsl(),
        generation_count=1,
        population_size=2,
        mutation_step=0.01,
    )

    assert report.operator_review_required is True
    assert report.execution_disabled is True
    assert report.non_actuating is True
    assert report.live_merge_permitted is False
    assert report.hot_patch_permitted is False
    assert report.actuation_permitted is False

    assert all(candidate.operator_review_required for candidate in report.candidates)
    assert all(candidate.execution_disabled for candidate in report.candidates)
    assert all(not candidate.live_merge_permitted for candidate in report.candidates)
    assert all(not candidate.hot_patch_permitted for candidate in report.candidates)
    assert all(not candidate.actuation_permitted for candidate in report.candidates)
