# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for offline evolutionary supervisor policy search

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from scpn_phase_orchestrator.supervisor.evolutionary_search import (
    EvolutionaryCandidate,
    EvolutionarySearchConfig,
    EvolutionarySearchReport,
    run_offline_evolutionary_supervisor_search,
)


@pytest.fixture
def _parent_policy() -> dict[str, float]:
    return {
        "K": 0.42,
        "alpha": 0.18,
        "zeta": 0.09,
        "eta": 0.12,
    }


@pytest.fixture
def _audit_replays() -> list[dict[str, object]]:
    return [
        {
            "replay_id": "nominal",
            "reward": 0.86,
            "safety_margin": 0.11,
            "violations": [],
        },
        {
            "replay_id": "disturbance",
            "reward": 0.90,
            "safety_margin": 0.14,
            "violations": [],
        },
    ]


def _trace() -> dict[str, list[float]]:
    return {
        "R": [0.99, 0.98, 0.97, 0.96],
    }


def test_offline_evolutionary_search_is_deterministic_and_hard_gated() -> None:
    safe = run_offline_evolutionary_supervisor_search(
        {"K": 0.4},
        [
            {
                "replay_id": "r1",
                "reward": 0.95,
                "safety_margin": 0.12,
                "violations": [],
            }
        ],
        stl_spec="always (R >= 0.9)",
        trace={"R": [0.95, 0.96]},
        generation_count=2,
        population_size=4,
        mutation_step=0.03,
    )
    same = run_offline_evolutionary_supervisor_search(
        {"K": 0.4},
        [
            {
                "replay_id": "r1",
                "reward": 0.95,
                "safety_margin": 0.12,
                "violations": [],
            }
        ],
        stl_spec="always (R >= 0.9)",
        trace={"R": [0.95, 0.96]},
        generation_count=2,
        population_size=4,
        mutation_step=0.03,
    )

    assert safe == same
    assert isinstance(safe.to_audit_record(), dict)
    assert safe.report_hash == same.report_hash
    assert len(safe.report_hash) == 64
    assert safe.non_actuating is True
    assert safe.execution_disabled is True
    assert safe.hot_patch_permitted is False
    assert safe.live_merge_permitted is False
    assert safe.operator_review_required is True


def test_search_returns_mixed_candidates_with_counterfactual_rejects(
    _parent_policy: dict[str, float],
) -> None:
    replays = [
        {
            "replay_id": "nominal",
            "reward": 0.95,
            "safety_margin": 0.02,
            "violations": [],
        },
        {
            "replay_id": "disturbance",
            "reward": 0.94,
            "safety_margin": 0.03,
            "violations": [],
        },
    ]

    report = run_offline_evolutionary_supervisor_search(
        _parent_policy,
        replays,
        stl_spec="always (R >= 0.8)",
        trace=_trace(),
        generation_count=1,
        population_size=6,
        mutation_step=0.05,
        minimum_replay_reward=0.6,
        minimum_safety_margin=0.0,
    )

    assert report.accepted_count > 0
    assert report.rejected_count > 0
    assert all(
        isinstance(candidate, EvolutionaryCandidate) for candidate in report.candidates
    )
    assert any(
        "counterfactual_safety_delta_exceeds_replay_margin" in candidate.blocked_reasons
        for candidate in report.candidates
    )


def test_stl_violation_blocks_all_candidates(
    _parent_policy: dict[str, float], _audit_replays: list[dict[str, object]]
) -> None:
    report = run_offline_evolutionary_supervisor_search(
        _parent_policy,
        _audit_replays,
        stl_spec="always (R >= 1.5)",
        trace=_trace(),
        generation_count=1,
        population_size=3,
        mutation_step=0.03,
    )

    assert report.accepted_count == 0
    assert report.rejected_count == report.candidate_count
    assert all(
        "stl_spec_not_satisfied" in candidate.blocked_reasons
        for candidate in report.candidates
    )


def test_replay_violations_block_candidates(
    _parent_policy: dict[str, float],
) -> None:
    replays = [
        {
            "replay_id": "bad",
            "reward": 0.95,
            "safety_margin": 0.20,
            "violations": ["unsafe_policy_step"],
        }
    ]

    report = run_offline_evolutionary_supervisor_search(
        _parent_policy,
        replays,
        stl_spec="always (R >= 0.8)",
        trace=_trace(),
        generation_count=1,
        population_size=2,
        mutation_step=0.02,
    )

    assert report.accepted_count == 0
    assert all(
        "replay_violations_present" in candidate.blocked_reasons
        for candidate in report.candidates
    )


def test_input_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="parent_policy must be non-empty"):
        run_offline_evolutionary_supervisor_search(
            {},
            [
                {
                    "replay_id": "r1",
                    "reward": 0.5,
                    "safety_margin": 0.1,
                    "violations": [],
                }
            ],
            stl_spec="always (R >= 0.8)",
            trace={"R": [0.95]},
        )

    with pytest.raises(ValueError, match="audit_replays"):
        run_offline_evolutionary_supervisor_search(
            {"K": 0.1},
            [],
            stl_spec="always (R >= 0.8)",
            trace={"R": [0.95]},
        )

    with pytest.raises(ValueError, match="parent_policy"):
        run_offline_evolutionary_supervisor_search(
            {"K": "bad"},
            [
                {
                    "replay_id": "r1",
                    "reward": 0.5,
                    "safety_margin": 0.1,
                    "violations": [],
                }
            ],
            stl_spec="always (R >= 0.8)",
            trace={"R": [0.95]},
        )

    with pytest.raises(ValueError, match="generation_count"):
        run_offline_evolutionary_supervisor_search(
            {"K": 0.1},
            [
                {
                    "replay_id": "r1",
                    "reward": 0.5,
                    "safety_margin": 0.1,
                    "violations": [],
                }
            ],
            stl_spec="always (R >= 0.8)",
            trace={"R": [0.95]},
            generation_count=0,
        )

    with pytest.raises(ValueError, match="stl_spec"):
        run_offline_evolutionary_supervisor_search(
            {"K": 0.1},
            [
                {
                    "replay_id": "r1",
                    "reward": 0.5,
                    "safety_margin": 0.1,
                    "violations": [],
                }
            ],
            stl_spec=" ",
            trace={"R": [0.95]},
        )


def test_stl_evaluation_validation_errors_are_mapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_value_error(
        self: object, trace: Mapping[str, Sequence[object]]
    ) -> float:
        del self, trace
        raise ValueError("invalid trace payload")

    monkeypatch.setattr(
        "scpn_phase_orchestrator.supervisor.evolutionary_search.STLMonitor.evaluate",
        _raise_value_error,
    )
    with pytest.raises(
        ValueError, match="stl_spec and trace must be valid for offline monitoring"
    ):
        run_offline_evolutionary_supervisor_search(
            {"K": 0.1},
            [
                {
                    "replay_id": "r1",
                    "reward": 0.5,
                    "safety_margin": 0.1,
                    "violations": [],
                }
            ],
            stl_spec="always (R >= 0.8)",
            trace={"R": [0.95]},
        )


def test_stl_evaluation_unexpected_runtime_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_runtime_error(
        self: object, trace: Mapping[str, Sequence[object]]
    ) -> float:
        del self, trace
        raise RuntimeError("unexpected evaluator fault")

    monkeypatch.setattr(
        "scpn_phase_orchestrator.supervisor.evolutionary_search.STLMonitor.evaluate",
        _raise_runtime_error,
    )
    with pytest.raises(RuntimeError, match="unexpected evaluator fault"):
        run_offline_evolutionary_supervisor_search(
            {"K": 0.1},
            [
                {
                    "replay_id": "r1",
                    "reward": 0.5,
                    "safety_margin": 0.1,
                    "violations": [],
                }
            ],
            stl_spec="always (R >= 0.8)",
            trace={"R": [0.95]},
        )


def test_audit_record_is_json_safe_and_serialisable(
    _parent_policy: dict[str, float], _audit_replays: list[dict[str, object]]
) -> None:
    report = run_offline_evolutionary_supervisor_search(
        _parent_policy,
        _audit_replays,
        stl_spec="always (R >= 0.8)",
        trace=_trace(),
        generation_count=1,
        population_size=2,
        mutation_step=0.04,
    )

    record = report.to_audit_record()
    json.loads(json.dumps(record, allow_nan=False))

    assert record["non_actuating"] is True
    assert record["execution_disabled"] is True
    assert record["hot_patch_permitted"] is False
    assert record["live_merge_permitted"] is False
    assert record["operator_review_required"] is True
    assert isinstance(record["candidates"], list)
    assert all(
        "candidate_hash" in candidate and len(candidate["candidate_hash"]) == 64
        for candidate in record["candidates"]
    )
    assert (
        record["best_candidate"] is None or "candidate_hash" in record["best_candidate"]
    )
    assert isinstance(report, EvolutionarySearchReport)
    assert isinstance(report.config, EvolutionarySearchConfig)
