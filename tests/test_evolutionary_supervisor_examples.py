# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Evolutionary supervisor example tests

from __future__ import annotations

import copy
import json

import pytest

import scpn_phase_orchestrator.supervisor.evolutionary_examples as examples


def _records() -> tuple[dict[str, object], ...]:
    return examples.build_evolutionary_supervisor_search_examples()


def test_examples_are_deterministic_json_safe_and_domain_complete() -> None:
    baseline = _records()
    repeat = examples.build_evolutionary_supervisor_search_examples()

    assert baseline == repeat
    assert len(baseline) >= 3
    assert {"power_grid", "cardiac_rhythm", "cyber_industrial", "traffic_flow"} <= {
        record["domain"] for record in baseline
    }

    for record in baseline:
        examples._validate_evolutionary_supervisor_search_record(record)
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        assert json.loads(payload)
        assert record["operator_review_required"] is True
        assert record["execution_disabled"] is True
        assert record["hot_patch_permitted"] is False
        assert record["live_merge_permitted"] is False
        assert record["actuation_permitted"] is False
        assert record["claim_boundary"] == examples.EvolutionaryBoundary
        assert len(str(record["scenario_hash"])) == 64


def test_examples_exercise_safe_and_rejected_replay_conditions() -> None:
    records = _records()

    assert any(
        all(not replay["violations"] for replay in record["audit_replays"])
        for record in records
    )
    cyber = next(record for record in records if record["domain"] == "cyber_industrial")
    assert any(replay["violations"] for replay in cyber["audit_replays"])
    assert cyber["minimum_replay_reward"] > max(
        float(replay["reward"]) for replay in cyber["audit_replays"]
    )


def test_validate_record_rejects_malformed_examples() -> None:
    baseline = _records()

    missing_hash = copy.deepcopy(baseline[0])
    missing_hash.pop("scenario_hash", None)
    with pytest.raises(ValueError, match="scenario_hash"):
        examples._validate_evolutionary_supervisor_search_record(missing_hash)

    bad_hash = copy.deepcopy(baseline[0])
    bad_hash["scenario_hash"] = "not-a-valid-hash"
    with pytest.raises(ValueError, match="scenario_hash"):
        examples._validate_evolutionary_supervisor_search_record(bad_hash)

    bad_gate = copy.deepcopy(baseline[0])
    bad_gate["hot_patch_permitted"] = True
    with pytest.raises(ValueError, match="hot_patch_permitted"):
        examples._validate_evolutionary_supervisor_search_record(bad_gate)


def test_worker_a_api_enrichment_runs_offline_search() -> None:
    records = examples.build_evolutionary_supervisor_search_examples_from_worker_a_api()

    assert len(records) == len(_records())
    for record in records:
        examples._validate_evolutionary_supervisor_search_record(record)
        assert int(record["candidate_count"]) >= int(record["population_size"])
        assert len(str(record["report_hash"])) == 64

    cyber = next(record for record in records if record["domain"] == "cyber_industrial")
    assert cyber["accepted_candidate_count"] == 0
    assert cyber["rejected_candidate_count"] == cyber["candidate_count"]
