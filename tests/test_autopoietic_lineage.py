# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for autopoietic lineage sandbox

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.supervisor import build_autopoietic_lineage_sandbox


def test_autopoietic_lineage_sandbox_is_deterministic_and_review_only() -> None:
    parent_policy = {
        "K": 0.42,
        "alpha": 0.18,
        "zeta": 0.09,
    }
    audit_replays = [
        {
            "replay_id": "nominal_grid_replay",
            "reward": 0.82,
            "safety_margin": 0.24,
            "violations": [],
        },
        {
            "replay_id": "disturbance_grid_replay",
            "reward": 0.74,
            "safety_margin": 0.18,
            "violations": [],
        },
    ]

    first = build_autopoietic_lineage_sandbox(
        parent_policy,
        audit_replays,
        child_budget=3,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    second = build_autopoietic_lineage_sandbox(
        parent_policy,
        audit_replays,
        child_budget=3,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )

    assert first == second
    assert first["schema"] == "scpn_autopoietic_lineage_sandbox_v1"
    assert first["child_budget"] == 3
    assert first["child_candidate_count"] == 3
    assert first["accepted_child_count"] >= 1
    assert first["review_required"] is True
    assert first["live_merge_permitted"] is False
    assert first["actuation_permitted"] is False
    assert len(str(first["lineage_sha256"])) == 64
    assert all(
        candidate["policy_diff"]
        for candidate in first["child_candidates"]
        if candidate["status"] == "accepted_for_review"
    )


def test_autopoietic_lineage_sandbox_rejects_unsafe_child_candidates() -> None:
    manifest = build_autopoietic_lineage_sandbox(
        {"K": 0.8, "alpha": 0.4},
        [
            {
                "replay_id": "unsafe_replay",
                "reward": 0.3,
                "safety_margin": 0.02,
                "violations": ["stl_margin_breach"],
            }
        ],
        child_budget=2,
        mutation_step=0.1,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )

    assert manifest["accepted_child_count"] == 0
    assert manifest["rejected_child_count"] == 2
    assert {candidate["status"] for candidate in manifest["child_candidates"]} == {
        "rejected"
    }
    assert all(
        "replay_reward_below_minimum" in candidate["blocked_reasons"]
        for candidate in manifest["child_candidates"]
    )
    assert all(
        "safety_margin_below_minimum" in candidate["blocked_reasons"]
        for candidate in manifest["child_candidates"]
    )
    assert all(
        "replay_violations_present" in candidate["blocked_reasons"]
        for candidate in manifest["child_candidates"]
    )


def test_autopoietic_lineage_sandbox_rejects_invalid_resource_bounds() -> None:
    with pytest.raises(ValueError, match="child_budget"):
        build_autopoietic_lineage_sandbox({"K": 0.1}, [], child_budget=0)

    with pytest.raises(ValueError, match="audit_replays"):
        build_autopoietic_lineage_sandbox({"K": 0.1}, [], child_budget=1)

    with pytest.raises(ValueError, match="numeric"):
        build_autopoietic_lineage_sandbox({"K": "high"}, [{"reward": 1.0}])

    with pytest.raises(ValueError, match="mutation_step"):
        build_autopoietic_lineage_sandbox(
            {"K": 0.1},
            [{"reward": 1.0}],
            mutation_step=0.0,
        )
