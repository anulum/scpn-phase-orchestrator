# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for autopoietic lineage sandbox

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.supervisor.lineage import (
    build_autopoietic_lineage_replay_corpus,
    build_autopoietic_lineage_sandbox,
    build_intergenerational_policy_inheritance,
    build_intergenerational_policy_inheritance_history,
)


def test_autopoietic_lineage_sandbox_is_deterministic_and_review_only() -> None:
    parent_policy = {
        "K": 0.42,
        "alpha": 0.18,
        "zeta": 0.09,
    }
    audit_replays = build_autopoietic_lineage_replay_corpus()

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
    assert first["execution_disabled"] is True
    assert first["live_merge_permitted"] is False
    assert first["hot_patch_permitted"] is False
    assert first["actuation_permitted"] is False
    assert first["replay_corpus_count"] == len(audit_replays)
    assert first["replay_domain_count"] == 4
    assert first["replay_domains"] == (
        "cardiac_rhythm",
        "cyber_industrial",
        "power_grid",
        "traffic_flow",
    )
    assert len(str(first["lineage_sha256"])) == 64
    assert len(str(first["replay_corpus_sha256"])) == 64
    assert all(
        candidate["execution_disabled"] is True
        and candidate["hot_patch_permitted"] is False
        for candidate in first["child_candidates"]
    )
    assert all(
        candidate["policy_diff"]
        for candidate in first["child_candidates"]
        if candidate["status"] == "accepted_for_review"
    )


def test_autopoietic_lineage_replay_corpus_is_curated_and_deterministic() -> None:
    first = build_autopoietic_lineage_replay_corpus()
    second = build_autopoietic_lineage_replay_corpus()

    assert first == second
    assert len(first) == 4
    assert {row["domain"] for row in first} == {
        "cardiac_rhythm",
        "cyber_industrial",
        "power_grid",
        "traffic_flow",
    }
    assert all(row["reward"] >= 0.7 for row in first)
    assert all(row["safety_margin"] >= 0.1 for row in first)
    assert all(row["violations"] == [] for row in first)


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


def test_intergenerational_policy_inheritance_is_signed_and_review_only() -> None:
    lineage = build_autopoietic_lineage_sandbox(
        {"K": 0.42, "alpha": 0.18, "zeta": 0.09},
        [{"replay_id": "nominal", "reward": 0.82, "safety_margin": 0.24}],
        child_budget=2,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    child = lineage["child_candidates"][0]

    first = build_intergenerational_policy_inheritance(
        lineage,
        child,
        signer_id="operator-review-key",
        signing_key="local-test-key",
        objective_weights={"reward": 0.6, "safety": 0.3, "simplicity": 0.1},
    )
    second = build_intergenerational_policy_inheritance(
        lineage,
        child,
        signer_id="operator-review-key",
        signing_key="local-test-key",
        objective_weights={"reward": 0.6, "safety": 0.3, "simplicity": 0.1},
    )

    assert first == second
    assert first["schema"] == "scpn_intergenerational_policy_inheritance_v1"
    assert first["signed_metadata"]["signer_id"] == "operator-review-key"
    assert len(str(first["signed_metadata"]["signature_sha256"])) == 64
    assert len(str(first["inheritance_sha256"])) == 64
    assert first["inherited_policy_genome"]["K"] == pytest.approx(0.44)
    assert first["multi_objective_replay_fitness"]["fitness_score"] > 0.0
    assert first["hot_patch_review_required"] is True
    assert first["direct_hot_patch_permitted"] is False
    assert first["actuation_permitted"] is False
    assert first["merge_strategy"] == "reviewed_hot_patch_only"


def test_intergenerational_policy_inheritance_history_preserves_review_records() -> (
    None
):
    lineage = build_autopoietic_lineage_sandbox(
        {"K": 0.42, "alpha": 0.18, "zeta": 0.09},
        build_autopoietic_lineage_replay_corpus(),
        child_budget=2,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )
    inheritances = [
        build_intergenerational_policy_inheritance(
            lineage,
            child,
            signer_id="operator-review-key",
            signing_key="local-test-key",
            objective_weights={"reward": 0.6, "safety": 0.3, "simplicity": 0.1},
        )
        for child in lineage["child_candidates"]
        if child["status"] == "accepted_for_review"
    ]

    first = build_intergenerational_policy_inheritance_history(lineage, inheritances)
    second = build_intergenerational_policy_inheritance_history(lineage, inheritances)

    assert first == second
    assert first["schema"] == "scpn_intergenerational_policy_inheritance_history_v1"
    assert first["history_record_count"] == len(inheritances)
    assert first["signed_metadata_count"] == len(inheritances)
    assert first["replay_domain_count"] == 4
    assert first["direct_hot_patch_permitted"] is False
    assert first["actuation_permitted"] is False
    assert first["operator_review_required"] is True
    assert len(str(first["history_sha256"])) == 64
    assert [row["generation_index"] for row in first["child_rows"]] == [0, 1]
    assert all(row["fitness_score"] > 0.0 for row in first["child_rows"])


def test_intergenerational_policy_inheritance_rejects_unreviewed_children() -> None:
    lineage = build_autopoietic_lineage_sandbox(
        {"K": 0.42, "alpha": 0.18},
        [
            {
                "replay_id": "unsafe",
                "reward": 0.2,
                "safety_margin": 0.01,
                "violations": ["stl_margin_breach"],
            }
        ],
        child_budget=1,
        mutation_step=0.02,
        minimum_replay_reward=0.7,
        minimum_safety_margin=0.1,
    )

    with pytest.raises(ValueError, match="accepted_for_review"):
        build_intergenerational_policy_inheritance(
            lineage,
            lineage["child_candidates"][0],
            signer_id="operator-review-key",
            signing_key="local-test-key",
        )


def test_intergenerational_policy_inheritance_rejects_bad_signature_inputs() -> None:
    lineage = build_autopoietic_lineage_sandbox(
        {"K": 0.42},
        [{"reward": 0.82, "safety_margin": 0.24}],
        child_budget=1,
        mutation_step=0.02,
    )

    with pytest.raises(ValueError, match="signer_id"):
        build_intergenerational_policy_inheritance(
            lineage,
            lineage["child_candidates"][0],
            signer_id="",
            signing_key="local-test-key",
        )

    with pytest.raises(ValueError, match="signing_key"):
        build_intergenerational_policy_inheritance(
            lineage,
            lineage["child_candidates"][0],
            signer_id="operator-review-key",
            signing_key="",
        )
